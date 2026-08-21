"""Incremental KNN classifier with validation and safe persistence."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DistanceMode = Literal["cosine", "standardized_euclidean"]
ThresholdMode = Literal["auto", "fixed"]
DecisionMode = Literal["class_balanced", "uniform_knn"]
CalibrationSource = Literal["fallback", "registration", "validation"]
RejectionReason = Literal["distance", "ambiguous", "untrained"]
DISTANCE_MODES = frozenset({"cosine", "standardized_euclidean"})
THRESHOLD_MODES = frozenset({"auto", "fixed"})
DECISION_MODES = frozenset({"class_balanced", "uniform_knn"})
MIN_AUTO_SAMPLES = 5
MIN_COSINE_THRESHOLD = 0.01
AUTO_PERCENTILE = 90
AUTO_MARGIN = 1.10
DEFAULT_SEPARATION_THRESHOLD = 0.15
MAX_SEPARATION_THRESHOLD = 0.50
SEPARATION_PERCENTILE = 10
SEPARATION_MARGIN = 0.90
SEPARATION_EPSILON = 1e-12


@dataclass(frozen=True)
class ThresholdCalibration:
    """Describe how one label's rejection threshold was obtained."""

    threshold: float
    sample_count: int
    median_distance: float | None
    max_distance: float | None
    source: CalibrationSource

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class SeparationCalibration:
    """Describe how one label's inter-class separation threshold was obtained."""

    threshold: float
    sample_count: int
    classification_accuracy: float | None
    median_separation: float | None
    source: CalibrationSource

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationDecision:
    """Capture classification scores and both rejection decisions."""

    predicted_label: str | None
    runner_up_label: str | None
    best_score: float | None
    runner_up_score: float | None
    nearest_distance: float | None
    separation: float | None
    distance_threshold: float | None
    separation_threshold: float | None
    accepted: bool
    rejection_reason: RejectionReason | None


def validate_distance_mode(distance_mode: str) -> DistanceMode:
    """Return a supported distance mode or raise a configuration error."""
    if distance_mode not in DISTANCE_MODES:
        choices = ", ".join(sorted(DISTANCE_MODES))
        raise ValueError(f"distance_mode must be one of: {choices}")
    return cast(DistanceMode, distance_mode)


def validate_threshold_mode(threshold_mode: str) -> ThresholdMode:
    """Return a supported threshold mode or raise a configuration error."""
    if threshold_mode not in THRESHOLD_MODES:
        choices = ", ".join(sorted(THRESHOLD_MODES))
        raise ValueError(f"threshold_mode must be one of: {choices}")
    return cast(ThresholdMode, threshold_mode)


def validate_decision_mode(decision_mode: str) -> DecisionMode:
    """Return a supported decision mode or raise a configuration error."""
    if decision_mode not in DECISION_MODES:
        choices = ", ".join(sorted(DECISION_MODES))
        raise ValueError(f"decision_mode must be one of: {choices}")
    return cast(DecisionMode, decision_mode)


def normalize_feature_matrix(features: np.ndarray, name: str) -> np.ndarray:
    """L2-normalize feature rows for cosine distance."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero-norm feature vector")
    return matrix / norms


class TapClassifier:
    """Incrementally learn labeled feature vectors and reject distant inputs."""

    SCHEMA_VERSION = 4

    def __init__(
        self,
        n_neighbors: int = 3,
        distance_threshold: float = 2.5,
        distance_mode: DistanceMode = "standardized_euclidean",
        threshold_mode: ThresholdMode = "fixed",
        decision_mode: DecisionMode | None = None,
        separation_threshold: float = DEFAULT_SEPARATION_THRESHOLD,
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if not np.isfinite(distance_threshold) or distance_threshold < 0:
            raise ValueError("distance_threshold must be finite and non-negative")

        self.n_neighbors = n_neighbors
        self.distance_threshold = float(distance_threshold)
        self.distance_mode = validate_distance_mode(distance_mode)
        self.threshold_mode = validate_threshold_mode(threshold_mode)
        selected_decision_mode = decision_mode
        if selected_decision_mode is None:
            selected_decision_mode = (
                "class_balanced" if self.distance_mode == "cosine" else "uniform_knn"
            )
        self.decision_mode = validate_decision_mode(selected_decision_mode)
        if self.decision_mode == "class_balanced" and self.distance_mode != "cosine":
            raise ValueError("class-balanced decisions require cosine distance")
        if not np.isfinite(separation_threshold) or not (
            0 <= separation_threshold <= MAX_SEPARATION_THRESHOLD
        ):
            raise ValueError(
                f"separation_threshold must be between 0 and {MAX_SEPARATION_THRESHOLD}"
            )
        self.separation_threshold = float(separation_threshold)
        if self.threshold_mode == "auto" and self.distance_mode != "cosine":
            raise ValueError("automatic thresholds require cosine distance")
        if self.threshold_mode == "auto" and self.distance_threshold < MIN_COSINE_THRESHOLD:
            raise ValueError(
                f"automatic cosine threshold cap must be at least {MIN_COSINE_THRESHOLD}"
            )
        if self.distance_mode == "cosine":
            self.model = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                metric="cosine",
                algorithm="brute",
            )
            self.scaler: StandardScaler | None = None
        else:
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            self.scaler = StandardScaler()
        self.X_data: list[np.ndarray] = []
        self.y_labels: list[str] = []
        self.is_trained = False
        self.feature_dim: int | None = None
        self.distance_thresholds: dict[str, float] = {}
        self.threshold_calibrations: dict[str, ThresholdCalibration] = {}
        self.separation_thresholds: dict[str, float] = {}
        self.separation_calibrations: dict[str, SeparationCalibration] = {}
        self.metadata: dict[str, Any] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_label(label: str) -> str:
        if not isinstance(label, str):
            raise TypeError("label must be a string")
        normalized = label.strip()
        if not normalized:
            raise ValueError("label must not be empty")
        return normalized

    def _validate_feature(self, feature_vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(feature_vector)
        if vector.ndim != 1:
            raise ValueError(f"feature_vector must be one-dimensional, got {vector.shape}")
        if vector.size == 0:
            raise ValueError("feature_vector must not be empty")
        if not np.issubdtype(vector.dtype, np.number):
            raise TypeError("feature_vector must contain numeric values")
        if not np.isfinite(vector).all():
            raise ValueError("feature_vector contains NaN or infinite values")
        if self.distance_mode == "cosine" and np.linalg.norm(vector) == 0:
            raise ValueError("feature_vector must have nonzero norm for cosine distance")
        if self.feature_dim is not None and vector.size != self.feature_dim:
            raise ValueError(
                f"feature dimension mismatch: expected {self.feature_dim}, got {vector.size}"
            )
        return vector.astype(np.float64, copy=True)

    def _fit_if_ready(self) -> None:
        label_counts = {
            label: self.y_labels.count(label) for label in set(self.y_labels)
        }
        if len(self.X_data) < self.n_neighbors or (
            self.decision_mode == "class_balanced"
            and not any(count >= self.n_neighbors for count in label_counts.values())
        ):
            self.is_trained = False
            return
        x_raw = np.vstack(self.X_data)
        y_train = np.asarray(self.y_labels)
        if self.distance_mode == "standardized_euclidean":
            assert self.scaler is not None
            self.scaler.fit(x_raw)
        x_train = self._transform_matrix(x_raw, "registered features")
        self.model.fit(x_train, y_train)
        self.is_trained = True

    def _transform_matrix(self, matrix: np.ndarray, name: str) -> np.ndarray:
        if self.distance_mode == "cosine":
            return normalize_feature_matrix(matrix, name)
        assert self.scaler is not None
        return self.scaler.transform(matrix)

    def _transform_query(self, vector: np.ndarray) -> np.ndarray:
        return self._transform_matrix(vector.reshape(1, -1), "feature_vector")

    def _nearest_distance(self, vector: np.ndarray) -> float:
        transformed_vector = self._transform_query(vector)
        distances, _ = self.model.kneighbors(transformed_vector, n_neighbors=1)
        return float(distances[0][0])

    def distance_to_label(self, feature_vector: np.ndarray, label: str) -> float:
        """Return distance from a feature to the nearest example of ``label``."""
        with self._lock:
            if not self.is_trained:
                raise RuntimeError("classifier is not trained")
            normalized_label = self._validate_label(label)
            vector = self._validate_feature(feature_vector)
            label_indices = [
                index for index, stored_label in enumerate(self.y_labels)
                if stored_label == normalized_label
            ]
            if not label_indices:
                raise ValueError(f"label has no registered examples: {normalized_label!r}")
            label_features = np.vstack([self.X_data[index] for index in label_indices])
            transformed_label_features = self._transform_matrix(
                label_features, f"registered features for {normalized_label!r}"
            )
            transformed_vector = self._transform_query(vector)
            if self.distance_mode == "cosine":
                distances = 1.0 - transformed_label_features @ transformed_vector[0]
                distances = np.maximum(distances, 0.0)
            else:
                distances = np.linalg.norm(
                    transformed_label_features - transformed_vector,
                    axis=1,
                )
            return float(np.min(distances))

    def _class_scores(
        self,
        vector: np.ndarray,
        excluded_index: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Return each eligible label's mean-k score and nearest distance."""
        transformed_vector = self._transform_query(vector)
        scores: dict[str, tuple[float, float]] = {}
        for label in sorted(set(self.y_labels)):
            label_features = [
                feature
                for index, (feature, stored_label) in enumerate(
                    zip(self.X_data, self.y_labels, strict=True)
                )
                if stored_label == label and index != excluded_index
            ]
            if len(label_features) < self.n_neighbors:
                continue
            transformed = normalize_feature_matrix(
                np.vstack(label_features), f"registered features for {label!r}"
            )
            distances = np.maximum(1.0 - transformed @ transformed_vector[0], 0.0)
            nearest = np.partition(distances, self.n_neighbors - 1)[: self.n_neighbors]
            scores[label] = (float(np.mean(nearest)), float(np.min(nearest)))
        return scores

    @staticmethod
    def _normalized_separation(best_score: float, runner_up_score: float) -> float:
        return float(
            (runner_up_score - best_score)
            / max(runner_up_score, SEPARATION_EPSILON)
        )

    def threshold_for_label(self, label: str) -> float:
        """Return the calibrated threshold or global fallback for ``label``."""
        normalized_label = self._validate_label(label)
        return self.distance_thresholds.get(normalized_label, self.distance_threshold)

    def separation_threshold_for_label(self, label: str) -> float:
        """Return the calibrated inter-class margin for ``label``."""
        normalized_label = self._validate_label(label)
        return self.separation_thresholds.get(
            normalized_label, self.separation_threshold
        )

    def _set_fallback_calibration(self, label: str) -> None:
        sample_count = self.y_labels.count(label)
        self.distance_thresholds[label] = self.distance_threshold
        self.threshold_calibrations[label] = ThresholdCalibration(
            threshold=self.distance_threshold,
            sample_count=sample_count,
            median_distance=None,
            max_distance=None,
            source="fallback",
        )

    def _calibrate_label_from_registrations(self, label: str) -> None:
        label_features = np.vstack(
            [
                feature
                for feature, stored_label in zip(self.X_data, self.y_labels, strict=True)
                if stored_label == label
            ]
        )
        sample_count = len(label_features)
        if self.threshold_mode == "fixed" or sample_count < MIN_AUTO_SAMPLES:
            self._set_fallback_calibration(label)
            return

        transformed = normalize_feature_matrix(
            label_features, f"registered features for {label!r}"
        )
        pairwise = np.maximum(1.0 - transformed @ transformed.T, 0.0)
        np.fill_diagonal(pairwise, np.inf)
        leave_one_out = np.min(pairwise, axis=1)
        percentile = float(
            np.percentile(leave_one_out, AUTO_PERCENTILE, method="higher")
        )
        threshold = min(
            max(percentile * AUTO_MARGIN, MIN_COSINE_THRESHOLD),
            self.distance_threshold,
        )
        self.distance_thresholds[label] = threshold
        self.threshold_calibrations[label] = ThresholdCalibration(
            threshold=threshold,
            sample_count=sample_count,
            median_distance=float(np.median(leave_one_out)),
            max_distance=float(np.max(leave_one_out)),
            source="registration",
        )

    def set_validation_calibration(
        self,
        label: str,
        threshold: float,
        distances: np.ndarray,
    ) -> None:
        """Install a validation-derived threshold for a registered label."""
        with self._lock:
            if self.threshold_mode != "auto":
                raise ValueError("validation calibration requires automatic threshold mode")
            normalized_label = self._validate_label(label)
            if normalized_label not in self.y_labels:
                raise ValueError(f"label has no registered examples: {normalized_label!r}")
            if not np.isfinite(threshold) or not (
                MIN_COSINE_THRESHOLD <= threshold <= self.distance_threshold
            ):
                raise ValueError(
                    "validation threshold must be finite and within the configured cap"
                )
            values = np.asarray(distances, dtype=np.float64)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError("validation distances must be a finite non-empty vector")
            selected_threshold = float(threshold)
            self.distance_thresholds[normalized_label] = selected_threshold
            self.threshold_calibrations[normalized_label] = ThresholdCalibration(
                threshold=selected_threshold,
                sample_count=int(values.size),
                median_distance=float(np.median(values)),
                max_distance=float(np.max(values)),
                source="validation",
            )

    def _set_separation_fallback(self, label: str) -> None:
        sample_count = self.y_labels.count(label)
        self.separation_thresholds[label] = self.separation_threshold
        self.separation_calibrations[label] = SeparationCalibration(
            threshold=self.separation_threshold,
            sample_count=sample_count,
            classification_accuracy=None,
            median_separation=None,
            source="fallback",
        )

    def _calibrate_all_separations_from_registrations(self) -> None:
        labels = sorted(set(self.y_labels))
        if self.decision_mode != "class_balanced":
            return
        minimum_samples = max(MIN_AUTO_SAMPLES, self.n_neighbors + 1)
        for label in labels:
            label_indices = [
                index
                for index, stored_label in enumerate(self.y_labels)
                if stored_label == label
            ]
            if len(label_indices) < minimum_samples:
                self._set_separation_fallback(label)
                continue
            separations: list[float] = []
            correct = 0
            calibration_available = True
            for index in label_indices:
                scores = self._class_scores(self.X_data[index], excluded_index=index)
                if label not in scores or len(scores) < 2:
                    calibration_available = False
                    break
                ranked = sorted(
                    (score, candidate_label)
                    for candidate_label, (score, _nearest) in scores.items()
                )
                best_score, predicted_label = ranked[0]
                runner_up_score, _runner_up_label = ranked[1]
                if predicted_label == label:
                    correct += 1
                    separations.append(
                        self._normalized_separation(best_score, runner_up_score)
                    )
                else:
                    separations.append(0.0)
            if not calibration_available:
                self._set_separation_fallback(label)
                continue
            values = np.asarray(separations, dtype=np.float64)
            percentile = float(
                np.percentile(values, SEPARATION_PERCENTILE, method="lower")
            )
            threshold = min(
                max(percentile * SEPARATION_MARGIN, self.separation_threshold),
                MAX_SEPARATION_THRESHOLD,
            )
            self.separation_thresholds[label] = threshold
            self.separation_calibrations[label] = SeparationCalibration(
                threshold=threshold,
                sample_count=len(label_indices),
                classification_accuracy=correct / len(label_indices),
                median_separation=float(np.median(values)),
                source="registration",
            )

    def set_validation_separation_calibration(
        self,
        label: str,
        threshold: float,
        separations: np.ndarray,
        accuracy: float,
    ) -> None:
        """Install a validation-derived separation threshold for one label."""
        with self._lock:
            normalized_label = self._validate_label(label)
            if normalized_label not in self.y_labels:
                raise ValueError(f"label has no registered examples: {normalized_label!r}")
            if not np.isfinite(threshold) or not (
                self.separation_threshold <= threshold <= MAX_SEPARATION_THRESHOLD
            ):
                raise ValueError(
                    "validation separation must be within the configured limits"
                )
            values = np.asarray(separations, dtype=np.float64)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError("validation separations must be a finite non-empty vector")
            if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError("validation accuracy must be between zero and one")
            selected_threshold = float(threshold)
            self.separation_thresholds[normalized_label] = selected_threshold
            self.separation_calibrations[normalized_label] = SeparationCalibration(
                threshold=selected_threshold,
                sample_count=int(values.size),
                classification_accuracy=float(accuracy),
                median_separation=float(np.median(values)),
                source="validation",
            )

    def classify(
        self,
        feature_vector: np.ndarray,
        *,
        apply_rejection: bool = True,
    ) -> ClassificationDecision:
        """Return the complete classification and rejection decision."""
        with self._lock:
            if not self.is_trained:
                return ClassificationDecision(
                    predicted_label=None,
                    runner_up_label=None,
                    best_score=None,
                    runner_up_score=None,
                    nearest_distance=None,
                    separation=None,
                    distance_threshold=None,
                    separation_threshold=None,
                    accepted=False,
                    rejection_reason="untrained",
                )
            vector = self._validate_feature(feature_vector)
            if self.decision_mode == "class_balanced":
                class_scores = self._class_scores(vector)
                if not class_scores:
                    return ClassificationDecision(
                        predicted_label=None,
                        runner_up_label=None,
                        best_score=None,
                        runner_up_score=None,
                        nearest_distance=None,
                        separation=None,
                        distance_threshold=None,
                        separation_threshold=None,
                        accepted=False,
                        rejection_reason="untrained",
                    )
                ranked = sorted(
                    (
                        (score, label, nearest)
                        for label, (score, nearest) in class_scores.items()
                    )
                )
                best_score, prediction, closest_distance = ranked[0]
                if len(ranked) > 1:
                    runner_up_score, runner_up_label, _runner_nearest = ranked[1]
                    separation = self._normalized_separation(
                        best_score, runner_up_score
                    )
                else:
                    runner_up_label = None
                    runner_up_score = None
                    separation = None
            else:
                transformed_vector = self._transform_query(vector)
                prediction = str(self.model.predict(transformed_vector)[0])
                closest_distance = self.distance_to_label(vector, prediction)
                best_score = closest_distance
                runner_up_label = None
                runner_up_score = None
                separation = None

            distance_threshold = self.threshold_for_label(prediction)
            separation_threshold = self.separation_threshold_for_label(prediction)
            distance_accepted = closest_distance <= distance_threshold or np.isclose(
                closest_distance,
                distance_threshold,
                rtol=1e-9,
                atol=1e-12,
            )
            separation_accepted = (
                separation is None
                or separation >= separation_threshold
                or np.isclose(
                    separation,
                    separation_threshold,
                    rtol=1e-9,
                    atol=1e-12,
                )
            )
            rejection_reason: RejectionReason | None = None
            if apply_rejection and not distance_accepted:
                rejection_reason = "distance"
            elif apply_rejection and not separation_accepted:
                rejection_reason = "ambiguous"
            return ClassificationDecision(
                predicted_label=prediction,
                runner_up_label=runner_up_label,
                best_score=best_score,
                runner_up_score=runner_up_score,
                nearest_distance=closest_distance,
                separation=separation,
                distance_threshold=distance_threshold,
                separation_threshold=separation_threshold,
                accepted=rejection_reason is None,
                rejection_reason=rejection_reason,
            )

    def classify_without_rejection(
        self, feature_vector: np.ndarray
    ) -> ClassificationDecision:
        """Return class scores without applying distance or separation rejection."""
        return self.classify(feature_vector, apply_rejection=False)

    def register_tap(self, feature_vector: np.ndarray, label: str) -> None:
        """Add one captured-window example and refit when enough samples exist."""
        with self._lock:
            normalized_label = self._validate_label(label)
            vector = self._validate_feature(feature_vector)
            if self.feature_dim is None:
                self.feature_dim = int(vector.size)
            self.X_data.append(vector)
            self.y_labels.append(normalized_label)
            self._fit_if_ready()
            self._calibrate_label_from_registrations(normalized_label)
            self._calibrate_all_separations_from_registrations()

            if self.is_trained:
                print(f"[*] Model updated. Total registered samples: {len(self.X_data)}")
            else:
                remaining = self.n_neighbors - len(self.X_data)
                print(f"[*] Sample recorded. Need {remaining} more to train.")

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict a label, returning ``Unknown`` for untrained or distant input."""
        with self._lock:
            decision = self.classify(feature_vector)
            if decision.predicted_label is None:
                return "Unknown"
            prediction = decision.predicted_label
            calibration = self.threshold_calibrations.get(prediction)
            calibration_source = "fallback" if calibration is None else calibration.source
            separation_calibration = self.separation_calibrations.get(prediction)
            separation_source = (
                "fallback"
                if separation_calibration is None
                else separation_calibration.source
            )
            if not decision.accepted:
                print(
                    f"[-] Ignored: {decision.rejection_reason} "
                    f"(Predicted: {prediction}, Runner-up: {decision.runner_up_label}, "
                    f"Scores: {decision.best_score:.4f}/{decision.runner_up_score}, "
                    f"Distance: {decision.nearest_distance:.4f}/"
                    f"{decision.distance_threshold:.4f}, Separation: "
                    f"{decision.separation}/{decision.separation_threshold:.4f})"
                )
                return "Unknown"

            print(
                f"[+] Match found: {prediction} (Runner-up: {decision.runner_up_label}, "
                f"Scores: {decision.best_score:.4f}/{decision.runner_up_score}, "
                f"Distance: {decision.nearest_distance:.4f}/"
                f"{decision.distance_threshold:.4f}, Separation: "
                f"{decision.separation}/{decision.separation_threshold:.4f}, "
                f"Distance calibration: {calibration_source}, "
                f"Separation calibration: {separation_source})"
            )
            return prediction

    def save(
        self,
        path: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically persist raw examples and reproducibility metadata as NPZ."""
        with self._lock:
            if not self.X_data:
                raise ValueError("cannot save a classifier with no registered samples")
            destination = Path(path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload_metadata = dict(self.metadata)
            if metadata is not None:
                payload_metadata.update(metadata)
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "n_neighbors": self.n_neighbors,
                "distance_threshold": self.distance_threshold,
                "distance_mode": self.distance_mode,
                "threshold_mode": self.threshold_mode,
                "decision_mode": self.decision_mode,
                "separation_threshold": self.separation_threshold,
                "distance_thresholds": self.distance_thresholds,
                "threshold_calibrations": {
                    label: calibration.to_dict()
                    for label, calibration in self.threshold_calibrations.items()
                },
                "separation_thresholds": self.separation_thresholds,
                "separation_calibrations": {
                    label: calibration.to_dict()
                    for label, calibration in self.separation_calibrations.items()
                },
                "feature_dim": self.feature_dim,
                "metadata": payload_metadata,
            }

            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    np.savez_compressed(
                        temporary_file,
                        features=np.vstack(self.X_data),
                        labels=np.asarray(self.y_labels, dtype=np.str_),
                        manifest=np.asarray(json.dumps(payload, sort_keys=True)),
                    )
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                os.replace(temporary_path, destination)
                temporary_path = None
                self.metadata = payload_metadata
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> TapClassifier:
        """Load a non-pickle NPZ state file and rebuild scaler/KNN state."""
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as persisted:
            required = {"features", "labels", "manifest"}
            if not required.issubset(persisted.files):
                raise ValueError(
                    f"classifier state is missing keys: {required - set(persisted.files)}"
                )
            manifest = json.loads(str(persisted["manifest"].item()))
            schema_version = manifest.get("schema_version")
            if schema_version not in {1, 2, 3, cls.SCHEMA_VERSION}:
                raise ValueError(f"unsupported classifier schema: {schema_version}")
            features = np.asarray(persisted["features"], dtype=np.float64)
            labels = np.asarray(persisted["labels"], dtype=np.str_)

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("persisted features must be a non-empty 2D array")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("persisted labels do not align with features")
        if not np.isfinite(features).all():
            raise ValueError("persisted features contain NaN or infinite values")
        if int(manifest["feature_dim"]) != features.shape[1]:
            raise ValueError("persisted feature dimension does not match manifest")

        metadata = dict(manifest.get("metadata", {}))
        if schema_version == 1:
            if metadata.get("feature_extractor") == "pann":
                raise ValueError(
                    "schema-v1 PANN registrations use the obsolete standardized-Euclidean "
                    "distance; re-register using a new --state-path or remove the old state file"
                )
            distance_mode: DistanceMode = "standardized_euclidean"
            threshold_mode: ThresholdMode = "fixed"
        else:
            if "distance_mode" not in manifest:
                raise ValueError("classifier state is missing distance_mode")
            distance_mode = validate_distance_mode(str(manifest["distance_mode"]))
            if schema_version == 2:
                threshold_mode = "auto" if distance_mode == "cosine" else "fixed"
            else:
                if "threshold_mode" not in manifest:
                    raise ValueError("classifier state is missing threshold_mode")
                threshold_mode = validate_threshold_mode(str(manifest["threshold_mode"]))

        if schema_version < cls.SCHEMA_VERSION:
            decision_mode: DecisionMode = (
                "class_balanced" if distance_mode == "cosine" else "uniform_knn"
            )
            separation_threshold = DEFAULT_SEPARATION_THRESHOLD
        else:
            if "decision_mode" not in manifest or "separation_threshold" not in manifest:
                raise ValueError("classifier state is missing separation configuration")
            decision_mode = validate_decision_mode(str(manifest["decision_mode"]))
            separation_threshold = float(manifest["separation_threshold"])

        classifier = cls(
            n_neighbors=int(manifest["n_neighbors"]),
            distance_threshold=float(manifest["distance_threshold"]),
            distance_mode=distance_mode,
            threshold_mode=threshold_mode,
            decision_mode=decision_mode,
            separation_threshold=separation_threshold,
        )
        classifier.feature_dim = features.shape[1]
        classifier.X_data = [row.copy() for row in features]
        classifier.y_labels = [str(label) for label in labels]
        classifier.metadata = metadata
        classifier._fit_if_ready()
        registered_labels = sorted(set(classifier.y_labels))
        if schema_version < 3:
            for label in registered_labels:
                classifier._calibrate_label_from_registrations(label)
        else:
            persisted_thresholds = manifest.get("distance_thresholds")
            persisted_calibrations = manifest.get("threshold_calibrations")
            if not isinstance(persisted_thresholds, dict) or not isinstance(
                persisted_calibrations, dict
            ):
                raise ValueError("classifier state has invalid threshold calibration data")
            for label in registered_labels:
                threshold = float(persisted_thresholds.get(label, classifier.distance_threshold))
                raw_calibration = persisted_calibrations.get(label)
                if not isinstance(raw_calibration, dict):
                    raise ValueError(f"classifier state is missing calibration for {label!r}")
                calibration_source = str(raw_calibration.get("source"))
                if calibration_source not in {"fallback", "registration", "validation"}:
                    raise ValueError(
                        "classifier state has invalid calibration source: "
                        f"{calibration_source}"
                    )
                calibration = ThresholdCalibration(
                    threshold=float(raw_calibration["threshold"]),
                    sample_count=int(raw_calibration["sample_count"]),
                    median_distance=(
                        None
                        if raw_calibration.get("median_distance") is None
                        else float(raw_calibration["median_distance"])
                    ),
                    max_distance=(
                        None
                        if raw_calibration.get("max_distance") is None
                        else float(raw_calibration["max_distance"])
                    ),
                    source=cast(CalibrationSource, calibration_source),
                )
                if not np.isfinite(threshold) or threshold < 0:
                    raise ValueError(f"classifier state has invalid threshold for {label!r}")
                if classifier.threshold_mode == "auto" and not (
                    MIN_COSINE_THRESHOLD <= threshold <= classifier.distance_threshold
                ):
                    raise ValueError(
                        f"classifier state threshold is outside the auto cap for {label!r}"
                    )
                if not np.isclose(threshold, calibration.threshold):
                    raise ValueError(f"classifier state threshold mismatch for {label!r}")
                classifier.distance_thresholds[label] = threshold
                classifier.threshold_calibrations[label] = calibration
        if classifier.decision_mode != "class_balanced":
            pass
        elif schema_version < cls.SCHEMA_VERSION:
            classifier._calibrate_all_separations_from_registrations()
        else:
            persisted_separation_thresholds = manifest.get("separation_thresholds")
            persisted_separation_calibrations = manifest.get("separation_calibrations")
            if not isinstance(persisted_separation_thresholds, dict) or not isinstance(
                persisted_separation_calibrations, dict
            ):
                raise ValueError("classifier state has invalid separation calibration data")
            for label in registered_labels:
                separation = float(
                    persisted_separation_thresholds.get(
                        label, classifier.separation_threshold
                    )
                )
                raw_separation_calibration = persisted_separation_calibrations.get(label)
                if not isinstance(raw_separation_calibration, dict):
                    raise ValueError(
                        f"classifier state is missing separation calibration for {label!r}"
                    )
                calibration_source = str(raw_separation_calibration.get("source"))
                if calibration_source not in {"fallback", "registration", "validation"}:
                    raise ValueError(
                        "classifier state has invalid separation source: "
                        f"{calibration_source}"
                    )
                separation_calibration = SeparationCalibration(
                    threshold=float(raw_separation_calibration["threshold"]),
                    sample_count=int(raw_separation_calibration["sample_count"]),
                    classification_accuracy=(
                        None
                        if raw_separation_calibration.get("classification_accuracy") is None
                        else float(raw_separation_calibration["classification_accuracy"])
                    ),
                    median_separation=(
                        None
                        if raw_separation_calibration.get("median_separation") is None
                        else float(raw_separation_calibration["median_separation"])
                    ),
                    source=cast(CalibrationSource, calibration_source),
                )
                if not np.isfinite(separation) or not (
                    classifier.separation_threshold
                    <= separation
                    <= MAX_SEPARATION_THRESHOLD
                ):
                    raise ValueError(
                        f"classifier state has invalid separation for {label!r}"
                    )
                if not np.isclose(separation, separation_calibration.threshold):
                    raise ValueError(
                        f"classifier state separation mismatch for {label!r}"
                    )
                classifier.separation_thresholds[label] = separation
                classifier.separation_calibrations[label] = separation_calibration
        return classifier
