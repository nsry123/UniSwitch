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
CalibrationSource = Literal["fallback", "registration", "validation"]
DISTANCE_MODES = frozenset({"cosine", "standardized_euclidean"})
THRESHOLD_MODES = frozenset({"auto", "fixed"})
MIN_AUTO_SAMPLES = 5
MIN_COSINE_THRESHOLD = 0.01
AUTO_PERCENTILE = 90
AUTO_MARGIN = 1.10


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

    SCHEMA_VERSION = 3

    def __init__(
        self,
        n_neighbors: int = 3,
        distance_threshold: float = 2.5,
        distance_mode: DistanceMode = "standardized_euclidean",
        threshold_mode: ThresholdMode = "fixed",
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if not np.isfinite(distance_threshold) or distance_threshold < 0:
            raise ValueError("distance_threshold must be finite and non-negative")

        self.n_neighbors = n_neighbors
        self.distance_threshold = float(distance_threshold)
        self.distance_mode = validate_distance_mode(distance_mode)
        self.threshold_mode = validate_threshold_mode(threshold_mode)
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
        if len(self.X_data) < self.n_neighbors:
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

    def threshold_for_label(self, label: str) -> float:
        """Return the calibrated threshold or global fallback for ``label``."""
        normalized_label = self._validate_label(label)
        return self.distance_thresholds.get(normalized_label, self.distance_threshold)

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

    def classify_without_rejection(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """Return the KNN label and nearest same-label distance before rejection."""
        with self._lock:
            if not self.is_trained:
                raise RuntimeError("classifier is not trained")
            vector = self._validate_feature(feature_vector)
            transformed_vector = self._transform_query(vector)
            prediction = str(self.model.predict(transformed_vector)[0])
            return prediction, self.distance_to_label(vector, prediction)

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

            if self.is_trained:
                print(f"[*] Model updated. Total registered samples: {len(self.X_data)}")
            else:
                remaining = self.n_neighbors - len(self.X_data)
                print(f"[*] Sample recorded. Need {remaining} more to train.")

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict a label, returning ``Unknown`` for untrained or distant input."""
        with self._lock:
            if not self.is_trained:
                return "Unknown"
            prediction, closest_distance = self.classify_without_rejection(feature_vector)
            threshold = self.threshold_for_label(prediction)
            calibration = self.threshold_calibrations.get(prediction)
            calibration_source = "fallback" if calibration is None else calibration.source

            accepted = closest_distance <= threshold or np.isclose(
                closest_distance,
                threshold,
                rtol=1e-9,
                atol=1e-12,
            )
            if not accepted:
                print(
                    f"[-] Ignored: Unregistered sound (Predicted: {prediction}, "
                    f"Distance: {closest_distance:.4f}, Threshold: {threshold:.4f}, "
                    f"Calibration: {calibration_source})"
                )
                return "Unknown"

            print(
                f"[+] Match found: {prediction} (Distance: {closest_distance:.4f}, "
                f"Threshold: {threshold:.4f}, Calibration: {calibration_source})"
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
                "distance_thresholds": self.distance_thresholds,
                "threshold_calibrations": {
                    label: calibration.to_dict()
                    for label, calibration in self.threshold_calibrations.items()
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
            if schema_version not in {1, 2, cls.SCHEMA_VERSION}:
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

        classifier = cls(
            n_neighbors=int(manifest["n_neighbors"]),
            distance_threshold=float(manifest["distance_threshold"]),
            distance_mode=distance_mode,
            threshold_mode=threshold_mode,
        )
        classifier.feature_dim = features.shape[1]
        classifier.X_data = [row.copy() for row in features]
        classifier.y_labels = [str(label) for label in labels]
        classifier.metadata = metadata
        classifier._fit_if_ready()
        registered_labels = sorted(set(classifier.y_labels))
        if schema_version < cls.SCHEMA_VERSION:
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
        return classifier
