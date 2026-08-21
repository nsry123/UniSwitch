"""Incremental prototype classifier with open-set rejection and persistence."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
from sklearn.preprocessing import StandardScaler

DistanceMode = Literal["cosine", "standardized_euclidean"]
DISTANCE_MODES = frozenset({"cosine", "standardized_euclidean"})
MIN_CLASS_SAMPLES = 3


def validate_distance_mode(distance_mode: str) -> DistanceMode:
    """Return a supported distance mode or raise a configuration error."""
    if distance_mode not in DISTANCE_MODES:
        choices = ", ".join(sorted(DISTANCE_MODES))
        raise ValueError(f"distance_mode must be one of: {choices}")
    return cast(DistanceMode, distance_mode)


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


@dataclass(frozen=True)
class PredictionResult:
    """Detailed, immutable result for one open-set classification."""

    label: str
    accepted: bool
    probabilities: Mapping[str, float]
    prototype_distance: float | None
    exemplar_distance: float | None
    score: float | None
    acceptance_radius: float | None
    margin: float | None
    required_margin: float
    rejection_reason: str | None


@dataclass(frozen=True)
class _ClassStatistics:
    prototype: np.ndarray
    exemplars: np.ndarray
    within_mean: float
    within_std: float
    acceptance_radius: float


@dataclass(frozen=True)
class _ClassScore:
    label: str
    prototype_distance: float
    exemplar_distance: float
    score: float
    acceptance_radius: float


class TapClassifier:
    """Incrementally learn classes and reject distant or ambiguous inputs."""

    SCHEMA_VERSION = 3

    def __init__(
        self,
        n_neighbors: int = 3,
        distance_threshold: float = 2.5,
        distance_mode: DistanceMode = "standardized_euclidean",
        prototype_weight: float = 0.5,
        spread_multiplier: float = 2.0,
        margin_threshold: float | None = None,
        probability_temperature: float | None = None,
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if not np.isfinite(distance_threshold) or distance_threshold <= 0:
            raise ValueError("distance_threshold must be finite and positive")
        if not np.isfinite(prototype_weight) or not 0 <= prototype_weight <= 1:
            raise ValueError("prototype_weight must be between zero and one")
        if not np.isfinite(spread_multiplier) or spread_multiplier < 0:
            raise ValueError("spread_multiplier must be finite and non-negative")
        if margin_threshold is None:
            margin_threshold = 0.1 * distance_threshold
        if not np.isfinite(margin_threshold) or margin_threshold < 0:
            raise ValueError("margin_threshold must be finite and non-negative")
        if probability_temperature is None:
            probability_temperature = 0.5 * distance_threshold
        if not np.isfinite(probability_temperature) or probability_temperature <= 0:
            raise ValueError("probability_temperature must be finite and positive")

        self.n_neighbors = int(n_neighbors)
        self.distance_threshold = float(distance_threshold)
        self.distance_mode = validate_distance_mode(distance_mode)
        self.prototype_weight = float(prototype_weight)
        self.spread_multiplier = float(spread_multiplier)
        self.margin_threshold = float(margin_threshold)
        self.probability_temperature = float(probability_temperature)
        self.scaler: StandardScaler | None = (
            None if self.distance_mode == "cosine" else StandardScaler()
        )
        self.X_data: list[np.ndarray] = []
        self.y_labels: list[str] = []
        self.is_trained = False
        self.feature_dim: int | None = None
        self.metadata: dict[str, Any] = {}
        self._class_statistics: dict[str, _ClassStatistics] = {}
        self._transformed_features: np.ndarray | None = None
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

    def _transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if self.distance_mode == "cosine":
            return normalize_feature_matrix(matrix, "features")
        assert self.scaler is not None
        return self.scaler.transform(matrix)

    def _transform_query(self, vector: np.ndarray) -> np.ndarray:
        return self._transform_matrix(vector.reshape(1, -1))

    def _distance(self, left: np.ndarray, right: np.ndarray) -> float:
        if self.distance_mode == "cosine":
            return float(np.clip(1.0 - np.dot(left, right), 0.0, 2.0))
        return float(np.linalg.norm(left - right))

    def _prototype(self, exemplars: np.ndarray) -> np.ndarray:
        prototype = np.mean(exemplars, axis=0)
        if self.distance_mode == "cosine":
            norm = np.linalg.norm(prototype)
            if norm == 0:
                return exemplars[0].copy()
            prototype = prototype / norm
        return prototype

    def _score_against(
        self, vector: np.ndarray, prototype: np.ndarray, exemplars: np.ndarray
    ) -> tuple[float, float, float]:
        prototype_distance = self._distance(vector, prototype)
        distances = sorted(self._distance(vector, exemplar) for exemplar in exemplars)
        neighbor_count = min(self.n_neighbors, len(distances))
        exemplar_distance = float(np.mean(distances[:neighbor_count]))
        score = (
            self.prototype_weight * prototype_distance
            + (1.0 - self.prototype_weight) * exemplar_distance
        )
        return prototype_distance, exemplar_distance, score

    def _fit_if_ready(self) -> None:
        if not self.X_data:
            self.is_trained = False
            self._class_statistics = {}
            self._transformed_features = None
            return
        raw = np.vstack(self.X_data)
        if self.distance_mode == "standardized_euclidean":
            assert self.scaler is not None
            self.scaler.fit(raw)
        transformed = self._transform_matrix(raw)
        labels = np.asarray(self.y_labels)
        statistics: dict[str, _ClassStatistics] = {}
        for label in sorted(set(self.y_labels)):
            exemplars = transformed[labels == label]
            if len(exemplars) < MIN_CLASS_SAMPLES:
                continue
            prototype = self._prototype(exemplars)
            leave_one_out_scores: list[float] = []
            for index, vector in enumerate(exemplars):
                other = np.delete(exemplars, index, axis=0)
                other_prototype = self._prototype(other)
                _, _, score = self._score_against(vector, other_prototype, other)
                leave_one_out_scores.append(score)
            within_mean = float(np.mean(leave_one_out_scores))
            within_std = float(np.std(leave_one_out_scores))
            acceptance_radius = min(
                self.distance_threshold,
                max(
                    0.25 * self.distance_threshold,
                    within_mean + self.spread_multiplier * within_std,
                ),
            )
            statistics[label] = _ClassStatistics(
                prototype=prototype,
                exemplars=exemplars,
                within_mean=within_mean,
                within_std=within_std,
                acceptance_radius=acceptance_radius,
            )
        self._transformed_features = transformed
        self._class_statistics = statistics
        self.is_trained = bool(statistics)

    def _nearest_distance(self, vector: np.ndarray) -> float:
        """Return the nearest stored-exemplar distance for compatibility."""
        if not self.X_data:
            raise ValueError("classifier has no registered features")
        validated = self._validate_feature(vector)
        transformed = self._transform_query(validated)[0]
        assert self._transformed_features is not None
        return min(self._distance(transformed, row) for row in self._transformed_features)

    def register_tap(
        self, feature_vector: np.ndarray, label: str, *, announce: bool = True
    ) -> None:
        """Add one captured-window example and rebuild derived class statistics."""
        with self._lock:
            normalized_label = self._validate_label(label)
            vector = self._validate_feature(feature_vector)
            if self.feature_dim is None:
                self.feature_dim = int(vector.size)
            self.X_data.append(vector)
            self.y_labels.append(normalized_label)
            self._fit_if_ready()
            class_count = self.y_labels.count(normalized_label)
            if not announce:
                return
            if class_count >= MIN_CLASS_SAMPLES:
                print(f"[*] Model updated. Total registered samples: {len(self.X_data)}")
            else:
                remaining = MIN_CLASS_SAMPLES - class_count
                print(f"[*] Sample recorded. Need {remaining} more for '{normalized_label}'.")

    def delete_class(self, label: str) -> int:
        """Delete every registered example for a class and rebuild statistics."""
        with self._lock:
            normalized_label = self._validate_label(label)
            matching = [
                index
                for index, registered_label in enumerate(self.y_labels)
                if registered_label == normalized_label
            ]
            if not matching:
                raise ValueError(f"class is not registered: {normalized_label!r}")
            retained = [index for index in range(len(self.y_labels)) if index not in matching]
            self.X_data = [self.X_data[index] for index in retained]
            self.y_labels = [self.y_labels[index] for index in retained]
            if not self.X_data:
                self.feature_dim = None
            self._fit_if_ready()
            return len(matching)

    def class_counts(self) -> dict[str, int]:
        """Return registered labels and example counts in deterministic order."""
        with self._lock:
            return {
                label: self.y_labels.count(label) for label in sorted(set(self.y_labels))
            }

    def predict_detailed(self, feature_vector: np.ndarray) -> PredictionResult:
        """Return class probabilities and the open-set decision for one vector."""
        with self._lock:
            if not self.is_trained:
                return PredictionResult(
                    "Unknown", False, MappingProxyType({}), None, None, None, None, None,
                    self.margin_threshold,
                    "insufficient_class_samples",
                )
            vector = self._validate_feature(feature_vector)
            transformed = self._transform_query(vector)[0]
            scores: list[_ClassScore] = []
            for label, statistics in sorted(self._class_statistics.items()):
                prototype_distance, exemplar_distance, score = self._score_against(
                    transformed, statistics.prototype, statistics.exemplars
                )
                scores.append(
                    _ClassScore(
                        label, prototype_distance, exemplar_distance, score,
                        statistics.acceptance_radius,
                    )
                )
            scores.sort(key=lambda item: (item.score, item.label))
            logits = -np.asarray([item.score for item in scores]) / self.probability_temperature
            logits -= np.max(logits)
            probability_values = np.exp(logits)
            probability_values /= np.sum(probability_values)
            probabilities = MappingProxyType(
                {
                    item.label: float(probability)
                    for item, probability in zip(scores, probability_values, strict=True)
                }
            )
            best = scores[0]
            margin = float("inf") if len(scores) == 1 else scores[1].score - best.score
            rejection_reason: str | None = None
            if best.score > best.acceptance_radius:
                rejection_reason = "distance"
            elif margin < self.margin_threshold:
                rejection_reason = "margin"
            accepted = rejection_reason is None
            return PredictionResult(
                best.label if accepted else "Unknown",
                accepted,
                probabilities,
                best.prototype_distance,
                best.exemplar_distance,
                best.score,
                best.acceptance_radius,
                margin,
                self.margin_threshold,
                rejection_reason,
            )

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict a label, returning ``Unknown`` for rejected input."""
        result = self.predict_detailed(feature_vector)
        if result.accepted:
            print(
                f"[+] Match found: {result.label} "
                f"(Score: {result.score:.3f}, Margin: {result.margin:.3f})"
            )
        elif result.score is not None:
            print(
                f"[-] Ignored: {result.rejection_reason} rejection "
                f"(Score: {result.score:.3f}, Margin: {result.margin:.3f})"
            )
        return result.label

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
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
                "prototype_weight": self.prototype_weight,
                "spread_multiplier": self.spread_multiplier,
                "margin_threshold": self.margin_threshold,
                "probability_temperature": self.probability_temperature,
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
        """Load registrations, upgrading older schemas in memory."""
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
        else:
            if "distance_mode" not in manifest:
                raise ValueError("classifier state is missing distance_mode")
            distance_mode = validate_distance_mode(str(manifest["distance_mode"]))
        threshold = float(manifest["distance_threshold"])
        classifier = cls(
            n_neighbors=int(manifest["n_neighbors"]),
            distance_threshold=threshold,
            distance_mode=distance_mode,
            prototype_weight=float(manifest.get("prototype_weight", 0.5)),
            spread_multiplier=float(manifest.get("spread_multiplier", 2.0)),
            margin_threshold=float(manifest.get("margin_threshold", 0.1 * threshold)),
            probability_temperature=float(
                manifest.get("probability_temperature", 0.5 * threshold)
            ),
        )
        classifier.feature_dim = features.shape[1]
        classifier.X_data = [row.copy() for row in features]
        classifier.y_labels = [str(label) for label in labels]
        classifier.metadata = metadata
        classifier._fit_if_ready()
        return classifier
