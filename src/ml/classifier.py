"""Incremental KNN classifier with validation and safe persistence."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DistanceMode = Literal["cosine", "standardized_euclidean"]
DISTANCE_MODES = frozenset({"cosine", "standardized_euclidean"})


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


class TapClassifier:
    """Incrementally learn labeled feature vectors and reject distant inputs."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        n_neighbors: int = 3,
        distance_threshold: float = 2.5,
        distance_mode: DistanceMode = "standardized_euclidean",
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if not np.isfinite(distance_threshold) or distance_threshold < 0:
            raise ValueError("distance_threshold must be finite and non-negative")

        self.n_neighbors = n_neighbors
        self.distance_threshold = float(distance_threshold)
        self.distance_mode = validate_distance_mode(distance_mode)
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
        if self.distance_mode == "cosine":
            x_train = normalize_feature_matrix(x_raw, "registered features")
        else:
            assert self.scaler is not None
            self.scaler.fit(x_raw)
            x_train = self.scaler.transform(x_raw)
        self.model.fit(x_train, y_train)
        self.is_trained = True

    def _transform_query(self, vector: np.ndarray) -> np.ndarray:
        query = vector.reshape(1, -1)
        if self.distance_mode == "cosine":
            return normalize_feature_matrix(query, "feature_vector")
        assert self.scaler is not None
        return self.scaler.transform(query)

    def _nearest_distance(self, vector: np.ndarray) -> float:
        transformed_vector = self._transform_query(vector)
        distances, _ = self.model.kneighbors(transformed_vector, n_neighbors=1)
        return float(distances[0][0])

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
            vector = self._validate_feature(feature_vector)
            transformed_vector = self._transform_query(vector)
            closest_distance = self._nearest_distance(vector)

            if closest_distance > self.distance_threshold:
                print(f"[-] Ignored: Unregistered sound (Distance: {closest_distance:.2f})")
                return "Unknown"

            prediction = str(self.model.predict(transformed_vector)[0])
            print(f"[+] Match found: {prediction} (Distance: {closest_distance:.2f})")
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
            if schema_version not in {1, cls.SCHEMA_VERSION}:
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

        classifier = cls(
            n_neighbors=int(manifest["n_neighbors"]),
            distance_threshold=float(manifest["distance_threshold"]),
            distance_mode=distance_mode,
        )
        classifier.feature_dim = features.shape[1]
        classifier.X_data = [row.copy() for row in features]
        classifier.y_labels = [str(label) for label in labels]
        classifier.metadata = metadata
        classifier._fit_if_ready()
        return classifier
