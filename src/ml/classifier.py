"""Incremental KNN classifier with validation and safe persistence."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class TapClassifier:
    """Incrementally learn labeled feature vectors and reject distant inputs."""

    SCHEMA_VERSION = 1

    def __init__(self, n_neighbors: int = 3, distance_threshold: float = 2.5) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if not np.isfinite(distance_threshold) or distance_threshold < 0:
            raise ValueError("distance_threshold must be finite and non-negative")

        self.n_neighbors = n_neighbors
        self.distance_threshold = float(distance_threshold)
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
        self.scaler.fit(x_raw)
        self.model.fit(self.scaler.transform(x_raw), y_train)
        self.is_trained = True

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
            vector_scaled = self.scaler.transform(vector.reshape(1, -1))
            distances, _ = self.model.kneighbors(vector_scaled, n_neighbors=1)
            closest_distance = float(distances[0][0])

            if closest_distance > self.distance_threshold:
                print(f"[-] Ignored: Unregistered sound (Distance: {closest_distance:.2f})")
                return "Unknown"

            prediction = str(self.model.predict(vector_scaled)[0])
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
            if manifest.get("schema_version") != cls.SCHEMA_VERSION:
                raise ValueError(f"unsupported classifier schema: {manifest.get('schema_version')}")
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

        classifier = cls(
            n_neighbors=int(manifest["n_neighbors"]),
            distance_threshold=float(manifest["distance_threshold"]),
        )
        classifier.feature_dim = features.shape[1]
        classifier.X_data = [row.copy() for row in features]
        classifier.y_labels = [str(label) for label in labels]
        classifier.metadata = dict(manifest.get("metadata", {}))
        classifier._fit_if_ready()
        return classifier
