"""Validation-based calibration and held-out classifier evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.ml.classifier import (
    DistanceMode,
    TapClassifier,
    normalize_feature_matrix,
    validate_distance_mode,
)


@dataclass(frozen=True)
class CalibrationResult:
    n_neighbors: int
    distance_threshold: float
    objective: float
    known_end_to_end_accuracy: float
    unknown_rejection_rate: float | None


@dataclass(frozen=True)
class EvaluationReport:
    total_samples: int
    known_samples: int
    unknown_samples: int
    known_coverage: float
    known_accepted_accuracy: float | None
    known_end_to_end_accuracy: float
    known_macro_f1: float
    unknown_rejection_rate: float | None
    rejection_precision: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _as_feature_matrix(features: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def calibrate_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    neighbor_candidates: Iterable[int] = (1, 3, 5, 7),
    distance_mode: DistanceMode = "standardized_euclidean",
) -> tuple[TapClassifier, CalibrationResult]:
    """Select K and rejection distance using only the validation split."""
    x_train = _as_feature_matrix(train_features, "train_features")
    x_validation = _as_feature_matrix(validation_features, "validation_features")
    y_train = np.asarray(train_labels, dtype=np.str_)
    y_validation = np.asarray(validation_labels, dtype=np.str_)
    if y_train.shape != (x_train.shape[0],) or y_validation.shape != (x_validation.shape[0],):
        raise ValueError("labels must align with feature rows")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("train and validation feature dimensions differ")

    candidates = sorted({int(value) for value in neighbor_candidates if 0 < value <= len(x_train)})
    if not candidates:
        raise ValueError("no neighbor candidate is valid for the training sample count")
    known_labels = set(y_train.tolist())
    known_mask = np.asarray([label in known_labels for label in y_validation])
    unknown_mask = ~known_mask
    if not known_mask.any():
        raise ValueError("validation split contains no labels represented in training")

    selected_mode = validate_distance_mode(distance_mode)
    if selected_mode == "cosine":
        transformed_train = normalize_feature_matrix(x_train, "train_features")
        transformed_validation = normalize_feature_matrix(
            x_validation, "validation_features"
        )
    else:
        scaler = StandardScaler().fit(x_train)
        transformed_train = scaler.transform(x_train)
        transformed_validation = scaler.transform(x_validation)
    best: CalibrationResult | None = None

    for n_neighbors in candidates:
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="cosine" if selected_mode == "cosine" else "minkowski",
            algorithm="brute" if selected_mode == "cosine" else "auto",
        ).fit(transformed_train, y_train)
        predicted = model.predict(transformed_validation)
        distances, _ = model.kneighbors(transformed_validation, n_neighbors=1)
        nearest = distances[:, 0]
        thresholds = np.unique(
            np.concatenate(([0.0], nearest, [np.nextafter(nearest.max(), np.inf)]))
        )
        for threshold in thresholds:
            accepted = nearest <= threshold
            known_accuracy = float(
                np.mean((predicted == y_validation)[known_mask] & accepted[known_mask])
            )
            unknown_rejection = (
                float(np.mean(~accepted[unknown_mask])) if unknown_mask.any() else None
            )
            objective = (
                known_accuracy
                if unknown_rejection is None
                else (known_accuracy + unknown_rejection) / 2.0
            )
            result = CalibrationResult(
                n_neighbors=n_neighbors,
                distance_threshold=float(threshold),
                objective=objective,
                known_end_to_end_accuracy=known_accuracy,
                unknown_rejection_rate=unknown_rejection,
            )
            if best is None or (
                result.objective,
                -result.distance_threshold,
                -result.n_neighbors,
            ) > (
                best.objective,
                -best.distance_threshold,
                -best.n_neighbors,
            ):
                best = result

    assert best is not None
    classifier = TapClassifier(
        best.n_neighbors,
        best.distance_threshold,
        distance_mode=selected_mode,
    )
    for vector, label in zip(x_train, y_train, strict=True):
        classifier.register_tap(vector, str(label))
    return classifier, best


def evaluate_classifier(
    classifier: TapClassifier,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> EvaluationReport:
    """Report known classification and unknown rejection separately."""
    features = _as_feature_matrix(test_features, "test_features")
    labels = np.asarray(test_labels, dtype=np.str_)
    if labels.shape != (features.shape[0],):
        raise ValueError("test labels must align with feature rows")

    known_label_set = set(classifier.y_labels)
    known_mask = np.asarray([label in known_label_set for label in labels])
    unknown_mask = ~known_mask
    predictions = np.asarray([classifier.predict(vector) for vector in features], dtype=np.str_)
    accepted = predictions != "Unknown"
    known_count = int(known_mask.sum())
    unknown_count = int(unknown_mask.sum())
    known_accepted = known_mask & accepted
    known_coverage = float(known_accepted.sum() / known_count) if known_count else 0.0
    accepted_accuracy = (
        float(np.mean(predictions[known_accepted] == labels[known_accepted]))
        if known_accepted.any()
        else None
    )
    end_to_end_accuracy = (
        float(np.mean((predictions == labels)[known_mask] & accepted[known_mask]))
        if known_count
        else 0.0
    )
    macro_f1 = (
        float(
            f1_score(
                labels[known_mask],
                predictions[known_mask],
                labels=sorted(known_label_set),
                average="macro",
                zero_division=0,
            )
        )
        if known_count
        else 0.0
    )
    unknown_rejection = float(np.mean(~accepted[unknown_mask])) if unknown_count else None
    rejected = ~accepted
    rejection_precision = float(np.mean(unknown_mask[rejected])) if rejected.any() else None
    return EvaluationReport(
        total_samples=len(features),
        known_samples=known_count,
        unknown_samples=unknown_count,
        known_coverage=known_coverage,
        known_accepted_accuracy=accepted_accuracy,
        known_end_to_end_accuracy=end_to_end_accuracy,
        known_macro_f1=macro_f1,
        unknown_rejection_rate=unknown_rejection,
        rejection_precision=rejection_precision,
    )
