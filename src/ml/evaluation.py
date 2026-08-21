"""Validation-based calibration and held-out classifier evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score
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
    prototype_weight: float
    spread_multiplier: float
    margin_threshold: float
    probability_temperature: float
    known_probability_nll: float
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
    known_probability_nll: float | None

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
    prototype_weight_candidates: Iterable[float] = (0.25, 0.5, 0.75),
    spread_multiplier_candidates: Iterable[float] = (1.5, 2.0, 3.0),
    margin_ratio_candidates: Iterable[float] = (0.0, 0.05, 0.1, 0.2),
    temperature_ratio_candidates: Iterable[float] = (0.25, 0.5, 1.0),
) -> tuple[TapClassifier, CalibrationResult]:
    """Select prototype scoring and rejection settings on validation data."""
    x_train = _as_feature_matrix(train_features, "train_features")
    x_validation = _as_feature_matrix(validation_features, "validation_features")
    y_train = np.asarray(train_labels, dtype=np.str_)
    y_validation = np.asarray(validation_labels, dtype=np.str_)
    if y_train.shape != (x_train.shape[0],) or y_validation.shape != (x_validation.shape[0],):
        raise ValueError("labels must align with feature rows")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    class_counts = {
        str(label): int(np.sum(y_train == label)) for label in np.unique(y_train)
    }
    undersampled = {label: count for label, count in class_counts.items() if count < 3}
    if undersampled:
        raise ValueError(
            "each training class needs at least three examples for prototype calibration: "
            f"{undersampled}"
        )

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
    if selected_mode == "cosine":
        pairwise = 1.0 - transformed_validation @ transformed_train.T
        pairwise = np.clip(pairwise, 0.0, 2.0)
    else:
        pairwise = np.linalg.norm(
            transformed_validation[:, None, :] - transformed_train[None, :, :], axis=2
        )
    nearest = np.min(pairwise, axis=1)
    positive = nearest[nearest > 0]
    minimum = float(np.min(positive)) if positive.size else float(np.finfo(float).eps)
    thresholds = np.unique(
        np.concatenate(([minimum], positive, [np.nextafter(max(minimum, nearest.max()), np.inf)]))
    )
    weights = sorted({float(value) for value in prototype_weight_candidates})
    spreads = sorted({float(value) for value in spread_multiplier_candidates})
    margin_ratios = sorted({float(value) for value in margin_ratio_candidates})
    temperature_ratios = sorted({float(value) for value in temperature_ratio_candidates})
    if not weights or not spreads or not margin_ratios or not temperature_ratios:
        raise ValueError("prototype calibration candidate sets must not be empty")

    best_classifier: TapClassifier | None = None
    best: CalibrationResult | None = None
    for n_neighbors in candidates:
        for threshold in thresholds:
            for weight in weights:
                for spread in spreads:
                    for margin_ratio in margin_ratios:
                        classifier = TapClassifier(
                            n_neighbors=n_neighbors,
                            distance_threshold=float(threshold),
                            distance_mode=selected_mode,
                            prototype_weight=weight,
                            spread_multiplier=spread,
                            margin_threshold=margin_ratio * threshold,
                            probability_temperature=0.5 * threshold,
                        )
                        for vector, label in zip(x_train, y_train, strict=True):
                            classifier.register_tap(vector, str(label), announce=False)
                        details = [
                            classifier.predict_detailed(vector) for vector in x_validation
                        ]
                        predicted = np.asarray([item.label for item in details])
                        accepted = np.asarray([item.accepted for item in details])
                        known_accuracy = float(
                            np.mean(
                                (predicted == y_validation)[known_mask]
                                & accepted[known_mask]
                            )
                        )
                        unknown_rejection = (
                            float(np.mean(~accepted[unknown_mask]))
                            if unknown_mask.any()
                            else None
                        )
                        objective = (
                            known_accuracy
                            if unknown_rejection is None
                            else (known_accuracy + unknown_rejection) / 2.0
                        )
                        provisional = CalibrationResult(
                            n_neighbors=n_neighbors,
                            distance_threshold=float(threshold),
                            prototype_weight=weight,
                            spread_multiplier=spread,
                            margin_threshold=margin_ratio * threshold,
                            probability_temperature=0.5 * threshold,
                            known_probability_nll=float("inf"),
                            objective=objective,
                            known_end_to_end_accuracy=known_accuracy,
                            unknown_rejection_rate=unknown_rejection,
                        )
                        rejection = -1.0 if unknown_rejection is None else unknown_rejection
                        best_rejection = (
                            -1.0
                            if best is None or best.unknown_rejection_rate is None
                            else best.unknown_rejection_rate
                        )
                        if best is None or (
                            objective,
                            rejection,
                            known_accuracy,
                            provisional.margin_threshold,
                            -provisional.distance_threshold,
                            -provisional.n_neighbors,
                        ) > (
                            best.objective,
                            best_rejection,
                            best.known_end_to_end_accuracy,
                            best.margin_threshold,
                            -best.distance_threshold,
                            -best.n_neighbors,
                        ):
                            best = provisional
                            best_classifier = classifier

    assert best is not None and best_classifier is not None
    best_temperature = best.probability_temperature
    best_nll = float("inf")
    for ratio in temperature_ratios:
        candidate = TapClassifier(
            best.n_neighbors,
            best.distance_threshold,
            selected_mode,
            best.prototype_weight,
            best.spread_multiplier,
            best.margin_threshold,
            ratio * best.distance_threshold,
        )
        for vector, label in zip(x_train, y_train, strict=True):
            candidate.register_tap(vector, str(label), announce=False)
        details = [candidate.predict_detailed(vector) for vector in x_validation[known_mask]]
        truth = y_validation[known_mask]
        probabilities = np.asarray(
            [
                max(item.probabilities.get(str(label), 0.0), 1e-15)
                for item, label in zip(details, truth, strict=True)
            ]
        )
        nll = float(-np.mean(np.log(probabilities)))
        if (nll, ratio) < (best_nll, best_temperature / best.distance_threshold):
            best_nll = nll
            best_temperature = ratio * best.distance_threshold
            best_classifier = candidate
    best = CalibrationResult(
        **{
            **best.__dict__,
            "probability_temperature": best_temperature,
            "known_probability_nll": best_nll,
        }
    )
    return best_classifier, best


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
    details = [classifier.predict_detailed(vector) for vector in features]
    predictions = np.asarray([item.label for item in details], dtype=np.str_)
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
    known_probability_nll = None
    if known_count:
        known_probabilities = np.asarray(
            [
                max(details[index].probabilities.get(str(labels[index]), 0.0), 1e-15)
                for index in np.flatnonzero(known_mask)
            ]
        )
        known_probability_nll = float(-np.mean(np.log(known_probabilities)))
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
        known_probability_nll=known_probability_nll,
    )
