"""Validation-based calibration and held-out classifier evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score

from src.ml.classifier import (
    MIN_COSINE_THRESHOLD,
    DistanceMode,
    TapClassifier,
    ThresholdMode,
    validate_distance_mode,
    validate_threshold_mode,
)


@dataclass(frozen=True)
class CalibrationResult:
    n_neighbors: int
    distance_threshold: float
    distance_thresholds: dict[str, float]
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
    threshold_mode: ThresholdMode = "fixed",
    distance_threshold_cap: float | None = None,
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
    selected_threshold_mode = validate_threshold_mode(threshold_mode)
    if selected_threshold_mode == "auto":
        cap = 0.20 if distance_threshold_cap is None else float(distance_threshold_cap)
        return _calibrate_per_label_thresholds(
            x_train,
            y_train,
            x_validation,
            y_validation,
            candidates,
            known_mask,
            unknown_mask,
            selected_mode,
            cap,
        )
    return _calibrate_fixed_threshold(
        x_train,
        y_train,
        x_validation,
        y_validation,
        candidates,
        known_mask,
        unknown_mask,
        selected_mode,
    )


def _calibrate_fixed_threshold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    candidates: list[int],
    known_mask: np.ndarray,
    unknown_mask: np.ndarray,
    distance_mode: DistanceMode,
) -> tuple[TapClassifier, CalibrationResult]:
    best: CalibrationResult | None = None

    for n_neighbors in candidates:
        candidate_classifier = TapClassifier(
            n_neighbors=n_neighbors,
            distance_threshold=0.0,
            distance_mode=distance_mode,
            threshold_mode="fixed",
        )
        for vector, label in zip(x_train, y_train, strict=True):
            candidate_classifier.register_tap(vector, str(label))
        predicted_and_distances = [
            candidate_classifier.classify_without_rejection(vector)
            for vector in x_validation
        ]
        predicted = np.asarray(
            [prediction for prediction, _distance in predicted_and_distances],
            dtype=np.str_,
        )
        nearest = np.asarray(
            [distance for _prediction, distance in predicted_and_distances],
            dtype=np.float64,
        )
        thresholds = np.unique(
            np.concatenate(([0.0], nearest, [np.nextafter(nearest.max(), np.inf)]))
        )
        for threshold in thresholds:
            accepted = nearest <= threshold
            known_accuracy, unknown_rejection, objective = _calibration_scores(
                predicted,
                y_validation,
                accepted,
                known_mask,
                unknown_mask,
            )
            labels = sorted(set(y_train.tolist()))
            result = CalibrationResult(
                n_neighbors=n_neighbors,
                distance_threshold=float(threshold),
                distance_thresholds={label: float(threshold) for label in labels},
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
        distance_mode=distance_mode,
        threshold_mode="fixed",
    )
    for vector, label in zip(x_train, y_train, strict=True):
        classifier.register_tap(vector, str(label))
    return classifier, best


def _calibrate_per_label_thresholds(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    candidates: list[int],
    known_mask: np.ndarray,
    unknown_mask: np.ndarray,
    distance_mode: DistanceMode,
    cap: float,
) -> tuple[TapClassifier, CalibrationResult]:
    best: CalibrationResult | None = None
    best_classifier: TapClassifier | None = None
    labels = sorted(set(y_train.tolist()))

    for n_neighbors in candidates:
        classifier = TapClassifier(
            n_neighbors=n_neighbors,
            distance_threshold=cap,
            distance_mode=distance_mode,
            threshold_mode="auto",
        )
        for vector, label in zip(x_train, y_train, strict=True):
            classifier.register_tap(vector, str(label))

        predicted_and_distances = [
            classifier.classify_without_rejection(vector) for vector in x_validation
        ]
        predicted = np.asarray(
            [prediction for prediction, _distance in predicted_and_distances],
            dtype=np.str_,
        )
        nearest = np.asarray(
            [distance for _prediction, distance in predicted_and_distances],
            dtype=np.float64,
        )

        for label in labels:
            positive_mask = (predicted == label) & (y_validation == label)
            impostor_mask = (predicted == label) & (y_validation != label)
            if not positive_mask.any() or not impostor_mask.any():
                continue
            local_distances = nearest[predicted == label]
            thresholds = np.unique(
                np.clip(
                    np.concatenate(
                        (
                            [MIN_COSINE_THRESHOLD],
                            local_distances,
                            [cap],
                        )
                    ),
                    MIN_COSINE_THRESHOLD,
                    cap,
                )
            )
            best_threshold = MIN_COSINE_THRESHOLD
            best_local_objective = -np.inf
            for threshold in thresholds:
                positive_acceptance = float(np.mean(nearest[positive_mask] <= threshold))
                impostor_rejection = float(np.mean(nearest[impostor_mask] > threshold))
                local_objective = (positive_acceptance + impostor_rejection) / 2.0
                if (local_objective, -float(threshold)) > (
                    best_local_objective,
                    -best_threshold,
                ):
                    best_local_objective = local_objective
                    best_threshold = float(threshold)
            classifier.set_validation_calibration(
                str(label),
                best_threshold,
                nearest[positive_mask],
            )

        accepted = np.asarray(
            [
                distance <= classifier.threshold_for_label(str(label))
                for label, distance in zip(predicted, nearest, strict=True)
            ]
        )
        known_accuracy, unknown_rejection, objective = _calibration_scores(
            predicted,
            y_validation,
            accepted,
            known_mask,
            unknown_mask,
        )
        result = CalibrationResult(
            n_neighbors=n_neighbors,
            distance_threshold=cap,
            distance_thresholds=dict(classifier.distance_thresholds),
            objective=objective,
            known_end_to_end_accuracy=known_accuracy,
            unknown_rejection_rate=unknown_rejection,
        )
        result_tie_break = -sum(result.distance_thresholds.values())
        best_tie_break = (
            -sum(best.distance_thresholds.values()) if best is not None else -np.inf
        )
        if best is None or (
            result.objective,
            result_tie_break,
            -result.n_neighbors,
        ) > (
            best.objective,
            best_tie_break,
            -best.n_neighbors,
        ):
            best = result
            best_classifier = classifier

    assert best is not None and best_classifier is not None
    return best_classifier, best


def _calibration_scores(
    predicted: np.ndarray,
    actual: np.ndarray,
    accepted: np.ndarray,
    known_mask: np.ndarray,
    unknown_mask: np.ndarray,
) -> tuple[float, float | None, float]:
    known_accuracy = float(
        np.mean((predicted == actual)[known_mask] & accepted[known_mask])
    )
    unknown_rejection = (
        float(np.mean(~accepted[unknown_mask])) if unknown_mask.any() else None
    )
    objective = (
        known_accuracy
        if unknown_rejection is None
        else (known_accuracy + unknown_rejection) / 2.0
    )
    return known_accuracy, unknown_rejection, objective


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
