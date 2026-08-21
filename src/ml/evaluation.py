"""Validation-based calibration and held-out classifier evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from src.ml.classifier import (
    MAX_SEPARATION_THRESHOLD,
    MIN_COSINE_THRESHOLD,
    ClassificationDecision,
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
    separation_thresholds: dict[str, float]
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
    ambiguity_rejection_rate: float
    known_confusion_matrix: dict[str, dict[str, int]]
    per_label_recall: dict[str, float | None]
    separation_summary: dict[str, dict[str, int | float | None]]

    def to_dict(self) -> dict[str, object]:
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
    separation_threshold: float = 0.15,
) -> tuple[TapClassifier, CalibrationResult]:
    """Select neighbor count and rejection thresholds on validation data."""
    x_train = _as_feature_matrix(train_features, "train_features")
    x_validation = _as_feature_matrix(validation_features, "validation_features")
    y_train = np.asarray(train_labels, dtype=np.str_)
    y_validation = np.asarray(validation_labels, dtype=np.str_)
    if y_train.shape != (x_train.shape[0],) or y_validation.shape != (x_validation.shape[0],):
        raise ValueError("labels must align with feature rows")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("train and validation feature dimensions differ")

    selected_mode = validate_distance_mode(distance_mode)
    selected_threshold_mode = validate_threshold_mode(threshold_mode)
    label_counts = [int(np.sum(y_train == label)) for label in np.unique(y_train)]
    maximum_neighbors = max(label_counts) if selected_mode == "cosine" else len(x_train)
    candidates = sorted(
        {
            int(value)
            for value in neighbor_candidates
            if 0 < value <= maximum_neighbors
        }
    )
    if not candidates:
        raise ValueError("no neighbor candidate is valid for the training sample count")
    known_labels = set(y_train.tolist())
    known_mask = np.asarray([label in known_labels for label in y_validation])
    unknown_mask = ~known_mask
    if not known_mask.any():
        raise ValueError("validation split contains no labels represented in training")

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
            separation_threshold,
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
        separation_threshold,
    )


def _raw_decisions(
    classifier: TapClassifier,
    features: np.ndarray,
) -> list[ClassificationDecision]:
    decisions = [classifier.classify_without_rejection(vector) for vector in features]
    if any(decision.predicted_label is None for decision in decisions):
        raise ValueError("neighbor candidate leaves no eligible class")
    return decisions


def _decision_arrays(
    decisions: list[ClassificationDecision],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(
        [str(decision.predicted_label) for decision in decisions], dtype=np.str_
    )
    nearest_values: list[float] = []
    for decision in decisions:
        assert decision.nearest_distance is not None
        nearest_values.append(decision.nearest_distance)
    nearest = np.asarray(nearest_values, dtype=np.float64)
    separations = np.asarray(
        [
            np.nan if decision.separation is None else decision.separation
            for decision in decisions
        ],
        dtype=np.float64,
    )
    return predicted, nearest, separations


def _calibrate_fixed_threshold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    candidates: list[int],
    known_mask: np.ndarray,
    unknown_mask: np.ndarray,
    distance_mode: DistanceMode,
    separation_threshold: float,
) -> tuple[TapClassifier, CalibrationResult]:
    best: CalibrationResult | None = None

    for n_neighbors in candidates:
        candidate_classifier = TapClassifier(
            n_neighbors=n_neighbors,
            distance_threshold=0.0,
            distance_mode=distance_mode,
            threshold_mode="fixed",
            separation_threshold=separation_threshold,
        )
        for vector, label in zip(x_train, y_train, strict=True):
            candidate_classifier.register_tap(vector, str(label))
        decisions = _raw_decisions(candidate_classifier, x_validation)
        predicted, nearest, separations = _decision_arrays(decisions)
        thresholds = np.unique(
            np.concatenate(([0.0], nearest, [np.nextafter(nearest.max(), np.inf)]))
        )
        for threshold in thresholds:
            selected_threshold = float(threshold)
            candidate_classifier.distance_threshold = selected_threshold
            candidate_classifier.distance_thresholds = {
                label: selected_threshold for label in set(y_train.tolist())
            }
            candidate_classifier._calibrate_all_separations_from_registrations()
            _calibrate_validation_separations(
                candidate_classifier,
                sorted(set(y_train.tolist())),
                predicted,
                nearest,
                separations,
                y_validation,
            )
            final_decisions = [
                candidate_classifier.classify(vector) for vector in x_validation
            ]
            accepted = np.asarray([decision.accepted for decision in final_decisions])
            known_accuracy, unknown_rejection, objective = _calibration_scores(
                predicted, y_validation, accepted, known_mask, unknown_mask
            )
            labels = sorted(set(y_train.tolist()))
            result = CalibrationResult(
                n_neighbors=n_neighbors,
                distance_threshold=selected_threshold,
                distance_thresholds={label: selected_threshold for label in labels},
                separation_thresholds=dict(candidate_classifier.separation_thresholds),
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
        separation_threshold=separation_threshold,
    )
    for vector, label in zip(x_train, y_train, strict=True):
        classifier.register_tap(vector, str(label))
    final_decisions = _raw_decisions(classifier, x_validation)
    final_predicted, final_nearest, final_separations = _decision_arrays(final_decisions)
    _calibrate_validation_separations(
        classifier,
        sorted(set(y_train.tolist())),
        final_predicted,
        final_nearest,
        final_separations,
        y_validation,
    )
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
    separation_threshold: float,
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
            separation_threshold=separation_threshold,
        )
        for vector, label in zip(x_train, y_train, strict=True):
            classifier.register_tap(vector, str(label))

        decisions = _raw_decisions(classifier, x_validation)
        predicted, nearest, separations = _decision_arrays(decisions)
        _calibrate_validation_distances(
            classifier, labels, predicted, nearest, y_validation, cap
        )
        _calibrate_validation_separations(
            classifier,
            labels,
            predicted,
            nearest,
            separations,
            y_validation,
        )

        final_decisions = [classifier.classify(vector) for vector in x_validation]
        accepted = np.asarray([decision.accepted for decision in final_decisions])
        known_accuracy, unknown_rejection, objective = _calibration_scores(
            predicted, y_validation, accepted, known_mask, unknown_mask
        )
        result = CalibrationResult(
            n_neighbors=n_neighbors,
            distance_threshold=cap,
            distance_thresholds=dict(classifier.distance_thresholds),
            separation_thresholds=dict(classifier.separation_thresholds),
            objective=objective,
            known_end_to_end_accuracy=known_accuracy,
            unknown_rejection_rate=unknown_rejection,
        )
        strictness = sum(result.separation_thresholds.values()) - sum(
            result.distance_thresholds.values()
        )
        best_strictness = (
            sum(best.separation_thresholds.values())
            - sum(best.distance_thresholds.values())
            if best is not None
            else -np.inf
        )
        if best is None or (
            result.objective,
            strictness,
            -result.n_neighbors,
        ) > (
            best.objective,
            best_strictness,
            -best.n_neighbors,
        ):
            best = result
            best_classifier = classifier

    assert best is not None and best_classifier is not None
    return best_classifier, best


def _calibrate_validation_distances(
    classifier: TapClassifier,
    labels: list[str],
    predicted: np.ndarray,
    nearest: np.ndarray,
    actual: np.ndarray,
    cap: float,
) -> None:
    for label in labels:
        positive_mask = (predicted == label) & (actual == label)
        impostor_mask = (predicted == label) & (actual != label)
        if not positive_mask.any() or not impostor_mask.any():
            continue
        local_distances = nearest[predicted == label]
        thresholds = np.unique(
            np.clip(
                np.concatenate(([MIN_COSINE_THRESHOLD], local_distances, [cap])),
                MIN_COSINE_THRESHOLD,
                cap,
            )
        )
        best_threshold = MIN_COSINE_THRESHOLD
        best_objective = -np.inf
        for threshold in thresholds:
            positive_acceptance = float(np.mean(nearest[positive_mask] <= threshold))
            impostor_rejection = float(np.mean(nearest[impostor_mask] > threshold))
            objective = (positive_acceptance + impostor_rejection) / 2.0
            if (objective, -float(threshold)) > (best_objective, -best_threshold):
                best_objective = objective
                best_threshold = float(threshold)
        classifier.set_validation_calibration(
            label, best_threshold, nearest[positive_mask]
        )


def _calibrate_validation_separations(
    classifier: TapClassifier,
    labels: list[str],
    predicted: np.ndarray,
    nearest: np.ndarray,
    separations: np.ndarray,
    actual: np.ndarray,
) -> None:
    if classifier.decision_mode != "class_balanced":
        return
    for label in labels:
        positive_mask = (predicted == label) & (actual == label) & ~np.isnan(separations)
        impostor_mask = (predicted == label) & (actual != label) & ~np.isnan(separations)
        if not positive_mask.any() or not impostor_mask.any():
            continue
        local_separations = separations[(predicted == label) & ~np.isnan(separations)]
        candidates = np.unique(
            np.clip(
                np.concatenate(
                    (
                        [classifier.separation_threshold],
                        local_separations,
                        [MAX_SEPARATION_THRESHOLD],
                    )
                ),
                classifier.separation_threshold,
                MAX_SEPARATION_THRESHOLD,
            )
        )
        distance_threshold = classifier.threshold_for_label(label)
        best_threshold = classifier.separation_threshold
        best_objective = -np.inf
        for threshold in candidates:
            positive_acceptance = float(
                np.mean(
                    (nearest[positive_mask] <= distance_threshold)
                    & (separations[positive_mask] >= threshold)
                )
            )
            impostor_rejection = float(
                np.mean(
                    (nearest[impostor_mask] > distance_threshold)
                    | (separations[impostor_mask] < threshold)
                )
            )
            objective = (positive_acceptance + impostor_rejection) / 2.0
            if (objective, float(threshold)) > (best_objective, best_threshold):
                best_objective = objective
                best_threshold = float(threshold)
        label_mask = actual == label
        accuracy = float(np.mean(predicted[label_mask] == label))
        classifier.set_validation_separation_calibration(
            label,
            best_threshold,
            separations[positive_mask],
            accuracy,
        )


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
    """Report known classification, ambiguity, and unknown rejection."""
    features = _as_feature_matrix(test_features, "test_features")
    labels = np.asarray(test_labels, dtype=np.str_)
    if labels.shape != (features.shape[0],):
        raise ValueError("test labels must align with feature rows")

    known_label_list = sorted(set(classifier.y_labels))
    known_label_set = set(known_label_list)
    known_mask = np.asarray([label in known_label_set for label in labels])
    unknown_mask = ~known_mask
    decisions = [classifier.classify(vector) for vector in features]
    raw_predictions = np.asarray(
        [
            "Unknown" if decision.predicted_label is None else decision.predicted_label
            for decision in decisions
        ],
        dtype=np.str_,
    )
    accepted = np.asarray([decision.accepted for decision in decisions])
    predictions = np.where(accepted, raw_predictions, "Unknown")
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
                labels=known_label_list,
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
    ambiguity_rejections = np.asarray(
        [decision.rejection_reason == "ambiguous" for decision in decisions]
    )
    ambiguity_rejection_rate = float(np.mean(ambiguity_rejections))

    confusion_labels = [*known_label_list, "Unknown"]
    matrix = confusion_matrix(
        labels[known_mask], predictions[known_mask], labels=confusion_labels
    )
    confusion = {
        actual_label: {
            predicted_label: int(matrix[row_index, column_index])
            for column_index, predicted_label in enumerate(confusion_labels)
        }
        for row_index, actual_label in enumerate(known_label_list)
    }
    per_label_recall = {
        label: (
            float(np.mean(predictions[labels == label] == label))
            if np.any(labels == label)
            else None
        )
        for label in known_label_list
    }
    separation_summary: dict[str, dict[str, int | float | None]] = {}
    for label in known_label_list:
        values = np.asarray(
            [
                decision.separation
                for decision in decisions
                if decision.predicted_label == label and decision.separation is not None
            ],
            dtype=np.float64,
        )
        separation_summary[label] = {
            "count": int(values.size),
            "minimum": float(np.min(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "maximum": float(np.max(values)) if values.size else None,
        }

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
        ambiguity_rejection_rate=ambiguity_rejection_rate,
        known_confusion_matrix=confusion,
        per_label_recall=per_label_recall,
        separation_summary=separation_summary,
    )
