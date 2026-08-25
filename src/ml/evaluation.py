"""Validation-based calibration and held-out classifier evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
class GroupedCalibrationResult:
    """Cross-validated settings selected from independent physical captures."""

    n_neighbors: int
    distance_threshold: float
    prototype_weight: float
    spread_multiplier: float
    margin_threshold: float
    probability_temperature: float
    known_probability_nll: float
    objective: float
    known_end_to_end_accuracy: float
    unknown_rejection_rate: float
    fold_count: int
    evaluated_candidates: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


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
    class_counts = {str(label): int(np.sum(y_train == label)) for label in np.unique(y_train)}
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
        transformed_validation = normalize_feature_matrix(x_validation, "validation_features")
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
                        details = [classifier.predict_detailed(vector) for vector in x_validation]
                        predicted = np.asarray([item.label for item in details])
                        accepted = np.asarray([item.accepted for item in details])
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


def _build_grouped_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    capture_ids: np.ndarray,
    augmented: np.ndarray,
    *,
    n_neighbors: int,
    distance_threshold: float,
    distance_mode: DistanceMode,
    prototype_weight: float,
    spread_multiplier: float,
    margin_threshold: float,
    probability_temperature: float,
) -> TapClassifier:
    """Build one candidate without repeatedly fitting after every exemplar."""
    classifier = TapClassifier(
        n_neighbors=n_neighbors,
        distance_threshold=distance_threshold,
        distance_mode=distance_mode,
        prototype_weight=prototype_weight,
        spread_multiplier=spread_multiplier,
        margin_threshold=margin_threshold,
        probability_temperature=probability_temperature,
    )
    classifier.feature_dim = features.shape[1]
    classifier.X_data = [row.copy() for row in features]
    classifier.y_labels = [str(label) for label in labels]
    classifier.capture_ids = [int(value) for value in capture_ids]
    classifier.augmented_flags = [bool(value) for value in augmented]
    classifier.next_capture_id = int(np.max(capture_ids)) + 1
    classifier._fit_if_ready()
    return classifier


def _grouped_folds(
    labels: np.ndarray, capture_ids: np.ndarray, augmented: np.ndarray
) -> tuple[np.ndarray, int]:
    """Assign physical captures to deterministic stratified folds."""
    original = ~augmented
    counts = {
        str(label): len(np.unique(capture_ids[(labels == label) & original]))
        for label in np.unique(labels)
    }
    insufficient = {label: count for label, count in counts.items() if count < 4}
    if insufficient:
        raise ValueError(
            "interactive calibration needs at least four physical captures per class: "
            f"{insufficient}"
        )
    fold_count = min(5, min(counts.values()))
    assignments: dict[int, int] = {}
    for label in sorted(counts):
        ids = sorted(np.unique(capture_ids[(labels == label) & original]).tolist())
        for index, capture_id in enumerate(ids):
            assignments[int(capture_id)] = index % fold_count
    return np.asarray([assignments[int(value)] for value in capture_ids]), fold_count


def _distance_threshold_candidates(
    features: np.ndarray,
    labels: np.ndarray,
    capture_ids: np.ndarray,
    augmented: np.ndarray,
    negative_features: np.ndarray,
    fold_assignments: np.ndarray,
    fold_count: int,
    distance_mode: DistanceMode,
) -> np.ndarray:
    """Return a bounded empirical grid of held-out nearest distances."""
    distances: list[float] = []
    original = ~augmented
    for fold in range(fold_count):
        train = features[fold_assignments != fold]
        validation = features[(fold_assignments == fold) & original]
        queries = np.vstack((validation, negative_features))
        if distance_mode == "cosine":
            train_transformed = normalize_feature_matrix(train, "train_features")
            query_transformed = normalize_feature_matrix(queries, "validation_features")
            pairwise = np.clip(1.0 - query_transformed @ train_transformed.T, 0.0, 2.0)
        else:
            scaler = StandardScaler().fit(train)
            train_transformed = scaler.transform(train)
            query_transformed = scaler.transform(queries)
            pairwise = np.linalg.norm(
                query_transformed[:, None, :] - train_transformed[None, :, :], axis=2
            )
        distances.extend(np.min(pairwise, axis=1).tolist())
    positive = np.asarray([value for value in distances if value > 0], dtype=np.float64)
    if positive.size == 0:
        return np.asarray([float(np.finfo(float).eps)])
    quantiles = np.linspace(0.0, 1.0, min(24, positive.size))
    candidates = np.unique(np.quantile(positive, quantiles))
    return np.unique(np.concatenate((candidates, [np.nextafter(float(np.max(positive)), np.inf)])))


def calibrate_grouped_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    capture_ids: np.ndarray,
    augmented: np.ndarray,
    negative_features: np.ndarray,
    *,
    distance_mode: DistanceMode = "standardized_euclidean",
    neighbor_candidates: Iterable[int] = (1, 3, 5, 7),
    prototype_weight_candidates: Iterable[float] = (0.25, 0.5, 0.75),
    spread_multiplier_candidates: Iterable[float] = (1.5, 2.0, 3.0),
    margin_ratio_candidates: Iterable[float] = (0.0, 0.05, 0.1, 0.2),
    temperature_ratio_candidates: Iterable[float] = (0.25, 0.5, 1.0),
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[TapClassifier, GroupedCalibrationResult]:
    """Calibrate with capture-grouped folds and explicit unknown examples."""
    x = _as_feature_matrix(features, "features")
    y = np.asarray(labels, dtype=np.str_)
    groups = np.asarray(capture_ids, dtype=np.int64)
    augmented_mask = np.asarray(augmented, dtype=np.bool_)
    negatives = _as_feature_matrix(negative_features, "negative_features")
    if y.shape != (len(x),) or groups.shape != y.shape or augmented_mask.shape != y.shape:
        raise ValueError("labels and capture metadata must align with feature rows")
    if negatives.shape[1] != x.shape[1]:
        raise ValueError("negative feature dimension differs from positive features")
    if len(negatives) < 10:
        raise ValueError("interactive calibration needs at least ten unknown captures")
    selected_mode = validate_distance_mode(distance_mode)
    fold_assignments, fold_count = _grouped_folds(y, groups, augmented_mask)
    thresholds = _distance_threshold_candidates(
        x,
        y,
        groups,
        augmented_mask,
        negatives,
        fold_assignments,
        fold_count,
        selected_mode,
    )
    neighbors = sorted({int(value) for value in neighbor_candidates if int(value) > 0})
    weights = sorted({float(value) for value in prototype_weight_candidates})
    spreads = sorted({float(value) for value in spread_multiplier_candidates})
    margins = sorted({float(value) for value in margin_ratio_candidates})
    temperatures = sorted({float(value) for value in temperature_ratio_candidates})
    if not all((neighbors, weights, spreads, margins, temperatures)):
        raise ValueError("calibration candidate sets must not be empty")
    candidate_total = len(neighbors) * len(thresholds) * len(weights) * len(spreads) * len(margins)
    candidate_total += len(temperatures)
    evaluated = 0
    best_key: tuple[float, ...] | None = None
    best_values: tuple[int, float, float, float, float] | None = None
    best_known = 0.0
    best_unknown = 0.0
    original = ~augmented_mask

    for neighbor_count in neighbors:
        for threshold in thresholds:
            for weight in weights:
                for spread in spreads:
                    for margin_ratio in margins:
                        known_correct = 0
                        known_total = 0
                        unknown_rejected = 0
                        unknown_total = 0
                        for fold in range(fold_count):
                            train_mask = fold_assignments != fold
                            validation_mask = (fold_assignments == fold) & original
                            classifier = _build_grouped_classifier(
                                x[train_mask],
                                y[train_mask],
                                groups[train_mask],
                                augmented_mask[train_mask],
                                n_neighbors=neighbor_count,
                                distance_threshold=float(threshold),
                                distance_mode=selected_mode,
                                prototype_weight=weight,
                                spread_multiplier=spread,
                                margin_threshold=margin_ratio * threshold,
                                probability_temperature=0.5 * threshold,
                            )
                            for vector, truth in zip(
                                x[validation_mask], y[validation_mask], strict=True
                            ):
                                decision = classifier.predict_detailed(vector)
                                known_correct += int(
                                    decision.accepted and decision.label == str(truth)
                                )
                                known_total += 1
                            for vector in negatives:
                                unknown_rejected += int(
                                    not classifier.predict_detailed(vector).accepted
                                )
                                unknown_total += 1
                        known_accuracy = known_correct / known_total
                        unknown_rejection = unknown_rejected / unknown_total
                        objective = (known_accuracy + unknown_rejection) / 2.0
                        key = (
                            objective,
                            unknown_rejection,
                            known_accuracy,
                            margin_ratio * threshold,
                            -float(threshold),
                            -float(neighbor_count),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_values = (
                                neighbor_count,
                                float(threshold),
                                weight,
                                spread,
                                margin_ratio * threshold,
                            )
                            best_known = known_accuracy
                            best_unknown = unknown_rejection
                        evaluated += 1
                        if progress_callback is not None:
                            progress_callback(evaluated, candidate_total)

    assert best_values is not None and best_key is not None
    neighbor_count, threshold, weight, spread, margin_threshold = best_values
    best_temperature = 0.5 * threshold
    best_nll = float("inf")
    for temperature_ratio in temperatures:
        probability_values: list[float] = []
        for fold in range(fold_count):
            train_mask = fold_assignments != fold
            validation_mask = (fold_assignments == fold) & original
            classifier = _build_grouped_classifier(
                x[train_mask],
                y[train_mask],
                groups[train_mask],
                augmented_mask[train_mask],
                n_neighbors=neighbor_count,
                distance_threshold=threshold,
                distance_mode=selected_mode,
                prototype_weight=weight,
                spread_multiplier=spread,
                margin_threshold=margin_threshold,
                probability_temperature=temperature_ratio * threshold,
            )
            for vector, truth in zip(x[validation_mask], y[validation_mask], strict=True):
                decision = classifier.predict_detailed(vector)
                probability_values.append(max(decision.probabilities.get(str(truth), 0.0), 1e-15))
        nll = float(-np.mean(np.log(probability_values)))
        if (nll, temperature_ratio) < (best_nll, best_temperature / threshold):
            best_nll = nll
            best_temperature = temperature_ratio * threshold
        evaluated += 1
        if progress_callback is not None:
            progress_callback(evaluated, candidate_total)

    final_classifier = _build_grouped_classifier(
        x,
        y,
        groups,
        augmented_mask,
        n_neighbors=neighbor_count,
        distance_threshold=threshold,
        distance_mode=selected_mode,
        prototype_weight=weight,
        spread_multiplier=spread,
        margin_threshold=margin_threshold,
        probability_temperature=best_temperature,
    )
    result = GroupedCalibrationResult(
        n_neighbors=neighbor_count,
        distance_threshold=threshold,
        prototype_weight=weight,
        spread_multiplier=spread,
        margin_threshold=margin_threshold,
        probability_temperature=best_temperature,
        known_probability_nll=best_nll,
        objective=best_key[0],
        known_end_to_end_accuracy=best_known,
        unknown_rejection_rate=best_unknown,
        fold_count=fold_count,
        evaluated_candidates=evaluated,
    )
    return final_classifier, result


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
