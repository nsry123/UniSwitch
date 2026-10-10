import json

import numpy as np
import pytest

from src.ml.classifier import TapClassifier


def trained_classifier() -> TapClassifier:
    classifier = TapClassifier(n_neighbors=3, distance_threshold=2.5)
    for offset in (0.0, 0.01, -0.01):
        classifier.register_tap(np.array([1.0, 2.0, 3.0]) + offset, "desk")
    return classifier


def test_classifier_matches_and_rejects() -> None:
    classifier = trained_classifier()
    assert classifier.predict(np.array([1.0, 2.0, 3.0])) == "desk"
    assert classifier.predict(np.array([100.0, 200.0, 300.0])) == "Unknown"


def test_classifier_state_round_trip(tmp_path) -> None:
    path = tmp_path / "registrations.npz"
    classifier = trained_classifier()
    classifier.save(path, {"feature_extractor": "legacy", "sample_rate": 16000})

    restored = TapClassifier.load(path)

    assert restored.predict(np.array([1.0, 2.0, 3.0])) == "desk"
    assert restored.metadata["feature_extractor"] == "legacy"
    assert restored.feature_dim == 3
    assert restored.distance_mode == "standardized_euclidean"


def test_physical_capture_ids_control_eligibility_and_round_trip(tmp_path) -> None:
    classifier = TapClassifier(n_neighbors=1, distance_threshold=2.5)
    for capture_id in range(3):
        classifier.register_tap(
            np.array([1.0, 2.0 + capture_id * 0.01]),
            "desk",
            capture_id=capture_id,
        )
        classifier.register_tap(
            np.array([1.0, 2.0 + capture_id * 0.01]) * 0.5,
            "desk",
            announce=False,
            capture_id=capture_id,
            augmented=True,
        )
        if capture_id < 2:
            assert not classifier.is_trained

    classifier.register_unknown(np.array([20.0, 30.0]), capture_id=3)
    classifier.calibration_report = {"status": "passed", "state": "passed"}
    path = tmp_path / "schema-v4.npz"
    classifier.save(path)
    restored = TapClassifier.load(path)

    assert restored.is_trained
    assert restored.class_capture_counts() == {"desk": 3}
    assert restored.class_counts() == {"desk": 6}
    assert restored.capture_ids == [0, 0, 1, 1, 2, 2]
    assert restored.augmented_flags == [False, True] * 3
    assert len(restored.negative_data) == 1
    assert restored.calibration_report == {"state": "passed", "status": "passed"}


def test_classifier_rejects_bad_features() -> None:
    classifier = TapClassifier()
    with pytest.raises(ValueError, match="one-dimensional"):
        classifier.register_tap(np.ones((2, 2)), "desk")
    with pytest.raises(ValueError, match="label"):
        classifier.register_tap(np.ones(3), "  ")
    classifier.register_tap(np.ones(3), "desk")
    with pytest.raises(ValueError, match="dimension mismatch"):
        classifier.register_tap(np.ones(4), "desk")


def test_cosine_distance_is_scale_invariant() -> None:
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=0.01,
        distance_mode="cosine",
    )
    for scale in (1.0, 2.0, 3.0):
        classifier.register_tap(scale * np.array([1.0, 2.0, 3.0]), "desk")

    base_distance = classifier._nearest_distance(np.array([1.0, 2.0, 3.0]))
    scaled_distance = classifier._nearest_distance(np.array([10.0, 20.0, 30.0]))

    assert base_distance == pytest.approx(0.0, abs=1e-12)
    assert scaled_distance == pytest.approx(base_distance, abs=1e-12)
    assert classifier.predict(np.array([10.0, 20.0, 30.0])) == "desk"


def test_cosine_distance_does_not_amplify_low_variance_coordinate() -> None:
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=1e-6,
        distance_mode="cosine",
    )
    for value in (1e-12, 2e-12, 3e-12):
        classifier.register_tap(np.array([1.0, value, 0.0]), "desk")

    distance = classifier._nearest_distance(np.array([1.0, 1e-6, 0.0]))

    assert distance < 1e-9
    assert classifier.predict(np.array([1.0, 1e-6, 0.0])) == "desk"


@pytest.mark.parametrize(
    "feature",
    [np.zeros(3), np.array([np.nan, 1.0, 2.0]), np.array([np.inf, 1.0, 2.0])],
)
def test_cosine_classifier_rejects_invalid_features(feature) -> None:
    classifier = TapClassifier(distance_mode="cosine")
    with pytest.raises(ValueError):
        classifier.register_tap(feature, "desk")


def test_cosine_classifier_state_round_trip(tmp_path) -> None:
    path = tmp_path / "registrations.npz"
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=0.2,
        distance_mode="cosine",
    )
    for scale in (1.0, 2.0, 3.0):
        classifier.register_tap(scale * np.array([1.0, 2.0, 3.0]), "desk")
    classifier.save(path, {"feature_extractor": "pann", "sample_rate": 32000})

    restored = TapClassifier.load(path)

    assert restored.distance_mode == "cosine"
    assert restored.threshold_mode == "fixed"
    assert restored.distance_threshold == 0.2
    assert restored.prototype_weight == 0.5
    assert restored.spread_multiplier == 2.0
    assert restored.margin_threshold == pytest.approx(0.02)
    assert restored.probability_temperature == pytest.approx(0.1)
    assert restored.predict(np.array([2.0, 4.0, 6.0])) == "desk"
    assert restored._nearest_distance(np.array([2.0, 4.0, 6.0])) == pytest.approx(0.0, abs=1e-12)


def unit_vector(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)])


def test_auto_thresholds_reflect_each_labels_registration_spread() -> None:
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=0.2,
        distance_mode="cosine",
        threshold_mode="auto",
    )
    for angle in (0.0, 0.005, 0.01, 0.015, 0.02):
        classifier.register_tap(unit_vector(angle), "tight")
    for angle in (1.0, 1.2, 1.4, 1.6, 1.8):
        classifier.register_tap(unit_vector(angle), "variable")

    assert classifier.threshold_for_label("tight") == 0.01
    assert classifier.threshold_for_label("variable") > 0.01
    assert classifier.threshold_calibrations["tight"].source == "registration"
    assert classifier.threshold_calibrations["variable"].source == "registration"


def test_auto_threshold_uses_fallback_until_five_samples() -> None:
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=0.2,
        distance_mode="cosine",
        threshold_mode="auto",
    )
    for angle in (0.0, 0.1, 0.2, 0.3):
        classifier.register_tap(unit_vector(angle), "desk")

    assert classifier.threshold_for_label("desk") == 0.2
    assert classifier.threshold_calibrations["desk"].source == "fallback"

    classifier.register_tap(unit_vector(0.4), "desk")

    assert classifier.threshold_for_label("desk") < 0.2
    assert classifier.threshold_calibrations["desk"].source == "registration"


def test_auto_threshold_is_clamped_to_floor_and_cap() -> None:
    floor_classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=0.2,
        distance_mode="cosine",
        threshold_mode="auto",
    )
    for _index in range(5):
        floor_classifier.register_tap(unit_vector(0.0), "desk")

    cap_classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=0.2,
        distance_mode="cosine",
        threshold_mode="auto",
    )
    for angle in np.linspace(0.0, 2 * np.pi, 5, endpoint=False):
        cap_classifier.register_tap(unit_vector(float(angle)), "desk")

    assert floor_classifier.threshold_for_label("desk") == 0.01
    assert cap_classifier.threshold_for_label("desk") == 0.2


def test_rejection_uses_only_predicted_labels_threshold() -> None:
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=0.2,
        distance_mode="cosine",
        threshold_mode="auto",
    )
    for angle in (0.0, 0.01, 0.02, 0.03, 0.04):
        classifier.register_tap(unit_vector(angle), "desk")
        classifier.register_tap(unit_vector(1.5 + angle), "wall")
    query = unit_vector(0.2)
    classifier.set_validation_calibration("desk", 0.01, np.array([0.005]))
    classifier.set_validation_calibration("wall", 0.2, np.array([0.05]))

    assert classifier.predict(query) == "Unknown"

    classifier.set_validation_calibration("desk", 0.2, np.array([0.05]))

    assert classifier.predict(query) == "desk"


def write_schema_v1_state(path, feature_extractor: str) -> None:
    manifest = {
        "schema_version": 1,
        "n_neighbors": 1,
        "distance_threshold": 2.5,
        "feature_dim": 3,
        "metadata": {"feature_extractor": feature_extractor},
    }
    np.savez_compressed(
        path,
        features=np.array([[1.0, 2.0, 3.0]]),
        labels=np.array(["desk"]),
        manifest=np.asarray(json.dumps(manifest)),
    )


def test_schema_v1_legacy_state_requires_reregistration(tmp_path) -> None:
    path = tmp_path / "legacy-v1.npz"
    write_schema_v1_state(path, "legacy")

    with pytest.raises(ValueError, match="re-register using a new --state-path"):
        TapClassifier.load(path)


def test_schema_v1_pann_state_requires_reregistration(tmp_path) -> None:
    path = tmp_path / "pann-v1.npz"
    write_schema_v1_state(path, "pann")

    with pytest.raises(ValueError, match="re-register using a new --state-path"):
        TapClassifier.load(path)


def test_detailed_prediction_exposes_probabilities_margin_and_spread() -> None:
    classifier = TapClassifier(
        n_neighbors=2,
        distance_threshold=0.5,
        distance_mode="cosine",
        margin_threshold=0.05,
    )
    for vector in ([1.0, 0.0], [1.0, 0.02], [1.0, -0.02]):
        classifier.register_tap(np.asarray(vector), "desk")
    classifier.register_tap(np.array([1.0, 0.01]), "desk")
    for vector in ([0.0, 1.0], [0.02, 1.0], [-0.02, 1.0]):
        classifier.register_tap(np.asarray(vector), "wall")

    result = classifier.predict_detailed(np.array([1.0, 0.01]))

    assert result.accepted
    assert result.label == "desk"
    assert tuple(result.probabilities) == ("desk", "wall")
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.prototype_distance is not None
    assert result.exemplar_distance is not None
    assert result.acceptance_radius is not None
    assert result.score <= result.acceptance_radius
    assert result.margin >= classifier.margin_threshold


def test_ambiguous_query_is_rejected_by_margin() -> None:
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=2.0,
        distance_mode="cosine",
        margin_threshold=0.1,
    )
    for label, vector in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
        for _ in range(3):
            classifier.register_tap(np.asarray(vector), label)

    result = classifier.predict_detailed(np.array([1.0, 1.0]))

    assert not result.accepted
    assert result.label == "Unknown"
    assert result.rejection_reason == "margin"


def test_schema_v2_requires_reregistration(tmp_path) -> None:
    old_path = tmp_path / "schema-v2.npz"
    manifest = {
        "schema_version": 2,
        "n_neighbors": 1,
        "distance_threshold": 0.2,
        "distance_mode": "cosine",
        "feature_dim": 2,
        "metadata": {"feature_extractor": "pann"},
    }
    np.savez_compressed(
        old_path,
        features=np.array([[1.0, 0.0], [1.0, 0.01], [1.0, -0.01]]),
        labels=np.array(["desk", "desk", "desk"]),
        manifest=np.asarray(json.dumps(manifest)),
    )

    with pytest.raises(ValueError, match="re-register using a new --state-path"):
        TapClassifier.load(old_path)


def test_schema_v3_requires_reregistration(tmp_path) -> None:
    path = tmp_path / "schema-v3.npz"
    manifest = {
        "schema_version": 3,
        "n_neighbors": 1,
        "distance_threshold": 0.2,
        "distance_mode": "cosine",
        "feature_dim": 2,
        "metadata": {"feature_extractor": "beats"},
    }
    np.savez_compressed(
        path,
        features=np.array([[1.0, 0.0]]),
        labels=np.array(["desk"]),
        manifest=np.asarray(json.dumps(manifest)),
    )

    with pytest.raises(ValueError, match="re-register using a new --state-path"):
        TapClassifier.load(path)


def test_delete_class_removes_examples_and_rebuilds_statistics() -> None:
    classifier = TapClassifier(n_neighbors=1, distance_threshold=2.5)
    for label, center in (("desk", 0.0), ("wall", 10.0)):
        for offset in (0.0, 0.01, -0.01):
            classifier.register_tap(np.array([center + offset, center + 1.0]), label)

    deleted = classifier.delete_class("desk")

    assert deleted == 3
    assert set(classifier.y_labels) == {"wall"}
    assert classifier.predict(np.array([10.0, 11.0])) == "wall"
    with pytest.raises(ValueError, match="not registered"):
        classifier.delete_class("desk")


def test_class_counts_lists_all_labels_in_deterministic_order() -> None:
    classifier = TapClassifier()
    classifier.register_tap(np.ones(3), "wall")
    classifier.register_tap(np.ones(3), "desk")
    classifier.register_tap(np.ones(3), "wall")

    assert classifier.class_counts() == {"desk": 1, "wall": 2}
