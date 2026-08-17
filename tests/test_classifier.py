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
    classifier.register_tap(np.array([1.0, 2.0, 3.0]), "desk")

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
    classifier.register_tap(np.array([1.0, 1e-12, 0.0]), "desk")
    classifier.register_tap(np.array([1.0, 2e-12, 0.0]), "desk")

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
    classifier.register_tap(np.array([1.0, 2.0, 3.0]), "desk")
    classifier.save(path, {"feature_extractor": "pann", "sample_rate": 32000})

    restored = TapClassifier.load(path)

    assert restored.distance_mode == "cosine"
    assert restored.threshold_mode == "fixed"
    assert restored.distance_threshold == 0.2
    assert restored.predict(np.array([2.0, 4.0, 6.0])) == "desk"
    assert restored._nearest_distance(np.array([2.0, 4.0, 6.0])) == pytest.approx(
        0.0, abs=1e-12
    )


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


def write_schema_v2_cosine_state(path) -> None:
    features = np.vstack([unit_vector(angle) for angle in (0.0, 0.1, 0.2, 0.3, 0.4)])
    manifest = {
        "schema_version": 2,
        "n_neighbors": 1,
        "distance_threshold": 0.2,
        "distance_mode": "cosine",
        "feature_dim": 2,
        "metadata": {"feature_extractor": "pann"},
    }
    np.savez_compressed(
        path,
        features=features,
        labels=np.array(["desk"] * len(features)),
        manifest=np.asarray(json.dumps(manifest)),
    )


def test_schema_v1_legacy_state_remains_compatible(tmp_path) -> None:
    path = tmp_path / "legacy-v1.npz"
    write_schema_v1_state(path, "legacy")

    restored = TapClassifier.load(path)

    assert restored.distance_mode == "standardized_euclidean"
    assert restored.predict(np.array([1.0, 2.0, 3.0])) == "desk"


def test_schema_v1_pann_state_requires_reregistration(tmp_path) -> None:
    path = tmp_path / "pann-v1.npz"
    write_schema_v1_state(path, "pann")

    with pytest.raises(ValueError, match="re-register using a new --state-path"):
        TapClassifier.load(path)


def test_schema_v2_cosine_state_recomputes_auto_thresholds(tmp_path) -> None:
    path = tmp_path / "pann-v2.npz"
    write_schema_v2_cosine_state(path)

    restored = TapClassifier.load(path)

    assert restored.threshold_mode == "auto"
    assert restored.threshold_for_label("desk") < 0.2
    assert restored.threshold_calibrations["desk"].source == "registration"


def test_schema_v3_preserves_validation_calibration(tmp_path) -> None:
    path = tmp_path / "pann-v3.npz"
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=0.2,
        distance_mode="cosine",
        threshold_mode="auto",
    )
    for angle in (0.0, 0.1, 0.2, 0.3, 0.4):
        classifier.register_tap(unit_vector(angle), "desk")
    classifier.set_validation_calibration("desk", 0.05, np.array([0.01, 0.02]))
    classifier.save(path, {"feature_extractor": "pann"})

    restored = TapClassifier.load(path)

    assert restored.threshold_mode == "auto"
    assert restored.threshold_for_label("desk") == 0.05
    assert restored.threshold_calibrations["desk"].source == "validation"
    assert restored.threshold_calibrations["desk"].sample_count == 2
