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


def test_classifier_rejects_bad_features() -> None:
    classifier = TapClassifier()
    with pytest.raises(ValueError, match="one-dimensional"):
        classifier.register_tap(np.ones((2, 2)), "desk")
    with pytest.raises(ValueError, match="label"):
        classifier.register_tap(np.ones(3), "  ")
    classifier.register_tap(np.ones(3), "desk")
    with pytest.raises(ValueError, match="dimension mismatch"):
        classifier.register_tap(np.ones(4), "desk")
