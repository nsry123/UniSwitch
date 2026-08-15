import numpy as np

from src.controller import SwitchController
from src.ml.classifier import TapClassifier


def test_registration_transitions_and_persists(tmp_path) -> None:
    state_path = tmp_path / "state.npz"
    classifier = TapClassifier(n_neighbors=3, distance_threshold=2.5)
    controller = SwitchController(
        classifier,
        state_path=state_path,
        state_metadata={"feature_extractor": "legacy"},
    )
    vector = np.array([1.0, 2.0, 3.0])

    controller.set_registration_mode("Desk", taps=3)
    for offset in (0.0, 0.01, -0.01):
        assert controller.audio_callback(vector + offset, tap_count=2) is None

    assert controller.mode == "execute"
    assert controller.target_label is None
    assert controller.taps_collected == 3
    assert state_path.exists()
    assert controller.audio_callback(vector, tap_count=1) == "Desk"


def test_registration_counts_capture_windows_not_onsets() -> None:
    controller = SwitchController(TapClassifier(n_neighbors=1))
    controller.set_registration_mode("wall", taps=2)
    controller.audio_callback(np.ones(3), tap_count=5)
    assert controller.samples_collected == 1
    assert controller.mode == "register"
