import numpy as np

from src.controller import EventStatus, SwitchController
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


def test_temporal_confirmation_unknown_and_cooldown() -> None:
    now = [10.0]
    classifier = TapClassifier(n_neighbors=1, distance_threshold=2.5)
    for offset in (0.0, 0.01, -0.01):
        classifier.register_tap(np.array([1.0, 2.0, 3.0]) + offset, "desk")
    controller = SwitchController(
        classifier,
        confirmation_count=2,
        event_cooldown=2.0,
        clock=lambda: now[0],
    )

    first = controller.process_capture(np.array([1.0, 2.0, 3.0]), tap_count=1)
    second = controller.process_capture(np.array([1.0, 2.0, 3.0]), tap_count=1)
    suppressed = controller.process_capture(np.array([1.0, 2.0, 3.0]), tap_count=1)
    unknown = controller.process_capture(np.array([100.0, 200.0, 300.0]), tap_count=1)
    now[0] += 2.1
    resumed = controller.process_capture(np.array([1.0, 2.0, 3.0]), tap_count=1)
    reconfirmed = controller.process_capture(np.array([1.0, 2.0, 3.0]), tap_count=1)

    assert first.status is EventStatus.NONE
    assert second.status is EventStatus.EVENT
    assert suppressed.status is EventStatus.NONE
    assert unknown.status is EventStatus.UNKNOWN
    assert resumed.status is EventStatus.NONE
    assert reconfirmed.status is EventStatus.EVENT


def test_default_controller_triggers_on_first_accepted_capture() -> None:
    classifier = TapClassifier(n_neighbors=1, distance_threshold=2.5)
    for offset in (0.0, 0.01, -0.01):
        classifier.register_tap(np.array([1.0, 2.0, 3.0]) + offset, "desk")
    controller = SwitchController(classifier, event_cooldown=0.0)

    decision = controller.process_capture(np.array([1.0, 2.0, 3.0]), tap_count=1)

    assert decision.status is EventStatus.EVENT
    assert decision.label == "desk"


def test_delete_class_updates_state_and_removes_empty_state_file(tmp_path) -> None:
    state_path = tmp_path / "state.npz"
    classifier = TapClassifier(n_neighbors=1, distance_threshold=2.5)
    for label, center in (("desk", 0.0), ("wall", 10.0)):
        for offset in (0.0, 0.01, -0.01):
            classifier.register_tap(np.array([center + offset, center + 1.0]), label)
    classifier.save(state_path)
    controller = SwitchController(classifier, state_path=state_path)

    assert controller.delete_class("desk") == 3
    assert state_path.exists()
    assert set(TapClassifier.load(state_path).y_labels) == {"wall"}
    assert controller.delete_class("wall") == 3
    assert not state_path.exists()
