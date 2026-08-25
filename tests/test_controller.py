import numpy as np
import pytest

from src.controller import CapturePurpose, EventStatus, SwitchController
from src.ml.classifier import TapClassifier
from src.ml.evaluation import GroupedCalibrationResult


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


def test_registration_stores_augmented_exemplars_but_counts_one_capture() -> None:
    classifier = TapClassifier(n_neighbors=1)
    controller = SwitchController(classifier)
    controller.set_registration_mode("wall", taps=2)

    controller.process_capture(
        np.array([1.0, 2.0]),
        tap_count=1,
        augmented_features=(np.array([2.0, 3.0]), np.array([3.0, 4.0])),
    )

    assert controller.samples_collected == 1
    assert classifier.class_counts() == {"wall": 3}
    assert classifier.class_capture_counts() == {"wall": 1}
    assert not classifier.is_trained
    assert controller.is_registering()


def test_unknown_collection_cancel_and_clear() -> None:
    controller = SwitchController(TapClassifier(n_neighbors=1))
    controller.set_unknown_collection_mode(2)

    decision = controller.process_capture(np.array([3.0, 4.0]), tap_count=1)

    assert decision.purpose is CapturePurpose.UNKNOWN_CALIBRATION
    assert len(controller.classifier.negative_data) == 1
    assert controller.cancel_collection()
    assert controller.mode == "execute"
    assert controller.clear_unknowns() == 1
    assert controller.calibration_snapshot()["unknownCaptureCount"] == 0


def calibration_ready_controller() -> SwitchController:
    classifier = TapClassifier(n_neighbors=1, distance_threshold=2.5)
    for capture_id in range(4):
        classifier.register_tap(
            np.array([1.0, 2.0 + capture_id * 0.01]),
            "desk",
            capture_id=capture_id,
        )
    for capture_id in range(4, 14):
        classifier.register_unknown(np.array([20.0, float(capture_id)]), capture_id=capture_id)
    return SwitchController(classifier)


def grouped_result(known: float, unknown: float) -> GroupedCalibrationResult:
    return GroupedCalibrationResult(
        n_neighbors=1,
        distance_threshold=0.3,
        prototype_weight=0.5,
        spread_multiplier=2.0,
        margin_threshold=0.03,
        probability_temperature=0.15,
        known_probability_nll=0.0,
        objective=(known + unknown) / 2.0,
        known_end_to_end_accuracy=known,
        unknown_rejection_rate=unknown,
        fold_count=4,
        evaluated_candidates=12,
    )


def test_calibration_applies_only_after_quality_gate(monkeypatch) -> None:
    controller = calibration_ready_controller()
    original = controller.classifier

    def passing(*_args, **_kwargs):
        candidate = TapClassifier(n_neighbors=1, distance_threshold=0.3)
        candidate.feature_dim = 2
        candidate.X_data = [row.copy() for row in original.X_data]
        candidate.y_labels = list(original.y_labels)
        candidate.capture_ids = list(original.capture_ids)
        candidate.augmented_flags = list(original.augmented_flags)
        candidate._fit_if_ready()
        return candidate, grouped_result(0.9, 0.8)

    monkeypatch.setattr("src.controller.calibrate_grouped_classifier", passing)

    assert controller.calibrate()
    assert controller.classifier is not original
    assert controller.calibration_snapshot()["state"] == "passed"


def test_failed_calibration_retains_classifier(monkeypatch) -> None:
    controller = calibration_ready_controller()
    original = controller.classifier
    monkeypatch.setattr(
        "src.controller.calibrate_grouped_classifier",
        lambda *_args, **_kwargs: (TapClassifier(), grouped_result(0.79, 1.0)),
    )

    assert not controller.calibrate()
    assert controller.classifier is original
    assert controller.calibration_snapshot()["state"] == "failed"


def test_stale_calibration_state_survives_restart(tmp_path) -> None:
    source = calibration_ready_controller().classifier
    source.calibration_report = {
        "status": "passed",
        "state": "passed",
        "message": "Calibration applied successfully.",
        "classCaptureCounts": {"desk": 4},
        "unknownCaptureCount": 10,
        "evaluatedCandidates": 1,
        "totalCandidates": 1,
        "metrics": None,
        "parameters": None,
    }
    state_path = tmp_path / "state.npz"
    controller = SwitchController(source, state_path=state_path)
    controller.set_registration_mode("desk", taps=1)
    controller.process_capture(np.array([1.0, 2.1]), tap_count=1)

    restored = SwitchController(TapClassifier.load(state_path))

    assert restored.calibration_snapshot()["state"] == "stale"


def test_calibration_requires_unknown_and_independent_class_captures() -> None:
    controller = SwitchController(TapClassifier())
    with pytest.raises(ValueError, match="Register at least one class"):
        controller.calibrate()


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
