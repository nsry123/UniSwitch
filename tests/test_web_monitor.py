import json
from urllib.request import urlopen

import numpy as np

from src.controller import EventStatus, SwitchController
from src.ml.classifier import TapClassifier
from src.web.app import MicrophoneMonitor


def test_monitor_publishes_audio_snapshot() -> None:
    monitor = MicrophoneMonitor(threshold=0.015, sample_rate=32000, port=0)

    monitor.publish(np.array([-0.25, 0.5], dtype=np.float32), 0.02, True)
    snapshot = monitor.snapshot()

    assert snapshot["sequence"] == 1
    assert snapshot["energy"] == 0.02
    assert snapshot["peak"] == 0.5
    assert snapshot["threshold"] == 0.015
    assert snapshot["capturing"] is True
    assert snapshot["active"] is True
    assert snapshot["waveform"] == [-0.25, 0.5]
    assert snapshot["lastResult"] is None
    assert snapshot["calibration"]["state"] == "needs_data"


def test_monitor_publishes_calibration_and_unknown_capture() -> None:
    monitor = MicrophoneMonitor(threshold=0.015, sample_rate=16000)
    controller = SwitchController(TapClassifier())
    controller.set_unknown_collection_mode(1)
    decision = controller.process_capture(np.array([1.0, 2.0]), tap_count=1)
    monitor.publish_decision(decision)
    monitor.publish_calibration(controller.calibration_snapshot())

    snapshot = monitor.snapshot()
    assert snapshot["lastResult"]["purpose"] == "unknown_calibration"
    assert snapshot["lastResult"]["label"] == "Unknown calibration"
    assert snapshot["calibration"]["unknownCaptureCount"] == 1


def test_monitor_publishes_last_knock_result_and_reason() -> None:
    classifier = TapClassifier(n_neighbors=1, distance_threshold=2.5)
    for offset in (0.0, 0.01, -0.01):
        classifier.register_tap(np.array([1.0, 2.0, 3.0]) + offset, "desk")
    controller = SwitchController(classifier, event_cooldown=0.0)
    monitor = MicrophoneMonitor(threshold=0.015, sample_rate=16000)

    accepted = controller.process_capture(np.array([1.0, 2.0, 3.0]), tap_count=1)
    monitor.publish_decision(accepted)
    accepted_result = monitor.snapshot()["lastResult"]
    unknown = controller.process_capture(np.array([100.0, 200.0, 300.0]), tap_count=1)
    monitor.publish_decision(unknown)
    unknown_result = monitor.snapshot()["lastResult"]

    assert accepted.status is EventStatus.EVENT
    assert accepted_result["label"] == "desk"
    assert accepted_result["accepted"] is True
    assert "within" in accepted_result["reason"]
    assert unknown.status is EventStatus.UNKNOWN
    assert unknown_result["label"] == "Unknown"
    assert unknown_result["rejectionReason"] == "distance"
    assert "exceeds" in unknown_result["reason"]


def test_monitor_serves_dashboard_and_audio_api() -> None:
    monitor = MicrophoneMonitor(threshold=0.01, sample_rate=16000, port=0)
    monitor.start(open_browser=False)
    try:
        with urlopen(f"{monitor.url}/", timeout=2) as response:
            page = response.read().decode()
        with urlopen(f"{monitor.url}/api/audio", timeout=2) as response:
            state = json.load(response)
    finally:
        monitor.stop()

    assert "Microphone monitor" in page
    assert "Latest knock" in page
    assert "Interactive calibration" in page
    assert state["threshold"] == 0.01
    assert state["sampleRate"] == 16000
