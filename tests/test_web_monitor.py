import json
from urllib.request import urlopen

import numpy as np

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
    assert state["threshold"] == 0.01
    assert state["sampleRate"] == 16000
