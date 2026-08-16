import queue
import threading

import numpy as np

from src.audio.listener import AudioListener, CapturedAudio
from src.audio.processor import SignalProcessor
from src.controller import SwitchController
from src.ml.classifier import TapClassifier


def test_synthetic_audio_reaches_registration_and_inference(tmp_path) -> None:
    processor = SignalProcessor(model="legacy")
    waveform = np.zeros(processor.sample_rate * 2, dtype=np.float32)
    waveform[[4000, 12000, 22000]] = 1.0
    classifier = TapClassifier(n_neighbors=3, distance_threshold=2.5)
    state_path = tmp_path / "registrations.npz"
    controller = SwitchController(
        classifier,
        state_path=state_path,
        state_metadata={"feature_extractor": "legacy", "sample_rate": 16000},
    )

    listener = AudioListener.__new__(AudioListener)
    listener.processor = processor
    listener._capture_queue = queue.Queue()
    listener._stop_event = threading.Event()
    listener.last_error = None
    listener.threshold = 0.015
    callback_results: list[str | None] = []

    def callback(features: np.ndarray, tap_count: int) -> None:
        callback_results.append(controller.audio_callback(features, tap_count))

    worker = threading.Thread(target=listener._worker_loop, args=(callback,))
    worker.start()
    controller.set_registration_mode("desk", taps=3)
    for _ in range(3):
        listener._capture_queue.put(CapturedAudio(waveform.copy(), 0))
    listener._capture_queue.join()
    listener._capture_queue.put(CapturedAudio(waveform.copy(), 0))
    listener._capture_queue.join()
    listener.stop()
    worker.join(timeout=2.0)

    assert callback_results == [None, None, None, "desk"]
    assert state_path.exists()
    restored = TapClassifier.load(state_path)
    assert restored.metadata["feature_extractor"] == "legacy"
    assert restored.predict(processor.extract_features(waveform)) == "desk"
