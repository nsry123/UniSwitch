import queue
import threading
from collections import deque

import numpy as np

from src.audio.listener import AudioListener


class StubProcessor:
    sample_rate = 16000

    def calculate_energy(self, waveform: np.ndarray) -> float:
        return float(np.sqrt(np.mean(waveform**2)))

    def count_taps(self, waveform: np.ndarray) -> int:
        return int(np.count_nonzero(waveform))

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        return np.array([waveform.mean(), waveform.max()])


class FailingProcessor(StubProcessor):
    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        raise RuntimeError("feature failure")


def worker_only_listener(processor: StubProcessor) -> AudioListener:
    listener = AudioListener.__new__(AudioListener)
    listener.processor = processor
    listener._capture_queue = queue.Queue()
    listener._stop_event = threading.Event()
    listener.last_error = None
    return listener


def test_worker_delivers_processed_capture_without_microphone() -> None:
    listener = worker_only_listener(StubProcessor())
    received: list[tuple[np.ndarray, int]] = []
    callback_event = threading.Event()

    def callback(features: np.ndarray, tap_count: int) -> None:
        received.append((features, tap_count))
        callback_event.set()

    worker = threading.Thread(target=listener._worker_loop, args=(callback,))
    worker.start()
    listener._capture_queue.put(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert callback_event.wait(timeout=2.0)
    listener.stop()
    worker.join(timeout=2.0)

    assert received[0][1] == 1
    np.testing.assert_allclose(received[0][0], [1 / 3, 1.0])


def test_worker_surfaces_processing_failure() -> None:
    listener = worker_only_listener(FailingProcessor())
    errors: list[Exception] = []
    worker = threading.Thread(
        target=listener._worker_loop,
        args=(lambda _features, _count: None, errors.append),
    )
    worker.start()
    listener._capture_queue.put(np.ones(3, dtype=np.float32))
    worker.join(timeout=2.0)

    assert isinstance(listener.last_error, RuntimeError)
    assert errors and str(errors[0]) == "feature failure"
    assert listener._stop_event.is_set()


def test_listener_reports_live_frames_and_capture_state() -> None:
    listener = AudioListener.__new__(AudioListener)
    listener.processor = StubProcessor()
    listener.chunk_size = 3
    listener.threshold = 0.1
    listener.pre_roll_chunks = 1
    listener.record_chunks = 1
    listener.buffer = deque(maxlen=1)
    listener._capture_queue = queue.Queue()
    listener._stop_event = threading.Event()
    updates: list[tuple[float, bool]] = []

    def frame_callback(_frame: np.ndarray, energy: float, capturing: bool) -> None:
        updates.append((energy, capturing))
        if len(updates) == 2:
            listener.stop()

    listener.frame_callback = frame_callback

    class StubStream:
        frames = iter(
            (
                np.array([0.0, 0.5, 0.0], dtype=np.float32).tobytes(),
                np.zeros(3, dtype=np.float32).tobytes(),
            )
        )

        def read(self, _chunk_size: int, exception_on_overflow: bool = False) -> bytes:
            return next(self.frames)

        def stop_stream(self) -> None:
            pass

        def close(self) -> None:
            pass

    listener._listener_loop(StubStream())

    assert updates[0][0] > listener.threshold
    assert updates == [(updates[0][0], True), (0.0, False)]
