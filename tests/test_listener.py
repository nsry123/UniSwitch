import queue
import threading
from collections import deque

import numpy as np

from src.audio.listener import AudioListener, CapturedAudio


class StubProcessor:
    sample_rate = 16000

    def __init__(self) -> None:
        self.count_thresholds: list[float] = []

    def calculate_energy(self, waveform: np.ndarray) -> float:
        return float(np.sqrt(np.mean(waveform**2)))

    def count_taps(self, waveform: np.ndarray, energy_threshold: float = 0.015) -> int:
        self.count_thresholds.append(energy_threshold)
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
    listener.threshold = 0.123
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
    capture = CapturedAudio(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        trigger_sample=3,
    )
    listener._capture_queue.put(capture)
    assert callback_event.wait(timeout=2.0)
    listener.stop()
    worker.join(timeout=2.0)

    assert received[0][1] == 1
    np.testing.assert_allclose(received[0][0], [1 / 3, 1.0])
    assert listener.processor.count_thresholds == [0.123]


def test_worker_augments_only_when_requested() -> None:
    processor = StubProcessor()
    listener = worker_only_listener(processor)
    received: list[tuple[np.ndarray, int, tuple[np.ndarray, ...]]] = []
    callback_event = threading.Event()

    def augmented_callback(
        features: np.ndarray,
        tap_count: int,
        augmented_features: tuple[np.ndarray, ...],
    ) -> None:
        received.append((features, tap_count, augmented_features))
        callback_event.set()

    worker = threading.Thread(
        target=listener._worker_loop,
        args=(
            lambda _features, _count: None,
            None,
            augmented_callback,
            lambda: True,
            (-6.0, 6.0),
        ),
    )
    worker.start()
    waveform = np.array([0.5, 0.0, 0.0, 0.0, 0.25, 0.0], dtype=np.float32)
    listener._capture_queue.put(CapturedAudio(waveform, trigger_sample=3))
    assert callback_event.wait(timeout=2.0)
    listener.stop()
    worker.join(timeout=2.0)

    original, tap_count, variants = received[0]
    np.testing.assert_allclose(original, [0.125, 0.5])
    assert tap_count == 1
    assert len(variants) == 2
    np.testing.assert_allclose(
        variants[0],
        np.array([0.125, 0.5], dtype=np.float32) * (10.0 ** (-6.0 / 20.0)),
    )
    np.testing.assert_allclose(
        variants[1],
        np.array([0.125, 0.5], dtype=np.float32) * (10.0 ** (6.0 / 20.0)),
    )
    assert processor.count_thresholds == [0.123]


def test_worker_skips_augmented_extraction_in_execute_mode() -> None:
    processor = StubProcessor()
    listener = worker_only_listener(processor)
    callback_event = threading.Event()
    received: list[np.ndarray] = []

    def callback(features: np.ndarray, _tap_count: int) -> None:
        received.append(features)
        callback_event.set()

    worker = threading.Thread(
        target=listener._worker_loop,
        args=(callback, None, lambda *_args: None, lambda: False, (-9.0, 6.0)),
    )
    worker.start()
    listener._capture_queue.put(CapturedAudio(np.ones(4, dtype=np.float32), 0))
    assert callback_event.wait(timeout=2.0)
    listener.stop()
    worker.join(timeout=2.0)

    assert len(received) == 1


def test_worker_surfaces_processing_failure() -> None:
    listener = worker_only_listener(FailingProcessor())
    errors: list[Exception] = []
    worker = threading.Thread(
        target=listener._worker_loop,
        args=(lambda _features, _count: None, errors.append),
    )
    worker.start()
    listener._capture_queue.put(CapturedAudio(np.ones(3, dtype=np.float32), 0))
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
    capture = listener._capture_queue.get_nowait()
    assert capture.trigger_sample == 0
