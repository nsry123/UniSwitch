from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import librosa
import numpy as np
import pyaudio

from src.audio.augmentation import apply_gain_db, validate_gain_values

AudioCallback = Callable[[np.ndarray, int], object]
AugmentedAudioCallback = Callable[[np.ndarray, int, tuple[np.ndarray, ...]], object]
ErrorCallback = Callable[[Exception], object]
FrameCallback = Callable[[np.ndarray, float, bool], object]
AugmentationPredicate = Callable[[], bool]


@dataclass(frozen=True)
class CapturedAudio:
    """A complete capture plus the first sample belonging to its trigger chunk."""

    waveform: np.ndarray
    trigger_sample: int


class AudioProcessor(Protocol):
    """Processing contract consumed by the listener."""

    sample_rate: int

    def calculate_energy(self, audio_frame: np.ndarray) -> float: ...

    def count_taps(
        self, audio_clip: np.ndarray, energy_threshold: float = 0.015
    ) -> int: ...

    def extract_features(self, audio_clip: np.ndarray) -> np.ndarray: ...


class AudioStream(Protocol):
    """Subset of the PyAudio stream interface used by capture."""

    def read(self, chunk_size: int, exception_on_overflow: bool = False) -> bytes: ...

    def stop_stream(self) -> None: ...

    def close(self) -> None: ...


class AudioListener:
    def __init__(
        self,
        processor: AudioProcessor,
        device_index: int | None = None,
        threshold: float = 0.015,
        pre_roll_sec: float = 0.2,
        record_sec: float = 1.5,
        frame_callback: FrameCallback | None = None,
    ) -> None:
        """
        Initializes the microphone stream.
        - threshold:     RMS energy level required to trigger a capture.
        - pre_roll_sec:  Audio kept *before* the trigger (captures the attack transient).
        - record_sec:    Audio recorded *after* the trigger. Set to 1.5 s to
                         reliably capture slow triple-knocks.

        Design invariant: stream.read() is called in exactly one place —
        _listener_loop — which runs on the main thread. The worker thread
        receives only fully-assembled numpy arrays; it never touches the stream.
        This eliminates the interleaved-read race that caused a single knock to
        appear as multiple events.
        """
        if threshold < 0 or pre_roll_sec <= 0 or record_sec <= 0:
            raise ValueError(
                "threshold must be non-negative and capture durations must be positive"
            )
        self.processor = processor
        self.threshold = threshold
        self.chunk_size = 1024
        self.device_index = device_index
        self.frame_callback = frame_callback

        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = self.processor.sample_rate

        self.pre_roll_chunks = int((pre_roll_sec * self.rate) / self.chunk_size)
        self.record_chunks = int((record_sec * self.rate) / self.chunk_size)

        # Ring buffer — filled continuously by the listener thread only.
        # Provides the pre-trigger context that captures a knock's attack transient.
        self.buffer: deque[np.ndarray] = deque(maxlen=self.pre_roll_chunks)

        # Handoff point between the listener and worker threads.
        # Each item contains a complete, contiguous waveform and trigger offset.
        # The listener assembles the entire clip (pre-roll + post-trigger) before
        # enqueueing, so the worker never needs to read from the stream.
        self._capture_queue: queue.Queue[CapturedAudio] = queue.Queue()

        # Signals both threads to exit cleanly.
        self._stop_event = threading.Event()
        self.last_error: Exception | None = None

        self.audio = pyaudio.PyAudio()

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        """Request a graceful stop from another thread."""
        self._stop_event.set()

    def listen(
        self,
        trigger_callback: AudioCallback,
        error_callback: ErrorCallback | None = None,
        augmented_trigger_callback: AugmentedAudioCallback | None = None,
        should_augment: AugmentationPredicate | None = None,
        gain_db_values: tuple[float, ...] = (),
    ) -> None:
        """
        Opens the mic and starts two threads:

          1. Listener thread (main)  — sole reader of the audio stream.
             Maintains the pre-roll ring buffer, detects threshold crossings,
             collects the post-trigger window itself, then enqueues the finished
             clip as a single contiguous array.

          2. Worker thread (daemon)  — dequeues completed clips, runs
             count_taps / extract_features, and fires the callback.
             All CPU-heavy work lives here so the listener is never stalled.
        """
        self._stop_event.clear()
        self.last_error = None
        stream = None
        worker = None
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )
            validated_gains = validate_gain_values(gain_db_values)
            worker = threading.Thread(
                target=self._worker_loop,
                args=(
                    trigger_callback,
                    error_callback,
                    augmented_trigger_callback,
                    should_augment,
                    validated_gains,
                ),
                daemon=True,
            )
            worker.start()
            print(f"[*] Mic active. Listening for taps (Threshold: {self.threshold})...")
            self._listener_loop(stream)
        except KeyboardInterrupt:
            print("\n[*] Stopping listener...")
        finally:
            self.stop()
            if worker is not None:
                worker.join(timeout=3.0)
            if stream is not None:
                stream.stop_stream()
                stream.close()
            self.audio.terminate()

    def test_file(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Loads an audio file, simulates the energy gate by trimming silence,
        and extracts features just like the live microphone listener.
        """
        print(f"[*] Loading test file: {file_path}")
        try:
            y, sr = librosa.load(file_path, sr=self.processor.sample_rate, mono=True)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            original_duration = len(y) / sr
            trimmed_duration = len(y_trimmed) / sr
            print(
                f"[*] Audio loaded. Original: {original_duration:.2f}s "
                f"Trimmed: {trimmed_duration:.2f}s"
            )

            tap_count = self.processor.count_taps(y_trimmed, self.threshold)
            features = self.processor.extract_features(y_trimmed)

            print(f"[*] Taps detected: {tap_count}")
            print(f"[*] Feature vector length: {len(features)}")
            return features, tap_count

        except Exception as error:
            raise RuntimeError(f"could not process audio file {file_path}: {error}") from error

    # ------------------------------------------------------------------ #
    #  Private — listener thread (sole owner of the stream)               #
    # ------------------------------------------------------------------ #

    def _listener_loop(self, stream: AudioStream) -> None:
        """
        Runs on the main thread and is the *only* place stream.read() is called.

        State machine:

          IDLE      — filling the ring buffer, watching for threshold crossings.
          CAPTURING — collecting post-trigger chunks into a staging list.
                      The ring buffer is paused during capture so that any
                      leftover knock energy cannot corrupt the next pre-roll.
                      Once record_chunks have been collected, the full clip
                      (pre-roll snapshot + post-trigger frames) is concatenated
                      and placed on the queue for the worker, then the state
                      machine returns to IDLE.

        Because the listener reads every chunk and the worker reads none, the
        assembled clip is always perfectly contiguous — no interleaving gaps
        that would fool count_taps into seeing phantom knocks.
        """
        capturing = False
        post_trigger: list[np.ndarray] = []  # staging list for post-trigger chunks
        pre_roll_snapshot: list[np.ndarray] = []  # frozen ring buffer at trigger
        remaining_chunks = 0

        while not self._stop_event.is_set():
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.float32)
            energy = self.processor.calculate_energy(audio_frame)

            if not capturing:
                # IDLE: keep the ring buffer fresh.
                self.buffer.append(audio_frame)

                if energy > self.threshold:
                    # Freeze the ring buffer. The trigger chunk is already
                    # included via the append above, so it appears exactly once
                    # at the end of pre_roll_snapshot — no double-count at the seam.
                    pre_roll_snapshot = list(self.buffer)
                    post_trigger = []
                    remaining_chunks = self.record_chunks
                    capturing = True

            else:
                # CAPTURING: accumulate post-trigger chunks.
                # The ring buffer is intentionally not updated here; we do not
                # want decaying knock energy to pollute the next event's pre-roll.
                post_trigger.append(audio_frame)
                remaining_chunks -= 1

                if remaining_chunks <= 0:
                    # Assemble and ship the complete, contiguous clip.
                    full_clip = np.concatenate(pre_roll_snapshot + post_trigger)
                    trigger_sample = sum(
                        frame.size for frame in pre_roll_snapshot[:-1]
                    )
                    self._capture_queue.put(CapturedAudio(full_clip, trigger_sample))

                    # Seed the ring buffer with the tail of the just-captured
                    # window so the next event's pre-roll is meaningful.
                    tail = post_trigger[-self.pre_roll_chunks :]
                    self.buffer.clear()
                    self.buffer.extend(tail)

                    capturing = False

            if self.frame_callback is not None:
                try:
                    self.frame_callback(audio_frame, energy, capturing)
                except Exception as error:
                    print(f"[!] Microphone monitor stopped receiving audio: {error}")
                    self.frame_callback = None

    # ------------------------------------------------------------------ #
    #  Private — worker thread                                             #
    # ------------------------------------------------------------------ #

    def _worker_loop(
        self,
        trigger_callback: AudioCallback,
        error_callback: ErrorCallback | None = None,
        augmented_trigger_callback: AugmentedAudioCallback | None = None,
        should_augment: AugmentationPredicate | None = None,
        gain_db_values: tuple[float, ...] = (),
    ) -> None:
        """
        Runs on a background daemon thread. Dequeues complete audio clips
        (assembled entirely by the listener) and performs all CPU-heavy work:
        threshold-based knock counting, feature extraction, and the user callback.

        The worker never calls stream.read(). Its only input is the queue of
        finished captures, making it trivially thread-safe with respect to
        audio capture. Knock counting starts at the trigger chunk, while feature
        extraction retains the full pre-roll waveform.
        """
        while not self._stop_event.is_set():
            try:
                capture = self._capture_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                count_waveform = capture.waveform[capture.trigger_sample :]
                tap_count = self.processor.count_taps(count_waveform, self.threshold)
                features = self.processor.extract_features(capture.waveform)
                augment_capture = (
                    augmented_trigger_callback is not None
                    and should_augment is not None
                    and should_augment()
                    and bool(gain_db_values)
                )
                if augment_capture:
                    assert augmented_trigger_callback is not None
                    augmented_features = tuple(
                        self.processor.extract_features(
                            apply_gain_db(capture.waveform, gain_db)
                        )
                        for gain_db in gain_db_values
                    )
                    augmented_trigger_callback(features, tap_count, augmented_features)
                else:
                    trigger_callback(features, tap_count)
            except Exception as error:
                self.last_error = error
                if error_callback is not None:
                    error_callback(error)
                else:
                    print(f"[!] Audio processing worker stopped: {error}")
                self.stop()
                return
            finally:
                self._capture_queue.task_done()
