import pyaudio
import numpy as np
from collections import deque
import librosa
import threading
import queue


class AudioListener:
    def __init__(self, processor, device_index=None, threshold=0.015, pre_roll_sec=0.2, record_sec=1.5):
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
        self.processor    = processor
        self.threshold    = threshold
        self.chunk_size   = 1024
        self.device_index = device_index

        self.format   = pyaudio.paFloat32
        self.channels = 1
        self.rate     = self.processor.sample_rate

        self.pre_roll_chunks = int((pre_roll_sec * self.rate) / self.chunk_size)
        self.record_chunks   = int((record_sec   * self.rate) / self.chunk_size)

        # Ring buffer — filled continuously by the listener thread only.
        # Provides the pre-trigger context that captures a knock's attack transient.
        self.buffer = deque(maxlen=self.pre_roll_chunks)

        # Handoff point between the listener and worker threads.
        # Each item is a *complete*, contiguous np.ndarray ready for processing.
        # The listener assembles the entire clip (pre-roll + post-trigger) before
        # enqueueing, so the worker never needs to read from the stream.
        self._capture_queue = queue.Queue()

        # Signals both threads to exit cleanly.
        self._stop_event = threading.Event()

        self.audio = pyaudio.PyAudio()

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def listen(self, trigger_callback):
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
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )

        worker = threading.Thread(
            target=self._worker_loop,
            args=(trigger_callback,),
            daemon=True,
        )
        worker.start()

        print(f"[*] Mic active. Listening for taps (Threshold: {self.threshold})...")

        try:
            self._listener_loop(stream)
        except KeyboardInterrupt:
            print("\n[*] Stopping listener...")
        finally:
            self._stop_event.set()
            worker.join(timeout=3.0)
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

    def test_file(self, file_path: str):
        """
        Loads an audio file, simulates the energy gate by trimming silence,
        and extracts features just like the live microphone listener.
        """
        print(f"[*] Loading test file: {file_path}")
        try:
            y, sr    = librosa.load(file_path, sr=self.processor.sample_rate, mono=True)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            print(f"[*] Audio loaded. Original: {len(y)/sr:.2f}s  Trimmed: {len(y_trimmed)/sr:.2f}s")

            tap_count = self.processor.count_taps(y_trimmed)
            features  = self.processor.extract_features(y_trimmed)

            print(f"[*] Taps detected: {tap_count}")
            print(f"[*] Feature vector length: {len(features)}")
            return features, tap_count

        except Exception as e:
            print(f"[!] Error loading file {file_path}: {e}")
            return None, 0

    # ------------------------------------------------------------------ #
    #  Private — listener thread (sole owner of the stream)               #
    # ------------------------------------------------------------------ #

    def _listener_loop(self, stream):
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
        capturing         = False
        post_trigger      = []   # staging list for post-trigger chunks
        pre_roll_snapshot = []   # frozen copy of ring buffer at trigger moment
        remaining_chunks  = 0

        while not self._stop_event.is_set():
            data        = stream.read(self.chunk_size, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.float32)

            if not capturing:
                # IDLE: keep the ring buffer fresh.
                self.buffer.append(audio_frame)

                energy = self.processor.calculate_energy(audio_frame)

                if energy > self.threshold:
                    # Freeze the ring buffer. The trigger chunk is already
                    # included via the append above, so it appears exactly once
                    # at the end of pre_roll_snapshot — no double-count at the seam.
                    pre_roll_snapshot = list(self.buffer)
                    post_trigger      = []
                    remaining_chunks  = self.record_chunks
                    capturing         = True

            else:
                # CAPTURING: accumulate post-trigger chunks.
                # The ring buffer is intentionally not updated here; we do not
                # want decaying knock energy to pollute the next event's pre-roll.
                post_trigger.append(audio_frame)
                remaining_chunks -= 1

                if remaining_chunks <= 0:
                    # Assemble and ship the complete, contiguous clip.
                    full_clip = np.concatenate(pre_roll_snapshot + post_trigger)
                    self._capture_queue.put(full_clip)

                    # Seed the ring buffer with the tail of the just-captured
                    # window so the next event's pre-roll is meaningful.
                    tail = post_trigger[-self.pre_roll_chunks:]
                    self.buffer.clear()
                    self.buffer.extend(tail)

                    capturing = False

    # ------------------------------------------------------------------ #
    #  Private — worker thread                                             #
    # ------------------------------------------------------------------ #

    def _worker_loop(self, trigger_callback):
        """
        Runs on a background daemon thread. Dequeues complete audio clips
        (assembled entirely by the listener) and performs all CPU-heavy work:
        onset detection, feature extraction, and the user callback.

        The worker never calls stream.read(). Its only input is the queue of
        finished np.ndarrays, making it trivially thread-safe with respect to
        audio capture.
        """
        while not self._stop_event.is_set():
            try:
                full_clip = self._capture_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            tap_count = self.processor.count_taps(full_clip)
            features  = self.processor.extract_features(full_clip)

            trigger_callback(features, tap_count)