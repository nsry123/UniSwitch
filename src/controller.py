"""Thread-safe orchestration between audio processing and classification."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.ml.classifier import TapClassifier


class SwitchController:
    """Manage captured-window registration and normal prediction modes."""

    def __init__(
        self,
        classifier: TapClassifier,
        state_path: str | Path | None = None,
        state_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.classifier = classifier
        self.mode = "execute"
        self.target_label: str | None = None
        self.samples_collected = 0
        self.samples_needed = 3
        self.state_path = Path(state_path) if state_path is not None else None
        self.state_metadata = dict(state_metadata or {})
        self._lock = threading.RLock()

    @property
    def taps_collected(self) -> int:
        """Backward-compatible name for the number of captured examples."""
        return self.samples_collected

    @property
    def taps_needed(self) -> int:
        """Backward-compatible name for the requested captured examples."""
        return self.samples_needed

    def set_registration_mode(self, label: str, taps: int = 10) -> None:
        """Begin collecting capture-window examples for ``label``.

        ``taps`` is retained for CLI compatibility, but counts callback windows,
        not the number of knocks detected inside each window.
        """
        normalized_label = label.strip()
        if not normalized_label:
            raise ValueError("label must not be empty")
        if taps <= 0:
            raise ValueError("registration sample count must be positive")
        with self._lock:
            self.mode = "register"
            self.target_label = normalized_label
            self.samples_collected = 0
            self.samples_needed = taps
        print(f"\n[*] REGISTRATION MODE: Capture {taps} example(s) for '{normalized_label}'.")

    def set_execution_mode(self) -> None:
        with self._lock:
            self.mode = "execute"
            self.target_label = None
        print("\n[*] EXECUTION MODE: Actively listening")

    def audio_callback(self, features: np.ndarray, tap_count: int) -> str | None:
        """Route one processed capture window to registration or inference."""
        if tap_count < 0:
            raise ValueError("tap_count must be non-negative")

        with self._lock:
            if self.mode == "register":
                if self.target_label is None:
                    raise RuntimeError("registration mode has no target label")
                label = self.target_label
                self.classifier.register_tap(features, label)
                self.samples_collected += 1
                print(
                    f"[*] Captured example {self.samples_collected}/{self.samples_needed} "
                    f"for '{label}' ({tap_count} knock(s))"
                )
                if self.state_path is not None:
                    self.classifier.save(self.state_path, self.state_metadata)
                if self.samples_collected >= self.samples_needed:
                    print(f"[+] Successfully learned '{label}'. Returning to execute mode.")
                    self.mode = "execute"
                    self.target_label = None
                return None

            if self.mode != "execute":
                raise RuntimeError(f"unsupported controller mode: {self.mode}")
            prediction = self.classifier.predict(features)
            if prediction != "Unknown":
                print(f"[!] Action Triggered: Detected {tap_count} tap(s) on {prediction}")
            return prediction
