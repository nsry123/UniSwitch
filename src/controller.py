"""Thread-safe orchestration between audio processing and classification."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from src.ml.classifier import PredictionResult, TapClassifier


class EventStatus(StrEnum):
    """Action-level outcome for a processed capture window."""

    EVENT = "EVENT"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


@dataclass(frozen=True)
class EventDecision:
    """Detailed temporal decision emitted by the controller."""

    status: EventStatus
    label: str | None
    classification: PredictionResult | None
    tap_count: int


class SwitchController:
    """Manage captured-window registration and normal prediction modes."""

    def __init__(
        self,
        classifier: TapClassifier,
        state_path: str | Path | None = None,
        state_metadata: Mapping[str, Any] | None = None,
        confirmation_count: int = 1,
        event_cooldown: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if confirmation_count <= 0:
            raise ValueError("confirmation_count must be positive")
        if event_cooldown < 0:
            raise ValueError("event_cooldown must be non-negative")
        self.classifier = classifier
        self.mode = "execute"
        self.target_label: str | None = None
        self.samples_collected = 0
        self.samples_needed = 3
        self.state_path = Path(state_path) if state_path is not None else None
        self.state_metadata = dict(state_metadata or {})
        self.confirmation_count = confirmation_count
        self.event_cooldown = float(event_cooldown)
        self._clock = clock
        self._pending_label: str | None = None
        self._pending_count = 0
        self._last_event_time: float | None = None
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

    def process_capture(self, features: np.ndarray, tap_count: int) -> EventDecision:
        """Route a capture through registration, classification, and hysteresis."""
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
                return EventDecision(EventStatus.NONE, None, None, tap_count)

            if self.mode != "execute":
                raise RuntimeError(f"unsupported controller mode: {self.mode}")
            classification = self.classifier.predict_detailed(features)
            if not classification.accepted:
                self._pending_label = None
                self._pending_count = 0
                return EventDecision(EventStatus.UNKNOWN, None, classification, tap_count)

            now = self._clock()
            if (
                self._last_event_time is not None
                and now - self._last_event_time < self.event_cooldown
            ):
                self._pending_label = None
                self._pending_count = 0
                return EventDecision(
                    EventStatus.NONE, classification.label, classification, tap_count
                )

            if classification.label == self._pending_label:
                self._pending_count += 1
            else:
                self._pending_label = classification.label
                self._pending_count = 1

            if self._pending_count < self.confirmation_count:
                return EventDecision(
                    EventStatus.NONE, classification.label, classification, tap_count
                )

            self._last_event_time = now
            self._pending_label = None
            self._pending_count = 0
            print(
                f"[!] Action Triggered: Detected {tap_count} tap(s) on {classification.label}"
            )
            return EventDecision(
                EventStatus.EVENT,
                classification.label,
                classification,
                tap_count,
            )

    def audio_callback(self, features: np.ndarray, tap_count: int) -> str | None:
        """Compatibility wrapper returning the raw class label or ``Unknown``."""
        was_registering = self.mode == "register"
        decision = self.process_capture(features, tap_count)
        if was_registering:
            return None
        if decision.classification is None:
            return "Unknown"
        return decision.classification.label
