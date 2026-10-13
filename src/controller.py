"""Thread-safe orchestration between capture, calibration, and classification."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from src.ml.classifier import PredictionResult, TapClassifier
from src.ml.evaluation import calibrate_grouped_classifier

MIN_CALIBRATION_CAPTURES = 4
MIN_UNKNOWN_CAPTURES = 10
CALIBRATION_QUALITY_GATE = 0.8


class EventStatus(StrEnum):
    EVENT = "EVENT"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class CapturePurpose(StrEnum):
    INFERENCE = "inference"
    REGISTRATION = "registration"
    UNKNOWN_CALIBRATION = "unknown_calibration"


@dataclass(frozen=True)
class EventDecision:
    status: EventStatus
    label: str | None
    classification: PredictionResult | None
    tap_count: int
    purpose: CapturePurpose = CapturePurpose.INFERENCE
    message: str | None = None


class SwitchController:
    """Manage registration, unknown collection, calibration, and prediction."""

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
        report = classifier.calibration_report
        initial_state = str(report.get("state", "needs_data")) if report else "needs_data"
        self._calibration: dict[str, Any] = {
            "state": initial_state,
            "message": "",
            "classCaptureCounts": {},
            "unknownCaptureCount": 0,
            "evaluatedCandidates": 0,
            "totalCandidates": 0,
            "metrics": None,
            "parameters": None,
        }
        if report:
            self._calibration.update(report)
        self._refresh_calibration_guidance(preserve_state=report is not None)

    @property
    def taps_collected(self) -> int:
        return self.samples_collected

    @property
    def taps_needed(self) -> int:
        return self.samples_needed

    def is_registering(self) -> bool:
        with self._lock:
            return self.mode == "register"

    def _readiness_message(self) -> tuple[bool, str]:
        counts = self.classifier.class_capture_counts()
        if not counts:
            return False, "Register at least one class before calibration."
        short = {
            label: MIN_CALIBRATION_CAPTURES - count
            for label, count in counts.items()
            if count < MIN_CALIBRATION_CAPTURES
        }
        if short:
            details = ", ".join(f"{label}: {needed} more" for label, needed in short.items())
            return False, f"Collect more physical class captures ({details})."
        unknown_needed = MIN_UNKNOWN_CAPTURES - len(self.classifier.negative_data)
        if unknown_needed > 0:
            return False, f"Collect {unknown_needed} more unknown capture(s)."
        return True, "Ready. Type 'calibrate' in the terminal."

    def _refresh_calibration_guidance(self, *, preserve_state: bool = False) -> None:
        ready, message = self._readiness_message()
        self._calibration["classCaptureCounts"] = self.classifier.class_capture_counts()
        self._calibration["unknownCaptureCount"] = len(self.classifier.negative_data)
        if not preserve_state:
            self._calibration["state"] = "ready" if ready else "needs_data"
            self._calibration["message"] = message
        elif not self._calibration.get("message"):
            self._calibration["message"] = message

    def calibration_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._calibration,
                "classCaptureCounts": dict(self._calibration["classCaptureCounts"]),
                "metrics": dict(self._calibration["metrics"])
                if self._calibration.get("metrics")
                else None,
                "parameters": dict(self._calibration["parameters"])
                if self._calibration.get("parameters")
                else None,
            }

    def _mark_stale(self) -> None:
        prior = self._calibration.get("state")
        self._refresh_calibration_guidance()
        if prior in {"passed", "failed", "stale"}:
            self._calibration["state"] = "stale"
            self._calibration["message"] = (
                "Calibration data changed. Type 'calibrate' after collection is complete."
            )

    def _persist(self) -> None:
        if self.state_path is None:
            return
        if self.classifier.X_data or self.classifier.negative_data:
            self.classifier.calibration_report = self.calibration_snapshot()
            self.classifier.save(self.state_path, self.state_metadata)
        else:
            self.state_path.unlink(missing_ok=True)

    def set_registration_mode(self, label: str, taps: int = 10) -> None:
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

    def set_unknown_collection_mode(self, count: int = MIN_UNKNOWN_CAPTURES) -> None:
        if count <= 0:
            raise ValueError("unknown capture count must be positive")
        with self._lock:
            self.mode = "collect_unknown"
            self.target_label = None
            self.samples_collected = 0
            self.samples_needed = count
        print(f"\n[*] UNKNOWN COLLECTION: Make {count} ordinary non-command sound(s).")

    def cancel_collection(self) -> bool:
        with self._lock:
            if self.mode == "execute":
                return False
            self.mode = "execute"
            self.target_label = None
            self.samples_collected = 0
            self.samples_needed = 0
        print("[*] Collection cancelled. Returning to execute mode.")
        return True

    def set_execution_mode(self) -> None:
        with self._lock:
            self.mode = "execute"
            self.target_label = None
        print("\n[*] EXECUTION MODE: Actively listening")

    def clear_unknowns(self) -> int:
        with self._lock:
            removed = self.classifier.clear_unknowns()
            if self.mode == "collect_unknown":
                self.mode = "execute"
                self.samples_collected = 0
            if removed:
                self._mark_stale()
                self._persist()
            print(f"[*] Removed {removed} unknown calibration capture(s).")
            return removed

    def delete_class(self, label: str) -> int:
        with self._lock:
            deleted = self.classifier.delete_class(label)
            if self._pending_label == label.strip():
                self._pending_label = None
                self._pending_count = 0
            if self.mode == "register" and self.target_label == label.strip():
                self.mode = "execute"
                self.target_label = None
                self.samples_collected = 0
            self._mark_stale()
            self._persist()
            print(f"[*] Deleted class '{label.strip()}' ({deleted} exemplar(s)).")
            return deleted

    def calibrate(self, status_callback: Callable[[dict[str, Any]], None] | None = None) -> bool:
        """Calibrate a snapshot and atomically apply it after quality checks."""
        with self._lock:
            ready, message = self._readiness_message()
            if not ready:
                self._refresh_calibration_guidance()
                if status_callback is not None:
                    status_callback(self.calibration_snapshot())
                raise ValueError(message)
            source = self.classifier
            revision = source.data_revision
            features = np.vstack(source.X_data)
            labels = np.asarray(source.y_labels, dtype=np.str_)
            capture_ids = np.asarray(source.capture_ids, dtype=np.int64)
            augmented = np.asarray(source.augmented_flags, dtype=np.bool_)
            negatives = np.vstack(source.negative_data)
            negative_ids = list(source.negative_capture_ids)
            next_capture_id = source.next_capture_id
            distance_mode = source.distance_mode
            threshold_mode = source.threshold_mode
            metadata = dict(source.metadata)
            self._calibration.update(
                {
                    "state": "running",
                    "message": "Evaluating grouped calibration candidates...",
                    "evaluatedCandidates": 0,
                    "totalCandidates": 0,
                    "metrics": None,
                    "parameters": None,
                }
            )
        if status_callback is not None:
            status_callback(self.calibration_snapshot())

        def publish_progress(done: int, total: int) -> None:
            with self._lock:
                self._calibration["evaluatedCandidates"] = done
                self._calibration["totalCandidates"] = total
            if status_callback is not None and (done == total or done % 25 == 0):
                status_callback(self.calibration_snapshot())

        try:
            candidate, result = calibrate_grouped_classifier(
                features,
                labels,
                capture_ids,
                augmented,
                negatives,
                distance_mode=distance_mode,
                threshold_mode=threshold_mode,
                progress_callback=publish_progress,
            )
        except Exception as error:
            with self._lock:
                self._calibration.update(
                    {
                        "state": "failed",
                        "message": f"Calibration failed: {type(error).__name__}: {error}",
                    }
                )
                self._persist()
            if status_callback is not None:
                status_callback(self.calibration_snapshot())
            raise
        values = result.to_dict()
        metrics = {
            "knownAccuracy": result.known_end_to_end_accuracy,
            "unknownRejection": result.unknown_rejection_rate,
            "objective": result.objective,
            "probabilityNll": result.known_probability_nll,
            "foldCount": result.fold_count,
        }
        parameters = {
            key: values[key]
            for key in (
                "n_neighbors",
                "distance_threshold",
                "prototype_weight",
                "spread_multiplier",
                "margin_threshold",
                "probability_temperature",
            )
        }
        passed = (
            result.known_end_to_end_accuracy >= CALIBRATION_QUALITY_GATE
            and result.unknown_rejection_rate >= CALIBRATION_QUALITY_GATE
        )
        with self._lock:
            if self.classifier.data_revision != revision:
                self._calibration.update(
                    {
                        "state": "stale",
                        "message": "Capture data changed during calibration; run it again.",
                        "metrics": metrics,
                        "parameters": parameters,
                    }
                )
                if status_callback is not None:
                    status_callback(self.calibration_snapshot())
                return False
            report = {
                "status": "passed" if passed else "failed",
                "state": "passed" if passed else "failed",
                "message": "Calibration applied successfully."
                if passed
                else (
                    "Calibration did not meet the 80% known/unknown quality gate; "
                    "the existing classifier parameters were retained."
                ),
                "classCaptureCounts": source.class_capture_counts(),
                "unknownCaptureCount": len(source.negative_data),
                "evaluatedCandidates": result.evaluated_candidates,
                "totalCandidates": result.evaluated_candidates,
                "metrics": metrics,
                "parameters": parameters,
            }
            if passed:
                candidate.negative_data = [row.copy() for row in negatives]
                candidate.negative_capture_ids = negative_ids
                candidate.next_capture_id = next_capture_id
                candidate.data_revision = revision
                candidate.metadata = metadata
                candidate.calibration_report = report
                self.classifier = candidate
            else:
                self.classifier.calibration_report = report
            self._calibration = dict(report)
            self._persist()
        if status_callback is not None:
            status_callback(self.calibration_snapshot())
        print(f"[*] Held-out known accuracy: {result.known_end_to_end_accuracy:.1%}")
        print(f"[*] Unknown rejection: {result.unknown_rejection_rate:.1%}")
        print(f"[{'+' if passed else '!'}] {report['message']}")
        return passed

    def process_capture(
        self, features: np.ndarray, tap_count: int, augmented_features: Iterable[np.ndarray] = ()
    ) -> EventDecision:
        if tap_count < 0:
            raise ValueError("tap_count must be non-negative")
        with self._lock:
            if self.mode == "register":
                if self.target_label is None:
                    raise RuntimeError("registration mode has no target label")
                label = self.target_label
                capture_id = self.classifier.allocate_capture_id()
                self.classifier.register_tap(features, label, capture_id=capture_id)
                for vector in augmented_features:
                    self.classifier.register_tap(
                        vector, label, announce=False, capture_id=capture_id, augmented=True
                    )
                self.samples_collected += 1
                message = (
                    f"Captured example {self.samples_collected}/{self.samples_needed} "
                    f"for '{label}' ({tap_count} knock(s))."
                )
                print(f"[*] {message}")
                self._mark_stale()
                self._persist()
                if self.samples_collected >= self.samples_needed:
                    print(f"[+] Successfully learned '{label}'. Returning to execute mode.")
                    self.mode = "execute"
                    self.target_label = None
                return EventDecision(
                    EventStatus.NONE, label, None, tap_count, CapturePurpose.REGISTRATION, message
                )
            if self.mode == "collect_unknown":
                capture_id = self.classifier.allocate_capture_id()
                self.classifier.register_unknown(features, capture_id=capture_id)
                self.samples_collected += 1
                message = f"Stored unknown capture {self.samples_collected}/{self.samples_needed}."
                print(f"[*] {message}")
                self._mark_stale()
                self._persist()
                if self.samples_collected >= self.samples_needed:
                    print("[+] Unknown collection complete. Returning to execute mode.")
                    self.mode = "execute"
                return EventDecision(
                    EventStatus.NONE,
                    "Unknown calibration",
                    None,
                    tap_count,
                    CapturePurpose.UNKNOWN_CALIBRATION,
                    message,
                )
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
            print(f"[!] Action Triggered: Detected {tap_count} tap(s) on {classification.label}")
            return EventDecision(EventStatus.EVENT, classification.label, classification, tap_count)

    def audio_callback(self, features: np.ndarray, tap_count: int) -> str | None:
        was_collecting = self.mode in {"register", "collect_unknown"}
        decision = self.process_capture(features, tap_count)
        if was_collecting:
            return None
        if decision.classification is None:
            return "Unknown"
        return decision.classification.label
