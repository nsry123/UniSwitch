"""Local browser dashboard for inspecting live microphone input."""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from src.controller import EventDecision

WEB_ROOT = Path(__file__).parent
TEMPLATE_PATH = WEB_ROOT / "templates" / "monitor.html"
STATIC_ROOT = WEB_ROOT / "static"


class _MonitorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], monitor: MicrophoneMonitor) -> None:
        super().__init__(address, _MonitorRequestHandler)
        self.monitor = monitor


class _MonitorRequestHandler(BaseHTTPRequestHandler):
    server: _MonitorHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        routes = {
            "/": (TEMPLATE_PATH, "text/html; charset=utf-8"),
            "/monitor.css": (STATIC_ROOT / "monitor.css", "text/css; charset=utf-8"),
            "/monitor.js": (
                STATIC_ROOT / "monitor.js",
                "text/javascript; charset=utf-8",
            ),
        }
        if self.path == "/api/audio":
            self._send_json(self.server.monitor.snapshot())
            return
        route = routes.get(self.path)
        if route is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        file_path, content_type = route
        self._send_bytes(file_path.read_bytes(), content_type)

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self._send_bytes(body, "application/json; charset=utf-8")

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        """Keep rapid chart polling out of the command-line log."""


class MicrophoneMonitor:
    """Publish audio frames to a localhost-only monitoring page."""

    def __init__(self, threshold: float, sample_rate: int, port: int = 8765) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 <= port <= 65535:
            raise ValueError("monitor port must be between 0 and 65535")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.port = port
        self._lock = threading.Lock()
        self._sequence = 0
        self._energy = 0.0
        self._peak = 0.0
        self._capturing = False
        self._waveform: list[float] = []
        self._updated_at = 0.0
        self._result_sequence = 0
        self._last_result: dict[str, Any] | None = None
        self._server: _MonitorHTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Return the active dashboard URL."""
        return f"http://127.0.0.1:{self.port}"

    def publish(self, audio_frame: np.ndarray, energy: float, capturing: bool) -> None:
        """Store a downsampled frame without blocking microphone capture."""
        waveform = np.asarray(audio_frame, dtype=np.float32)
        stride = max(1, waveform.size // 192)
        display_samples = waveform[::stride][:192]
        with self._lock:
            self._sequence += 1
            self._energy = float(energy)
            self._peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
            self._capturing = capturing
            self._waveform = display_samples.astype(float).tolist()
            self._updated_at = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        """Return the most recent microphone state as a JSON-ready mapping."""
        with self._lock:
            return {
                "sequence": self._sequence,
                "energy": self._energy,
                "peak": self._peak,
                "threshold": self.threshold,
                "capturing": self._capturing,
                "waveform": self._waveform,
                "sampleRate": self.sample_rate,
                "active": self._updated_at > 0 and time.monotonic() - self._updated_at < 1.0,
                "resultSequence": self._result_sequence,
                "lastResult": self._last_result,
            }

    @staticmethod
    def _finite(value: float | None) -> float | None:
        return float(value) if value is not None and np.isfinite(value) else None

    def publish_decision(self, decision: EventDecision) -> None:
        """Publish the most recent classified or registration capture."""
        classification = decision.classification
        if classification is None:
            result: dict[str, Any] = {
                "status": decision.status.value,
                "label": decision.label or "Registration capture",
                "accepted": None,
                "reason": "Stored as a registration example; classification was not run.",
                "rejectionReason": None,
                "tapCount": decision.tap_count,
                "score": None,
                "acceptanceRadius": None,
                "margin": None,
                "requiredMargin": None,
                "confidence": None,
                "candidate": None,
                "probabilities": {},
            }
        else:
            probabilities = dict(classification.probabilities)
            candidate = (
                max(probabilities, key=lambda label: probabilities[label])
                if probabilities
                else None
            )
            score = self._finite(classification.score)
            radius = self._finite(classification.acceptance_radius)
            margin = self._finite(classification.margin)
            required_margin = self._finite(classification.required_margin)
            confidence = max(probabilities.values()) if probabilities else None
            if classification.accepted:
                reason = (
                    f"Score {score:.3f} is within the class radius {radius:.3f}."
                    if margin is None
                    else f"Score {score:.3f} is within radius {radius:.3f}; "
                    f"margin {margin:.3f} clears the required {required_margin:.3f}."
                )
                if decision.status.value == "NONE":
                    reason += " No action was emitted because confirmation or cooldown is active."
            elif classification.rejection_reason == "distance":
                reason = (
                    f"Closest score {score:.3f} exceeds the class radius {radius:.3f}."
                )
            elif classification.rejection_reason == "margin":
                reason = (
                    f"The closest classes are ambiguous: margin {margin:.3f} is below "
                    f"the required {required_margin:.3f}."
                )
            else:
                reason = "No class has the three examples required for classification."
            result = {
                "status": decision.status.value,
                "label": classification.label,
                "accepted": classification.accepted,
                "reason": reason,
                "rejectionReason": classification.rejection_reason,
                "tapCount": decision.tap_count,
                "score": score,
                "acceptanceRadius": radius,
                "margin": margin,
                "requiredMargin": required_margin,
                "confidence": confidence,
                "candidate": candidate,
                "probabilities": probabilities,
            }
        with self._lock:
            self._result_sequence += 1
            self._last_result = result

    def start(self, open_browser: bool = True) -> None:
        """Start the local server and optionally open its page in a browser."""
        if self._server is not None:
            return
        self._server = _MonitorHTTPServer(("127.0.0.1", self.port), self)
        self.port = self._server.server_port
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="microphone-monitor",
            daemon=True,
        )
        self._server_thread.start()
        print(f"[*] Microphone monitor available at {self.url}")
        if open_browser:
            webbrowser.open(self.url)

    def stop(self) -> None:
        """Stop the local dashboard server."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
        self._server = None
        self._server_thread = None
