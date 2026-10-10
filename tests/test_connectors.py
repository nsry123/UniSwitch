import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from main import action_event_from_decision, dispatch_decision
from src.connectors.base import (
    ActionConnector,
    ActionEvent,
    ConnectorAuthenticationError,
    ConnectorError,
    ConnectorHealth,
    DeliveryResult,
)
from src.connectors.config import load_connectors
from src.connectors.dispatcher import ConnectorDispatcher
from src.connectors.home_assistant import HomeAssistantConnector
from src.controller import EventDecision, EventStatus


def event(label: str = "desk", tap_count: int = 2) -> ActionEvent:
    return ActionEvent(
        event_id=f"event-{label}-{tap_count}",
        occurred_at=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
        label=label,
        tap_count=tap_count,
        confidence=0.94,
    )


class RecordingConnector(ActionConnector):
    def __init__(self, connector_id: str = "recording") -> None:
        self._connector_id = connector_id
        self.events: list[ActionEvent] = []
        self.closed = False

    @property
    def connector_id(self) -> str:
        return self._connector_id

    def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(True, "ready")

    def send(self, action_event: ActionEvent) -> None:
        self.events.append(action_event)

    def close(self) -> None:
        self.closed = True


class BlockingConnector(RecordingConnector):
    def __init__(self, connector_id: str = "blocking") -> None:
        super().__init__(connector_id)
        self.entered = threading.Event()
        self.release = threading.Event()

    def send(self, action_event: ActionEvent) -> None:
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        super().send(action_event)


class FlakyConnector(RecordingConnector):
    def send(self, action_event: ActionEvent) -> None:
        if not self.events:
            self.events.append(action_event)
            raise ConnectorError("temporary failure")
        self.events.append(action_event)


def test_action_event_payload_is_stable_and_utc() -> None:
    payload = event().to_payload()

    assert payload == {
        "event_id": "event-desk-2",
        "source": "universal_switch",
        "occurred_at": "2026-08-31T12:00:00Z",
        "label": "desk",
        "tap_count": 2,
        "confidence": 0.94,
    }


def test_dispatcher_preserves_order_and_drains_on_close() -> None:
    connector = RecordingConnector()
    dispatcher = ConnectorDispatcher([connector])
    dispatcher.start()

    assert dispatcher.submit(event("desk", 1))
    assert dispatcher.submit(event("wall", 2))
    dispatcher.close()

    assert [(item.label, item.tap_count) for item in connector.events] == [
        ("desk", 1),
        ("wall", 2),
    ]
    assert connector.closed


def test_dispatcher_isolates_connectors_and_drops_when_queue_is_full() -> None:
    blocking = BlockingConnector()
    fast = RecordingConnector("fast")
    results: list[DeliveryResult] = []
    dispatcher = ConnectorDispatcher(
        [blocking, fast], queue_capacity=1, result_callback=results.append
    )
    dispatcher.start()

    first = event("first", 1)
    assert dispatcher.submit(first)
    assert blocking.entered.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while not fast.events and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fast.events == [first]

    assert dispatcher.submit(event("second", 1))
    assert not dispatcher.submit(event("third", 1))
    assert any("queue is full" in result.message for result in results)

    blocking.release.set()
    dispatcher.close()
    assert [item.label for item in blocking.events] == ["first", "second"]


def test_dispatcher_contains_connector_errors_and_continues() -> None:
    connector = FlakyConnector()
    results: list[DeliveryResult] = []
    dispatcher = ConnectorDispatcher([connector], result_callback=results.append)
    dispatcher.start()

    dispatcher.submit(event("first", 1))
    dispatcher.submit(event("second", 1))
    dispatcher.close()

    assert [result.success for result in results] == [False, True]
    assert not dispatcher.submit(event("late", 1))


def test_only_event_decisions_are_dispatched() -> None:
    connector = RecordingConnector()
    dispatcher = ConnectorDispatcher([connector])
    dispatcher.start()
    decisions = [
        EventDecision(EventStatus.NONE, "desk", None, 1),
        EventDecision(EventStatus.UNKNOWN, None, None, 1),
        EventDecision(EventStatus.EVENT, "desk", None, 2),
    ]

    assert action_event_from_decision(decisions[0]) is None
    assert not dispatch_decision(decisions[0], dispatcher)
    assert not dispatch_decision(decisions[1], dispatcher)
    assert dispatch_decision(decisions[2], dispatcher)
    dispatcher.close()

    assert [(item.label, item.tap_count) for item in connector.events] == [("desk", 2)]


class _HomeAssistantHandler(BaseHTTPRequestHandler):
    server: "_HomeAssistantServer"

    def do_GET(self) -> None:  # noqa: N802
        self.server.requests.append(("GET", self.path, self.headers, None))
        self._respond()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.requests.append(("POST", self.path, self.headers, payload))
        self._respond()

    def _respond(self) -> None:
        status = self.server.status
        if self.headers.get("Authorization") != "Bearer test-token":
            status = 401
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, _format: str, *args: object) -> None:
        pass


class _HomeAssistantServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _HomeAssistantHandler)
        self.status = 200
        self.requests: list[tuple[str, str, object, object]] = []


@contextmanager
def home_assistant_server() -> Iterator[_HomeAssistantServer]:
    server = _HomeAssistantServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_home_assistant_health_and_event_delivery() -> None:
    with home_assistant_server() as server:
        connector = HomeAssistantConnector(
            "home",
            f"http://127.0.0.1:{server.server_port}",
            "test-token",
            event_type="universal_switch_custom",
        )

        assert connector.health_check().available
        connector.send(event())

    assert [(item[0], item[1]) for item in server.requests] == [
        ("GET", "/api/"),
        ("POST", "/api/events/universal_switch_custom"),
    ]
    assert server.requests[1][3] == event().to_payload()


def test_home_assistant_rejects_authentication_without_leaking_token() -> None:
    with home_assistant_server() as server:
        connector = HomeAssistantConnector(
            "home", f"http://127.0.0.1:{server.server_port}", "secret-rejected-token"
        )
        with pytest.raises(ConnectorAuthenticationError) as captured:
            connector.health_check()

    assert "secret-rejected-token" not in str(captured.value)


def test_home_assistant_reports_server_errors_and_recovers() -> None:
    with home_assistant_server() as server:
        connector = HomeAssistantConnector(
            "home", f"http://127.0.0.1:{server.server_port}", "test-token"
        )
        server.status = 503
        with pytest.raises(ConnectorError, match="HTTP 503"):
            connector.send(event())
        server.status = 200
        connector.send(event("wall", 1))


def test_home_assistant_sanitizes_timeouts(monkeypatch) -> None:
    connector = HomeAssistantConnector("home", "http://homeassistant.local:8123", "secret")

    def timeout(*_args, **_kwargs):
        raise TimeoutError

    monkeypatch.setattr("src.connectors.home_assistant.urlopen", timeout)

    with pytest.raises(ConnectorError, match="timed out") as captured:
        connector.send(event())
    assert "secret" not in str(captured.value)


def write_config(path: Path, connectors: object, version: int = 1) -> None:
    path.write_text(json.dumps({"version": version, "connectors": connectors}))


def valid_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "id": "home",
        "type": "home_assistant",
        "base_url": "http://homeassistant.local:8123",
    }
    config.update(overrides)
    return config


def test_connector_config_loads_defaults_and_overrides(tmp_path) -> None:
    path = tmp_path / "connectors.json"
    write_config(
        path,
        [valid_config(event_type="custom_event", timeout_seconds=2, token_env="HA_TOKEN")],
    )

    connectors = load_connectors(path, environ={"HA_TOKEN": "secret"})

    connector = connectors[0]
    assert isinstance(connector, HomeAssistantConnector)
    assert connector.connector_id == "home"
    assert connector.event_type == "custom_event"
    assert connector.timeout_seconds == 2.0


@pytest.mark.parametrize(
    ("connectors", "version", "message"),
    [
        ([valid_config()], 2, "version"),
        ([valid_config(id="duplicate"), valid_config(id="duplicate")], 1, "duplicate"),
        ([valid_config(type="unknown")], 1, "unsupported type"),
        ([valid_config(base_url="ftp://example.com")], 1, "http or https"),
        ([valid_config(event_type="Bad Event")], 1, "event_type"),
        ([valid_config(timeout_seconds=0)], 1, "timeout_seconds"),
        ([valid_config(token="embedded-secret")], 1, "unknown keys"),
    ],
)
def test_connector_config_rejects_invalid_values(
    tmp_path, connectors, version, message
) -> None:
    path = tmp_path / "connectors.json"
    write_config(path, connectors, version)

    with pytest.raises(ValueError, match=message):
        load_connectors(path, environ={"HOME_ASSISTANT_TOKEN": "secret"})


def test_connector_config_missing_and_malformed_files(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    assert load_connectors(missing, missing_ok=True) == []
    with pytest.raises(ValueError, match="does not exist"):
        load_connectors(missing)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    with pytest.raises(ValueError, match="JSONDecodeError"):
        load_connectors(malformed)


def test_connector_config_requires_token_without_exposing_value(tmp_path) -> None:
    path = tmp_path / "connectors.json"
    write_config(path, [valid_config()])

    with pytest.raises(ValueError, match="HOME_ASSISTANT_TOKEN") as captured:
        load_connectors(path, environ={})

    assert "secret-value" not in str(captured.value)
