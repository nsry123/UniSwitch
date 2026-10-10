"""Home Assistant connector using its authenticated REST event API."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from src.connectors.base import (
    ActionConnector,
    ActionEvent,
    ConnectorAuthenticationError,
    ConnectorError,
    ConnectorHealth,
)


class HomeAssistantConnector(ActionConnector):
    """Publish accepted action events to a Home Assistant event bus."""

    def __init__(
        self,
        connector_id: str,
        base_url: str,
        token: str,
        event_type: str = "universal_switch_event",
        timeout_seconds: float = 5.0,
    ) -> None:
        self._connector_id = connector_id
        self.base_url = self._validate_base_url(base_url)
        if not token:
            raise ValueError("Home Assistant token must not be empty")
        self._token = token
        self.event_type = event_type
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)

    @property
    def connector_id(self) -> str:
        return self._connector_id

    @staticmethod
    def _validate_base_url(value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Home Assistant base_url must be an http or https URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Home Assistant base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("Home Assistant base_url must not contain a query or fragment")
        return value.rstrip("/")

    def _request(self, method: str, path: str, payload: dict[str, object] | None = None) -> None:
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response.read()
        except HTTPError as error:
            if error.code in {401, 403}:
                raise ConnectorAuthenticationError(
                    f"connector {self.connector_id!r} credentials were rejected"
                ) from None
            raise ConnectorError(
                f"connector {self.connector_id!r} received HTTP {error.code}"
            ) from None
        except TimeoutError:
            raise ConnectorError(f"connector {self.connector_id!r} request timed out") from None
        except URLError:
            raise ConnectorError(f"connector {self.connector_id!r} is unreachable") from None
        except OSError:
            raise ConnectorError(f"connector {self.connector_id!r} is unreachable") from None

    def health_check(self) -> ConnectorHealth:
        try:
            self._request("GET", "/api/")
        except ConnectorAuthenticationError:
            raise
        except ConnectorError as error:
            return ConnectorHealth(False, str(error))
        return ConnectorHealth(True, "Home Assistant API is reachable")

    def send(self, event: ActionEvent) -> None:
        self._request("POST", f"/api/events/{self.event_type}", event.to_payload())
