"""Platform-neutral connector contracts and action event types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ActionEvent:
    """An accepted switch event safe to forward outside the classifier."""

    event_id: str
    occurred_at: datetime
    label: str
    tap_count: int
    confidence: float | None = None
    source: str = "universal_switch"

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if self.tap_count < 0:
            raise ValueError("tap_count must be non-negative")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")

    @classmethod
    def create(
        cls,
        label: str,
        tap_count: int,
        confidence: float | None = None,
    ) -> ActionEvent:
        """Create an event with a fresh identifier and current UTC timestamp."""
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            label=label,
            tap_count=tap_count,
            confidence=confidence,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the stable JSON-compatible connector payload."""
        occurred_at = self.occurred_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return {
            "event_id": self.event_id,
            "source": self.source,
            "occurred_at": occurred_at,
            "label": self.label,
            "tap_count": self.tap_count,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ConnectorHealth:
    """Sanitized result of checking one connector's availability."""

    available: bool
    message: str


@dataclass(frozen=True)
class DeliveryResult:
    """Sanitized outcome from delivering one action event."""

    connector_id: str
    event_id: str
    success: bool
    message: str
    completed_at: datetime


class ConnectorError(RuntimeError):
    """A safe-to-display connector failure."""


class ConnectorAuthenticationError(ConnectorError):
    """A connector credential was rejected and startup should fail."""


class ActionConnector(ABC):
    """Base class implemented by each external automation platform."""

    @property
    @abstractmethod
    def connector_id(self) -> str:
        """Return the unique configuration identifier for this connector."""

    @abstractmethod
    def health_check(self) -> ConnectorHealth:
        """Check availability, raising for rejected credentials."""

    @abstractmethod
    def send(self, event: ActionEvent) -> None:
        """Deliver one event or raise a sanitized ConnectorError."""

    def close(self) -> None:
        """Release connector resources; stateless connectors need no action."""
        return None
