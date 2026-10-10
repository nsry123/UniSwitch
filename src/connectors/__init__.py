"""Extensible action delivery for smart-home and automation platforms."""

from src.connectors.base import (
    ActionConnector,
    ActionEvent,
    ConnectorAuthenticationError,
    ConnectorError,
    ConnectorHealth,
    DeliveryResult,
)
from src.connectors.config import DEFAULT_CONNECTOR_CONFIG_PATH, load_connectors
from src.connectors.dispatcher import ConnectorDispatcher

__all__ = [
    "ActionConnector",
    "ActionEvent",
    "ConnectorAuthenticationError",
    "ConnectorDispatcher",
    "ConnectorError",
    "ConnectorHealth",
    "DEFAULT_CONNECTOR_CONFIG_PATH",
    "DeliveryResult",
    "load_connectors",
]
