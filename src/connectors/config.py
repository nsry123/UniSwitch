"""Versioned connector configuration loading and validation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.connectors.base import ActionConnector
from src.connectors.home_assistant import HomeAssistantConnector

DEFAULT_CONNECTOR_CONFIG_PATH = Path("config/connectors.json")
_CONNECTOR_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_EVENT_TYPE = re.compile(r"^[a-z0-9_]+$")
_ROOT_KEYS = {"version", "connectors"}
_HOME_ASSISTANT_KEYS = {
    "id",
    "type",
    "base_url",
    "token_env",
    "event_type",
    "timeout_seconds",
}


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _required_string(config: Mapping[str, Any], key: str, connector_id: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"connector {connector_id!r} requires a non-empty {key}")
    return value.strip()


def _home_assistant_connector(
    config: Mapping[str, Any],
    connector_id: str,
    environ: Mapping[str, str],
) -> HomeAssistantConnector:
    unknown = set(config) - _HOME_ASSISTANT_KEYS
    if unknown:
        raise ValueError(f"connector {connector_id!r} has unknown keys: {sorted(unknown)}")
    base_url = _required_string(config, "base_url", connector_id)
    token_env = config.get("token_env", "HOME_ASSISTANT_TOKEN")
    if not isinstance(token_env, str) or not token_env.strip():
        raise ValueError(f"connector {connector_id!r} token_env must be a non-empty string")
    token = environ.get(token_env.strip(), "")
    if not token.strip():
        raise ValueError(
            f"connector {connector_id!r} requires environment variable {token_env.strip()!r}"
        )
    event_type = config.get("event_type", "universal_switch_event")
    if not isinstance(event_type, str) or not _EVENT_TYPE.fullmatch(event_type):
        raise ValueError(
            f"connector {connector_id!r} event_type must contain lowercase letters, "
            "numbers, and underscores only"
        )
    timeout = config.get("timeout_seconds", 5.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"connector {connector_id!r} timeout_seconds must be positive")
    return HomeAssistantConnector(
        connector_id=connector_id,
        base_url=base_url,
        token=token.strip(),
        event_type=event_type,
        timeout_seconds=float(timeout),
    )


def load_connectors(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
    missing_ok: bool = False,
) -> list[ActionConnector]:
    """Load configured connectors, optionally treating a missing file as disabled."""
    if not path.exists():
        if missing_ok:
            return []
        raise ValueError(f"connector configuration does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        message = f"cannot read connector configuration {path}: {type(error).__name__}"
        raise ValueError(message) from None
    root = _mapping(raw, "connector configuration")
    unknown_root = set(root) - _ROOT_KEYS
    if unknown_root:
        raise ValueError(f"connector configuration has unknown keys: {sorted(unknown_root)}")
    if root.get("version") != 1:
        raise ValueError("connector configuration version must be 1")
    entries = root.get("connectors")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("connector configuration connectors must be a JSON array")

    environment = os.environ if environ is None else environ
    connectors: list[ActionConnector] = []
    identifiers: set[str] = set()
    for index, entry in enumerate(entries):
        config = _mapping(entry, f"connector at index {index}")
        connector_id = _required_string(config, "id", f"index {index}")
        if not _CONNECTOR_ID.fullmatch(connector_id):
            raise ValueError(
                f"connector id {connector_id!r} may contain letters, numbers, '.', '-', and '_'"
            )
        if connector_id in identifiers:
            raise ValueError(f"duplicate connector id: {connector_id!r}")
        identifiers.add(connector_id)
        connector_type = _required_string(config, "type", connector_id)
        if connector_type != "home_assistant":
            raise ValueError(f"connector {connector_id!r} has unsupported type {connector_type!r}")
        connectors.append(_home_assistant_connector(config, connector_id, environment))
    return connectors
