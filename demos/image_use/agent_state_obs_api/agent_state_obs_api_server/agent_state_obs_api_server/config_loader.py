"""Helpers for loading server configuration from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "server.json"


def load_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Server config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid server config: root must be an object")

    server_cfg = config.get("server")
    agent_cfg = config.get("agent")
    if not isinstance(server_cfg, dict):
        raise ValueError("Invalid server config: missing 'server' object")
    if not isinstance(agent_cfg, dict):
        raise ValueError("Invalid server config: missing 'agent' object")

    if "host" not in server_cfg or "port" not in server_cfg:
        raise ValueError("Invalid server config: server.host and server.port are required")
    if "implementation" not in agent_cfg:
        raise ValueError("Invalid server config: agent.implementation is required")
    if "model" not in agent_cfg:
        raise ValueError("Invalid server config: agent.model is required")

    return config
