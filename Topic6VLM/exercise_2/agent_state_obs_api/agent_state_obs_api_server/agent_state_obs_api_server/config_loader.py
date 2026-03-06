"""Helpers for loading server configuration from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from agent_state_obs_api.config_loader import resolve_agent_config as resolve_local_agent_config
except ModuleNotFoundError:  # pragma: no cover - fallback for repo-root imports
    from Topic6VLM.exercise_2.agent_state_obs_api.agent_state_obs_api.config_loader import (
        resolve_agent_config as resolve_local_agent_config,
    )


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
    for optional_key in ("base_url", "api_key", "api_key_env"):
        if optional_key in agent_cfg and agent_cfg[optional_key] is not None and not isinstance(
            agent_cfg[optional_key], str
        ):
            raise ValueError(f"Invalid server config: agent.{optional_key} must be a string or null")

    return config


def resolve_agent_config(
    agent_cfg: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return resolve_local_agent_config(agent_cfg, overrides)
