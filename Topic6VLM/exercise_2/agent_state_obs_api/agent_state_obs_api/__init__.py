"""In-process observation API helpers."""

from __future__ import annotations

from .config_loader import default_config_path, load_config, resolve_agent_config

__all__ = [
    "ObservationAgent",
    "create_agent",
    "default_config_path",
    "load_config",
    "resolve_agent_config",
]


def __getattr__(name: str):
    if name == "create_agent":
        from .factory import create_agent

        return create_agent
    if name == "create_text_fallback_agent":
        from .factory import create_text_fallback_agent

        return create_text_fallback_agent
    if name == "ObservationAgent":
        from .observer import ObservationAgent

        return ObservationAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
