"""Factory for observation agent implementations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

def _import_openai_obs_agent():
    try:
        from agent_state_obs_api_agent.agents import OpenAIObsAgent
    except ModuleNotFoundError:  # pragma: no cover - fallback for local exercise execution
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from agent_state_obs_api_agent.agents import OpenAIObsAgent
    return OpenAIObsAgent


def create_agent(agent_config: dict[str, Any]):
    implementation = str(agent_config["implementation"])
    if implementation == "openai":
        openai_obs_agent = _import_openai_obs_agent()
        return openai_obs_agent(
            model=str(agent_config["model"]),
            api_key=agent_config.get("api_key"),
            base_url=agent_config.get("base_url"),
        )
    raise ValueError(f"Unsupported agent implementation: {implementation}")


def create_text_fallback_agent(agent_config: dict[str, Any]):
    implementation = str(agent_config["implementation"])
    if implementation == "openai":
        from .fallback_agent import OpenAITextObsAgent

        return OpenAITextObsAgent(
            model=str(agent_config["model"]),
            api_key=agent_config.get("api_key"),
            base_url=agent_config.get("base_url"),
        )
    raise ValueError(f"Unsupported agent implementation: {implementation}")
