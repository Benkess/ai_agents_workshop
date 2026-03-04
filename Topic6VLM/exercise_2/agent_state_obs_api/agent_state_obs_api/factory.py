"""Factory for observation agent implementations."""

from __future__ import annotations

from typing import Any

try:
    from agent_state_obs_api_agent.agents import OpenAIObsAgent
except ModuleNotFoundError:  # pragma: no cover - fallback for repo-root imports
    from Topic6VLM.exercise_2.agent_state_obs_api.agent_state_obs_api_agent.agents import OpenAIObsAgent


def create_agent(agent_config: dict[str, Any]):
    implementation = str(agent_config["implementation"])
    if implementation == "openai":
        return OpenAIObsAgent(
            model=str(agent_config["model"]),
            api_key=agent_config.get("api_key"),
            base_url=agent_config.get("base_url"),
        )
    raise ValueError(f"Unsupported agent implementation: {implementation}")
