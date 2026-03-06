"""Flask app exposing the observation agent over HTTP."""

from __future__ import annotations

import threading
from typing import Any

from flask import Flask, jsonify, request

try:
    from agent_state_obs_api import create_agent
except ModuleNotFoundError:  # pragma: no cover - fallback for repo-root imports
    from Topic6VLM.exercise_2.agent_state_obs_api.agent_state_obs_api import create_agent

from .config_loader import load_config, resolve_agent_config

def create_app(
    config_path: str | None = None,
    agent_overrides: dict[str, Any] | None = None,
    server_overrides: dict[str, Any] | None = None,
) -> Flask:
    app = Flask(__name__)
    config = load_config(config_path)
    effective_agent_config = resolve_agent_config(config["agent"], agent_overrides)
    effective_server_config = dict(config["server"])
    if server_overrides:
        for key, value in server_overrides.items():
            if value is not None:
                effective_server_config[key] = value

    app.config["SERVER_CONFIG"] = effective_server_config
    app.config["AGENT_CONFIG"] = effective_agent_config
    agent_lock = threading.Lock()
    agent = create_agent(effective_agent_config)

    @app.route("/observe", methods=["POST"])
    def observe():
        if not agent_lock.acquire(blocking=False):
            return jsonify({"busy": True}), 503

        try:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return jsonify({"success": False, "error": "Missing required field: request JSON body"}), 400

            missing_fields = [
                field for field in ("image_b64", "mime_type", "prompt") if field not in data
            ]
            if missing_fields:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Missing required field: {missing_fields[0]}",
                        }
                    ),
                    400,
                )

            result = agent.query(
                image_b64=str(data["image_b64"]),
                mime_type=str(data["mime_type"]),
                prompt=str(data["prompt"]),
            )
            response: dict[str, Any] = {"success": True, **result}
            return jsonify(response)
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500
        finally:
            agent_lock.release()

    return app


app = create_app()
