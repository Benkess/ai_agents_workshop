"""Flask app exposing the observation agent over HTTP."""

from __future__ import annotations

import threading
from typing import Any

from flask import Flask, jsonify, request

from agent_state_obs_api_agent.agents import OpenAIObsAgent

from .config_loader import load_config


def create_agent(agent_config: dict[str, Any]):
    implementation = str(agent_config["implementation"])
    model = str(agent_config["model"])
    if implementation == "openai":
        return OpenAIObsAgent(model=model)
    raise ValueError(f"Unsupported agent implementation: {implementation}")

def create_app(config_path: str | None = None) -> Flask:
    app = Flask(__name__)
    config = load_config(config_path)
    agent_lock = threading.Lock()
    agent = create_agent(config["agent"])

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
