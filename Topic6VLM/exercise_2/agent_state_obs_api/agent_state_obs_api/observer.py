"""Direct-call wrapper around the observation agent."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from .config_loader import load_config, resolve_agent_config
from .factory import create_agent


class ObservationAgent:
    """Convenience wrapper for in-process image observation."""

    def __init__(self, config_path: str | None = None):
        config = load_config(config_path)
        self.agent_config = resolve_agent_config(config["agent"])
        self.agent = create_agent(self.agent_config)

    def observe_bytes(self, image_bytes: bytes, mime_type: str, prompt: str) -> dict:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return self.agent.query(image_b64=image_b64, mime_type=mime_type, prompt=prompt)

    def observe_image_path(self, image_path: str, prompt: str) -> dict:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        resolved_mime_type = mime_type or "application/octet-stream"
        return self.observe_bytes(path.read_bytes(), resolved_mime_type, prompt)
