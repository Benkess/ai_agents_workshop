"""Direct-call wrapper around the observation agent."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from .config_loader import load_config, resolve_agent_config
from .factory import create_agent, create_text_fallback_agent


class ObservationAgent:
    """Convenience wrapper for in-process image observation."""

    def __init__(self, config_path: str | None = None):
        config = load_config(config_path)
        self.agent_config = resolve_agent_config(config["agent"])
        self.agent = create_agent(self.agent_config)
        self._fallback_agent = None
        self._using_text_fallback = False

    @staticmethod
    def _is_tools_not_supported(exc: Exception) -> bool:
        message = str(exc).lower()
        return "does not support tools" in message or "tool" in message and "not support" in message

    def observe_bytes(self, image_bytes: bytes, mime_type: str, prompt: str) -> dict:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        if self._using_text_fallback:
            assert self._fallback_agent is not None
            return self._fallback_agent.query(image_b64=image_b64, mime_type=mime_type, prompt=prompt)

        try:
            return self.agent.query(image_b64=image_b64, mime_type=mime_type, prompt=prompt)
        except Exception as exc:
            if not self._is_tools_not_supported(exc):
                raise

            if self._fallback_agent is None:
                self._fallback_agent = create_text_fallback_agent(self.agent_config)
            self._using_text_fallback = True
            return self._fallback_agent.query(image_b64=image_b64, mime_type=mime_type, prompt=prompt)

    def observe_image_path(self, image_path: str, prompt: str) -> dict:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        resolved_mime_type = mime_type or "application/octet-stream"
        return self.observe_bytes(path.read_bytes(), resolved_mime_type, prompt)
