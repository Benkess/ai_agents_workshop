"""Base interface for observation agents."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseObsAgent(ABC):
    """Abstract base class for image observation agents."""

    @abstractmethod
    def query(self, image_b64: str, mime_type: str, prompt: str) -> dict:
        """
        Send an image and prompt to the agent.
        Returns a dict with keys: value, failure_mode, reason.
        """
        raise NotImplementedError
