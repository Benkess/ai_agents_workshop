"""Fallback observation agent for providers that do not support tool-calling."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class OpenAITextObsAgent:
    """Observation agent that requests JSON text instead of tool-calls."""

    system_prompt = (
        "You are a precise visual observation assistant for robotics. "
        "Return only JSON with keys value, failure_mode, and reason. "
        "value must be TRUE or FALSE. "
        "failure_mode must be CONFIDENT or UNCERTAIN."
    )

    def __init__(
        self,
        model: str = "llava",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        llm_kwargs = {"model": model}
        if api_key is not None:
            llm_kwargs["api_key"] = api_key
        if base_url is not None:
            llm_kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)

    def _extract_text(self, response: AIMessage) -> str:
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    def _parse_result(self, text: str) -> dict:
        json_match = _JSON_PATTERN.search(text)
        candidate = json_match.group(0) if json_match else text
        try:
            data = json.loads(candidate)
        except Exception:
            normalized = text.upper()
            if "TRUE" in normalized or "YES" in normalized:
                return {
                    "value": "TRUE",
                    "failure_mode": "UNCERTAIN",
                    "reason": f"Unstructured response: {text}",
                }
            if "FALSE" in normalized or "NO" in normalized:
                return {
                    "value": "FALSE",
                    "failure_mode": "UNCERTAIN",
                    "reason": f"Unstructured response: {text}",
                }
            return {
                "value": "FALSE",
                "failure_mode": "UNCERTAIN",
                "reason": f"Could not parse response: {text}",
            }

        value = str(data.get("value", "")).strip().upper()
        if value not in {"TRUE", "FALSE"}:
            value = "FALSE"

        failure_mode = str(data.get("failure_mode", "")).strip().upper()
        if failure_mode not in {"CONFIDENT", "UNCERTAIN"}:
            failure_mode = "UNCERTAIN"

        reason = str(data.get("reason", ""))
        return {"value": value, "failure_mode": failure_mode, "reason": reason}

    def query(self, image_b64: str, mime_type: str, prompt: str) -> dict:
        response = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content_blocks=[
                        {"type": "text", "text": prompt},
                        {"type": "image", "base64": image_b64, "mime_type": mime_type},
                    ]
                ),
            ]
        )
        if not isinstance(response, AIMessage):
            raise RuntimeError("Expected model response to be an AIMessage")
        return self._parse_result(self._extract_text(response))
