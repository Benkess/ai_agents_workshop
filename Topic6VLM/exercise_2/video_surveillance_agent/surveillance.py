"""Core video surveillance logic."""

from __future__ import annotations

from typing import Any, Protocol

from .video_utils import sample_video_frames

PERSON_DETECTION_PROMPT = (
    "Look only for whether a human person is visible anywhere in this image. "
    "Return TRUE only if a person is clearly present. Return FALSE if no person is visible."
)

_PRESENT_VALUES = {"TRUE", "YES", "PRESENT", "PERSON", "HUMAN"}
_ABSENT_VALUES = {"FALSE", "NO", "ABSENT", "NONE", "NO_PERSON", "NO HUMAN", "NO PERSON"}
_CONFIDENT_FAILURE_MODES = {"CONFIDENT", "NONE", "OK", "SUCCESS"}


class SupportsObserveBytes(Protocol):
    """Minimal observer interface required by the surveillance loop."""

    def observe_bytes(self, image_bytes: bytes, mime_type: str, prompt: str) -> dict:
        ...


def _classify_presence(result: dict[str, Any]) -> str:
    failure_mode = str(result.get("failure_mode", "")).strip().upper()
    if failure_mode not in _CONFIDENT_FAILURE_MODES:
        return "unknown"

    value = str(result.get("value", "")).strip().upper()
    if value in _PRESENT_VALUES:
        return "present"
    if value in _ABSENT_VALUES:
        return "absent"
    return "unknown"


def run_surveillance(
    video_path: str,
    observer: SupportsObserveBytes,
    interval_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    has_human = False

    for frame in sample_video_frames(video_path, interval_seconds=interval_seconds):
        result = observer.observe_bytes(
            image_bytes=frame.image_bytes,
            mime_type=frame.mime_type,
            prompt=PERSON_DETECTION_PROMPT,
        )
        print({"timestamp_seconds": round(frame.timestamp_seconds, 3), **result})

        presence = _classify_presence(result)
        if presence == "present" and not has_human:
            has_human = True
            events.append({"timestamp_seconds": frame.timestamp_seconds, "event": "enter"})
        elif presence == "absent" and has_human:
            has_human = False
            events.append({"timestamp_seconds": frame.timestamp_seconds, "event": "exit"})

    return events
