"""Video loading and frame sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class SampledFrame:
    """A sampled frame and its timestamp."""

    timestamp_seconds: float
    image_bytes: bytes
    mime_type: str = "image/jpeg"


def sample_video_frames(video_path: str, interval_seconds: float = 2.0) -> list[SampledFrame]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError(f"Could not determine FPS for video: {path}")

    frame_interval = max(1, int(round(fps * interval_seconds)))
    sampled_frames: list[SampledFrame] = []
    frame_index = 0

    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_interval == 0:
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    raise ValueError(f"Failed to encode frame {frame_index} from video: {path}")

                timestamp_seconds = frame_index / fps
                sampled_frames.append(
                    SampledFrame(
                        timestamp_seconds=timestamp_seconds,
                        image_bytes=buffer.tobytes(),
                    )
                )
            frame_index += 1
    finally:
        capture.release()

    return sampled_frames
