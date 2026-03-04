"""Tests for the direct-call wrapper and surveillance utilities."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from Topic6VLM.exercise_2.agent_state_obs_api.agent_state_obs_api.config_loader import (
    load_config,
    resolve_agent_config,
)
from Topic6VLM.exercise_2.video_surveillance_agent.surveillance import run_surveillance
from Topic6VLM.exercise_2.video_surveillance_agent.video_utils import sample_video_frames


def test_load_config_uses_default() -> None:
    config = load_config()
    assert config["agent"]["model"] == "llava"


def test_resolve_agent_config_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("EXERCISE_2_TEST_KEY", "secret")
    resolved = resolve_agent_config(
        {
            "implementation": "openai",
            "model": "llava",
            "base_url": None,
            "api_key": None,
            "api_key_env": "EXERCISE_2_TEST_KEY",
        }
    )
    assert resolved["api_key"] == "secret"


def test_sample_video_frames_returns_expected_timestamps(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    width = 32
    height = 32
    fps = 2.0
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()

    for frame_index in range(8):
        value = frame_index * 20
        frame = np.full((height, width, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = sample_video_frames(str(video_path), interval_seconds=2.0)
    timestamps = [round(frame.timestamp_seconds, 2) for frame in frames]
    assert timestamps == [0.0, 2.0]


def test_run_surveillance_tracks_enter_and_exit(tmp_path: Path) -> None:
    class StubObserver:
        def __init__(self) -> None:
            self.responses = iter(
                [
                    {"value": "FALSE", "failure_mode": "CONFIDENT", "reason": "empty"},
                    {"value": "FALSE", "failure_mode": "CONFIDENT", "reason": "empty"},
                    {"value": "TRUE", "failure_mode": "CONFIDENT", "reason": "person"},
                    {"value": "TRUE", "failure_mode": "CONFIDENT", "reason": "person"},
                    {"value": "TRUE", "failure_mode": "UNCERTAIN", "reason": "blurry"},
                    {"value": "FALSE", "failure_mode": "CONFIDENT", "reason": "empty"},
                ]
            )

        def observe_bytes(self, image_bytes: bytes, mime_type: str, prompt: str) -> dict:
            return next(self.responses)

    video_path = tmp_path / "sample.mp4"
    width = 32
    height = 32
    fps = 1.0
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()

    for frame_index in range(6):
        frame = np.full((height, width, 3), frame_index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    events = run_surveillance(
        video_path=str(video_path),
        observer=StubObserver(),
        interval_seconds=1.0,
    )

    assert events == [
        {"timestamp_seconds": 2.0, "event": "enter"},
        {"timestamp_seconds": 5.0, "event": "exit"},
    ]
