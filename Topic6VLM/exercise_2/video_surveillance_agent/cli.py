"""CLI entrypoint for the video surveillance workflow."""

from __future__ import annotations

import argparse

from Topic6VLM.exercise_2.agent_state_obs_api.agent_state_obs_api import ObservationAgent

from .surveillance import run_surveillance


def main() -> None:
    parser = argparse.ArgumentParser(description="Video surveillance agent")
    parser.add_argument("--video-path", required=True, help="Path to the input video")
    parser.add_argument("--config", default=None, help="Path to the agent config JSON file")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="How often to sample video frames",
    )
    args = parser.parse_args()

    observer = ObservationAgent(config_path=args.config)
    events = run_surveillance(
        video_path=args.video_path,
        observer=observer,
        interval_seconds=args.interval_seconds,
    )

    print("Events:")
    if not events:
        print("[]")
        return

    for event in events:
        print(event)


if __name__ == "__main__":
    main()
