"""CLI test script for the observation API client."""

from __future__ import annotations

import argparse
from pprint import pprint

from agent_state_obs_api_client import observe


def main() -> None:
    parser = argparse.ArgumentParser(description="Send an image to the observation API")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--prompt",
        default="Describe what you observe in this image.",
        help="Prompt to send to the observation agent",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:5000",
        help="Observation API host",
    )
    args = parser.parse_args()

    pprint(observe(args.image_path, args.prompt, host=args.host))


if __name__ == "__main__":
    main()
