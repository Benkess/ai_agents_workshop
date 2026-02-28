"""Command-line entrypoint for the observation API server."""

from __future__ import annotations

import argparse

from .app import create_app
from .config_loader import default_config_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent state observation API server")
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Path to the server config JSON file",
    )
    parser.add_argument("--host", default=None, help="Host interface to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to bind")
    args = parser.parse_args()

    config = load_config(args.config)
    host = args.host or str(config["server"]["host"])
    port = args.port or int(config["server"]["port"])

    app = create_app(args.config)
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
