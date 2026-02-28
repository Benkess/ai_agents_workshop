"""Command-line entrypoint for the observation API server."""

from __future__ import annotations

import argparse

from .app import create_app
from .config_loader import default_config_path, load_config, resolve_agent_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent state observation API server")
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Path to the server config JSON file",
    )
    parser.add_argument("--host", default=None, help="Host interface to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to bind")
    parser.add_argument("--model", default=None, help="Override the configured model")
    parser.add_argument("--base-url", default=None, help="Override the configured provider base URL")
    parser.add_argument("--api-key", default=None, help="Override the configured API key")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Override the configured API key environment variable name",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    server_config = dict(config["server"])
    host = args.host or str(server_config["host"])
    port = args.port or int(server_config["port"])
    agent_overrides = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "api_key_env": args.api_key_env,
    }
    server_overrides = {
        "host": args.host,
        "port": args.port,
    }
    effective_agent_config = resolve_agent_config(config["agent"], agent_overrides)
    effective_server_config = dict(server_config)
    for key, value in server_overrides.items():
        if value is not None:
            effective_server_config[key] = value

    print("Agent state observation server starting...")
    print(f"  Config file: {args.config}")
    print("  Server config:")
    print(f"    host: {effective_server_config['host']}")
    print(f"    port: {effective_server_config['port']}")
    print("  Agent config:")
    print(f"    implementation: {effective_agent_config['implementation']}")
    print(f"    model: {effective_agent_config['model']}")
    print(f"    base_url: {effective_agent_config['base_url']}")
    # print(f"    api_key: {effective_agent_config['api_key']}")

    app = create_app(
        config_path=args.config,
        agent_overrides=agent_overrides,
        server_overrides=server_overrides,
    )
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
