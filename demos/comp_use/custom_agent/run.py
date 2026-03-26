# run.py
# CLI entry point for the computer use agent.
#
# Required args:
#   --env     Path to environment config JSON
#   --agent   Path to agent config JSON
#
# Optional overrides (take precedence over values in the config files):
#   --task        Override user_prompt without editing the agent config
#   --start-url   Override start_url without editing the environment config
#   --headless    Run the browser in headless mode
#   --verbose     Enable verbose agent output
#
# Examples:
#   python run.py --env config/environment/playwright_gpt.json --agent config/agent/gpt_agent.json
#
#   python run.py \
#       --env config/environment/playwright_gpt.json \
#       --agent config/agent/gpt_agent.json \
#       --task "Fill in the contact form and submit it" \
#       --start-url "file:///C:/Users/benpk/projects/form.html" \
#       --headless \
#       --verbose

import argparse
import json
import sys
import os

# Ensure custom_agent/ is importable when run.py is invoked from other directories
sys.path.insert(0, os.path.dirname(__file__))


def load_json(path: str) -> dict:
    """Load a JSON config file and return it as a dict."""
    with open(path, "r") as f:
        return json.load(f)


def build_env(env_config: dict):
    """Instantiate the correct ComputerUseEnv subclass from a config dict."""
    env_type = env_config.get("type")
    env_params = env_config.get("params", {})

    if env_type == "playwright":
        from playwright_env import PlaywrightComputerUseEnv
        return PlaywrightComputerUseEnv(**env_params)
    elif env_type == "pyautogui":
        from pyautogui_env import PyAutoGUIComputerUseEnv
        return PyAutoGUIComputerUseEnv(**env_params)
    else:
        raise ValueError(f"Unsupported environment type: '{env_type}'")


def build_agent(agent_config: dict, env):
    """Instantiate a ComputerUseAgent from a config dict and a live env."""
    from custom_comp_use_agent import ComputerUseAgent
    return ComputerUseAgent(computer_use_env=env, **agent_config)


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Launch the computer use agent with a given environment and agent config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Basic launch
  python run.py --env config/environment/playwright_gpt.json --agent config/agent/gpt_agent.json

  # Override task and starting URL inline
  python run.py \\
      --env config/environment/playwright_gpt.json \\
      --agent config/agent/gpt_agent.json \\
      --task "Click the login button" \\
      --start-url "http://localhost:3000"

  # Run headless with verbose logging
  python run.py \\
      --env config/environment/playwright_qwen.json \\
      --agent config/agent/qwen_agent.json \\
      --headless --verbose
        """,
    )

    # Required
    parser.add_argument(
        "--env",
        required=True,
        metavar="PATH",
        help="Path to the environment config JSON (e.g. config/environment/playwright_gpt.json)",
    )
    parser.add_argument(
        "--agent",
        required=True,
        metavar="PATH",
        help="Path to the agent config JSON (e.g. config/agent/gpt_agent.json)",
    )

    # Overrides
    parser.add_argument(
        "--task",
        default=None,
        metavar="TEXT",
        help="Override the user_prompt in the agent config. Wrap in quotes for multi-word tasks.",
    )
    parser.add_argument(
        "--start-url",
        default=None,
        metavar="URL",
        help=(
            "Override start_url in the environment config. "
            "Accepts http://, https://, or file:// paths. "
            "Only applies to Playwright environments."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run the browser in headless mode. Only applies to Playwright environments.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output from the agent.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    try:
        env_config = load_json(args.env)
    except FileNotFoundError:
        parser.error(f"Environment config not found: {args.env}")

    try:
        raw_agent_config = load_json(args.agent)
    except FileNotFoundError:
        parser.error(f"Agent config not found: {args.agent}")

    # The config file has the shape {"name": "...", "agent": {...}}.
    # ComputerUseAgent only accepts the inner "agent" dict.
    agent_config = raw_agent_config.get("agent", raw_agent_config)

    # "implementation" is a config-file convention (e.g. "openai"), not a
    # ComputerUseAgent parameter — drop it before unpacking.
    agent_config.pop("implementation", None)

    # ------------------------------------------------------------------
    # Apply CLI overrides (mutate in-memory, never touch the files)
    # ------------------------------------------------------------------
    if args.task:
        agent_config["user_prompt"] = args.task

    if args.start_url:
        if env_config.get("type") != "playwright":
            print("[Warning] --start-url has no effect on non-Playwright environments.")
        else:
            env_config.setdefault("params", {})["start_url"] = args.start_url

    if args.headless:
        if env_config.get("type") != "playwright":
            print("[Warning] --headless has no effect on non-Playwright environments.")
        else:
            env_config.setdefault("params", {})["headless"] = True

    if args.verbose:
        agent_config["verbose"] = True

    # ------------------------------------------------------------------
    # Print effective config so you always know what's running
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  COMPUTER USE AGENT")
    print("=" * 60)
    print(f"  Env config  : {args.env}")
    print(f"  Agent config: {args.agent}")
    print(f"  Task        : {agent_config.get('user_prompt', '(not set)')}")
    if env_config.get("type") == "playwright":
        params = env_config.get("params", {})
        print(f"  Start URL   : {params.get('start_url') or '(none)'}")
        print(f"  Headless    : {params.get('headless', False)}")
    print(f"  Verbose     : {agent_config.get('verbose', False)}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Build and run
    # ------------------------------------------------------------------
    env = build_env(env_config)
    env.start_env()

    agent = build_agent(agent_config, env)

    try:
        agent.run()
    finally:
        env.stop_env()


if __name__ == "__main__":
    main()
