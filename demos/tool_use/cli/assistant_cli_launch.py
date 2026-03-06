# Imports
import argparse

# Import Project Modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from assistant_cli import run_lang_graph_agent
from assistant_factory import AgentFactory

def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Assistant CLI with a specified agent and thread ID.")
    parser.add_argument(
        "--agent",
        type=str,
        default="openai",
        help=f"The type of assistant agent to use: {', '.join(AgentFactory.get_supported_agents())} (default: openai).",
    )
    parser.add_argument(
        "--thread_id",
        type=str,
        default=None,
        help="The thread ID for the chat session (default: generated based on timestamp).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name override for Qwen agents (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct').",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Select the agent based on the provided argument
    try:
        agent = AgentFactory.create_agent(agent_type=args.agent, model_name=args.model)
    except ValueError as e:
        print(e)
        print(f"Supported agents are: {', '.join(AgentFactory.get_supported_agents())}")
        return

    # Generate a unique thread ID if not provided
    thread_id = args.thread_id
    if thread_id is None:
        import time
        thread_id = f"chat-{int(time.time())}"

    run_lang_graph_agent(agent, thread_id=thread_id)

if __name__ == "__main__":
    main()