"""
Version using LangChain's MCP integration from the official docs.

Install:
    pip install langchain langchain-openai langchain-mcp-adapters python-dotenv

Environment:
    OPENAI_API_KEY=...
    ASTA_API_KEY=...

Notes:
- This uses `langchain-mcp-adapters` and `MultiServerMCPClient`, which lets
  LangChain load MCP tools directly from the server. :contentReference[oaicite:0]{index=0}
- The docs show using `client.get_tools()` and then passing those tools into
  an agent. `MultiServerMCPClient` is stateless by default. :contentReference[oaicite:1]{index=1}
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTA_API_KEY = os.environ["ASTA_API_KEY"]

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"

SYSTEM_PROMPT = (
    "You are a research assistant with access to Semantic Scholar tools. "
    "Use tools when appropriate to answer questions about research papers, "
    "authors, citations, and references."
)


async def build_agent():
    # Official LangChain MCP docs show passing HTTP transport config and headers
    # through MultiServerMCPClient, then calling get_tools(). :contentReference[oaicite:2]{index=2}
    client = MultiServerMCPClient(
        {
            "asta": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {
                    "x-api-key": ASTA_API_KEY,
                },
            }
        }
    )

    tools = await client.get_tools()

    # The docs show create_agent(model, tools) with MCP-loaded tools. :contentReference[oaicite:3]{index=3}
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


async def main():
    agent = await build_agent()

    print("Asta MCP Research Chatbot (LangChain MCP version)")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": user_input},
                ]
            }
        )

        messages = result.get("messages", [])

        # Print tool activity so you can observe decisions
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            if msg_type == "ai":
                tool_calls = getattr(msg, "tool_calls", None) or []
                for tc in tool_calls:
                    print(f"\nCalling tool: {tc['name']}")
                    print(f"Arguments: {tc['args']}")
            elif msg_type == "tool":
                tool_name = getattr(msg, "name", "unknown_tool")
                tool_content = getattr(msg, "content", "")
                print(f"\nTool result from {tool_name}:")
                print(tool_content)

        # Final assistant response
        final_text = None
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "ai" and getattr(msg, "content", None):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    final_text = content
                    break
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    joined = "\n".join(part for part in text_parts if part).strip()
                    if joined:
                        final_text = joined
                        break

        print(f"\nAssistant: {final_text or '[No final text returned]'}\n")


if __name__ == "__main__":
    asyncio.run(main())