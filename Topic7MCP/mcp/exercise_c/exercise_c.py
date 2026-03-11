
import json
import os
from typing import TypedDict, List, Dict, Any

import requests
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END


load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTA_API_KEY = os.environ["ASTA_API_KEY"]

MODEL = "gpt-4o-mini"
MCP_URL = "https://asta-tools.allen.ai/mcp/v1"

client = OpenAI(api_key=OPENAI_API_KEY)


SYSTEM_PROMPT = (
    "You are a research assistant with access to Semantic Scholar tools. "
    "Use tools when appropriate to answer questions about research papers, "
    "authors, citations, and references."
)


class State(TypedDict):
    user_message: str
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]


def mcp_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "x-api-key": ASTA_API_KEY,
    }


def parse_mcp_response(response: requests.Response) -> Dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        pass

    data_lines = []
    for line in response.text.strip().splitlines():
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    if not data_lines:
        raise RuntimeError(f"Asta MCP returned invalid JSON: {response.text}")

    try:
        return json.loads("\n".join(data_lines))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Asta MCP returned invalid JSON: {response.text}") from exc


def mcp_to_openai_tool(mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": mcp_tool["name"],
            "description": mcp_tool.get("description", ""),
            "parameters": mcp_tool["inputSchema"],
        },
    }


def get_asta_tools() -> List[Dict[str, Any]]:
    """Fetch tool schemas from MCP and convert to OpenAI format."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }

    resp = requests.post(MCP_URL, headers=mcp_headers(), json=payload, timeout=60)
    data = parse_mcp_response(resp)

    if not resp.ok:
        raise RuntimeError(
            f"Asta MCP tools/list failed with HTTP {resp.status_code}: "
            f"{data.get('error', resp.text.strip())}"
        )

    if "error" in data:
        raise RuntimeError(f"MCP tools/list error: {data['error']}")

    mcp_tools = data["result"]["tools"]
    return [mcp_to_openai_tool(tool) for tool in mcp_tools]


def call_asta_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tools/call and return the text content."""
    print(f"\nCalling tool: {name}")
    print(f"Arguments: {json.dumps(arguments, indent=2)}")

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments,
        },
    }

    try:
        resp = requests.post(MCP_URL, headers=mcp_headers(), json=payload, timeout=120)
        data = parse_mcp_response(resp)

        if not resp.ok:
            return (
                f"Tool error: Asta MCP request failed with HTTP {resp.status_code}: "
                f"{data.get('error', resp.text.strip())}"
            )

        if "error" in data:
            return f"Tool error: {data['error']}"

        content = data["result"].get("content", [])
        if not content:
            return "Tool returned no content."

        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(json.dumps(item))
            else:
                text_parts.append(str(item))

        result_text = "\n".join(part for part in text_parts if part).strip()
        return result_text or "Tool returned empty text."

    except Exception as e:
        return f"Tool error: {e}"


def chat(user_message: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One turn of the chatbot loop, handling tool calls."""
    messages = list(messages)
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }

        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]

        messages.append(assistant_message)

        if not message.tool_calls:
            return messages

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                tool_result = f"Tool error: invalid JSON arguments for {tool_name}: {e}"
            else:
                tool_result = call_asta_tool(tool_name, tool_args)

            print(f"\nTool result from {tool_name}:\n{tool_result}\n")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )


def chatbot_node(state: State) -> State:
    updated_messages = chat(
        user_message=state["user_message"],
        messages=state["messages"],
        tools=state["tools"],
    )
    return {
        "user_message": "",
        "messages": updated_messages,
        "tools": state["tools"],
    }


def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


def main():
    tools = get_asta_tools()
    graph = build_graph()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("Asta MCP Research Chatbot")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        state: State = {
            "user_message": user_input,
            "messages": messages,
            "tools": tools,
        }

        result = graph.invoke(state)
        messages = result["messages"]

        last_message = messages[-1]
        if last_message["role"] == "assistant":
            print(f"\nAssistant: {last_message.get('content', '')}\n")
        else:
            print("\nAssistant: [No final assistant message returned]\n")


if __name__ == "__main__":
    main()
