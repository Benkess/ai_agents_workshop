"""
Qwen3-VL Transformers tool-use demo.

Recommended environment:
- Python 3.10+
- CUDA-capable NVIDIA GPU strongly recommended
- WSL2 Ubuntu + CUDA is often the smoothest setup on Windows

Install dependencies:
    pip install -U torch transformers accelerate safetensors
    pip install -U bitsandbytes

Environment variables:
- QWEN_MODEL: defaults to Qwen/Qwen3-VL-4B-Instruct
- QWEN_DEVICE: defaults to cuda if available else cpu
- QWEN_DTYPE: defaults to bfloat16 or float16 on GPU, float32 on CPU

Run:
    python demos/tool_use/cli/qwen3_transformers_assistant.py
"""

import json
import os
import sys
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _qwen_transformers_adapter import QwenTransformersAdapter
from cli.assistant import Assistant
from tools.assistant.tool_area import area
from tools.assistant.tool_calculator import calculator
from tools.assistant.tool_count_letter import count_letter
from tools.assistant.tool_get_weather import get_weather
from tools.assistant.tool_text_to_speech import text_to_speech


class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    verbose: bool
    skip_input: bool
    messages: Annotated[list[AnyMessage], add_messages]


class Qwen3TransformersAssistant(Assistant):
    system_prompt = "You are a helpful assistant with access to various tools."
    max_tool_calling_iterations = 5
    tool_output_char_limit = 4000
    default_model_name = "Qwen/Qwen3-VL-4B-Instruct"

    def __init__(self, model_name: str | None = None):
        tools = [get_weather, calculator, area, count_letter, text_to_speech]
        self.tool_map = {tool.name: tool for tool in tools}
        self.model_name = model_name or self.default_model_name
        self.adapter = QwenTransformersAdapter(tools=tools, model_name=self.model_name)
        self.initial_state = self.get_initial_state()

    def get_initial_state(self) -> AgentState:
        return AgentState(
            user_input="",
            should_exit=False,
            verbose=False,
            skip_input=False,
            messages=[SystemMessage(content=self.system_prompt)],
        )

    def _truncate_output(self, value: Any) -> str:
        text = str(value)
        if len(text) <= self.tool_output_char_limit:
            return text
        return text[: self.tool_output_char_limit - 20] + "\n...[truncated]"

    def _normalize_tool_args(self, raw_args: Any) -> tuple[dict[str, Any] | None, str | None]:
        if isinstance(raw_args, dict):
            return raw_args, None
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                return None, "Tool arguments were not valid JSON."
            if isinstance(parsed, dict):
                return parsed, None
            return None, "Tool arguments must decode to a JSON object."
        return None, "Tool arguments must be a dict or JSON object string."

    def create_graph(self, checkpointer=None):
        def get_user_input(state: AgentState) -> dict:
            print("\n" + "=" * 50)
            print("Enter your text (or 'quit' to exit):")
            print("=" * 50)
            print("\n> ", end="")
            user_input = input()

            lc = user_input.strip().lower()
            if lc in ["quit", "exit", "q"]:
                return {
                    "user_input": user_input,
                    "should_exit": True,
                    "skip_input": False,
                }

            if lc == "":
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                }

            if lc == "verbose":
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                    "verbose": True,
                }

            if lc == "quiet":
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                    "verbose": False,
                }

            return {
                "user_input": user_input,
                "should_exit": False,
                "skip_input": False,
                "messages": [HumanMessage(content=user_input)],
            }

        def call_llm(state: AgentState) -> dict:
            messages = state["messages"]
            verbose = state.get("verbose", False)

            for _ in range(self.max_tool_calling_iterations):
                if verbose:
                    print("Calling Qwen3-VL via local Transformers adapter...")
                response = self.adapter.invoke(messages)
                messages.append(response)

                if verbose and response.content:
                    print(f"Raw assistant content: {self._truncate_output(response.content)}")

                if not response.tool_calls:
                    break

                if verbose:
                    print(f"Qwen3-VL requested {len(response.tool_calls)} tool(s)")

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    raw_args = tool_call.get("args", {})
                    normalized_args, arg_error = self._normalize_tool_args(raw_args)

                    if verbose:
                        print(f"  Tool: {tool_name}")
                        print(f"  Args: {raw_args}")

                    if tool_name not in self.tool_map:
                        result = f"Error: Unknown function {tool_name}"
                    elif arg_error:
                        result = f"Error: {arg_error}"
                    else:
                        try:
                            result = self.tool_map[tool_name].invoke(normalized_args)
                        except Exception as exc:
                            result = f"Error running tool {tool_name}: {exc}"

                    result = self._truncate_output(result)
                    if verbose:
                        print(f"  Result: {result}")

                    messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        )
                    )

            return {"messages": messages}

        def print_response(state: AgentState) -> dict:
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    if msg.content:
                        print(f"Assistant: {msg.content}\n")
                    else:
                        print("Assistant: [No final response generated]\n")
                    break
            return {}

        def route_after_input(state: AgentState) -> str:
            if state["should_exit"]:
                return END
            if state["skip_input"]:
                return "get_user_input"
            return "call_llm"

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("get_user_input", get_user_input)
        graph_builder.add_node("call_llm", call_llm)
        graph_builder.add_node("print_response", print_response)

        graph_builder.add_edge(START, "get_user_input")
        graph_builder.add_conditional_edges(
            "get_user_input",
            route_after_input,
            {
                "call_llm": "call_llm",
                "get_user_input": "get_user_input",
                END: END,
            },
        )
        graph_builder.add_edge("call_llm", "print_response")
        graph_builder.add_edge("print_response", "get_user_input")

        return graph_builder.compile(checkpointer=checkpointer)


def main():
    print("=" * 50)
    print("Qwen3-VL Transformers Assistant")
    print("=" * 50)
    print()
    print("Expected local environment:")
    print("  pip install -U torch transformers accelerate safetensors")
    print("  python demos/tool_use/cli/qwen3_transformers_assistant.py")
    print()

    thread_id = "qwen3-transformers-chat-1"
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string("qwen3_transformers_checkpoints.db") as checkpointer:
        agent = Qwen3TransformersAssistant()

        print("Creating LangGraph...")
        graph = agent.create_graph(checkpointer=checkpointer)
        print("Graph created successfully!")

        state = graph.get_state(config)
        if state.next:
            print("\nResuming from checkpoint...")
            graph.invoke(None, config=config)
        else:
            print("\nStarting new chat...")
            graph.invoke(agent.initial_state, config=config)


if __name__ == "__main__":
    main()
