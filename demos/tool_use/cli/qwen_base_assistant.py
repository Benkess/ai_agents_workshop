# Import necessary libraries
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# Library Imports
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import core Assistant base class
from cli.assistant import Assistant

# Import tool implementations
from tools.assistant.tool_get_weather import get_weather
from tools.assistant.tool_calculator import calculator
from tools.assistant.tool_area import area
from tools.assistant.tool_count_letter import count_letter
from tools.assistant.tool_text_to_speech import text_to_speech


class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - user_input: The text entered by the user
    - should_exit: Boolean flag indicating if user wants to quit
    - verbose: Boolean flag to enable or disable verbose tracing
    - skip_input: Boolean flag to skip LLM call
    - messages: Chat history (list of AnyMessage)
    """

    user_input: str
    should_exit: bool
    verbose: bool
    skip_input: bool
    messages: Annotated[list[AnyMessage], add_messages]


class BaseQwenAssistant(Assistant):
    """
    Shared Qwen assistant implementation using an OpenAI-compatible local runtime.
    """

    system_prompt = "You are a helpful assistant with access to various tools."
    max_tool_calling_iterations = 5
    default_base_url = "http://localhost:8000/v1"
    default_model_name = ""
    display_name = "Qwen Assistant"

    def __init__(self, model_name: str = None, base_url: str = None):
        tools = [get_weather, calculator, area, count_letter, text_to_speech]
        self.tool_map = {tool.name: tool for tool in tools}
        self.model_name = model_name or self.default_model_name
        self.base_url = base_url or self.default_base_url

        llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key="EMPTY",
            temperature=0.7,
        )

        self.llm_with_tools = llm.bind_tools(tools)
        self.initial_state = self.get_initial_state()

    def get_initial_state(self) -> AgentState:
        return AgentState(
            user_input="",
            should_exit=False,
            verbose=False,
            skip_input=False,
            messages=[SystemMessage(content=self.system_prompt)],
        )

    def create_graph(self, checkpointer=None):
        return self.create_graph_from_llm(self.llm_with_tools, checkpointer=checkpointer)

    def create_graph_from_llm(self, llm_with_tools, checkpointer=None):
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
            if state.get("verbose", False):
                print(f"Calling {self.display_name} via OpenAI-compatible runtime...")

            messages = state["messages"]
            for _ in range(self.max_tool_calling_iterations):
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                if response.tool_calls:
                    if state.get("verbose", False):
                        print(f"{self.display_name} wants to call {len(response.tool_calls)} tool(s)")

                    for tool_call in response.tool_calls:
                        function_name = tool_call["name"]
                        function_args = tool_call["args"]

                        if state.get("verbose", False):
                            print(f"  Tool: {function_name}")
                            print(f"  Args: {function_args}")

                        if function_name in self.tool_map:
                            try:
                                result = self.tool_map[function_name].invoke(function_args)
                            except Exception as exc:
                                result = f"Error running tool {function_name}: {exc}"
                        else:
                            result = f"Error: Unknown function {function_name}"

                        if state.get("verbose", False):
                            print(f"  Result: {result}")

                        messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=tool_call["id"],
                            )
                        )
                else:
                    break

            return {"messages": messages}

        def print_response(state: AgentState) -> dict:
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    print(f"Assistant: {msg.content}\n")
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
