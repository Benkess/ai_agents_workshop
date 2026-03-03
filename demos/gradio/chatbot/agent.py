from __future__ import annotations

import os
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    verbose: bool
    skip_input: bool
    messages: Annotated[list[AnyMessage], add_messages]


class TurnResult(TypedDict):
    state: AgentState
    debug_log: list[str]
    model_context: list[dict[str, str]]
    full_history: list[dict[str, str]]
    reply_text: str
    session_closed: bool
    skipped: bool


class GradioChatAgent:
    system_prompt = "You are a helpful assistant."
    trim_strategy = "last"
    token_counter = "approximate"
    max_tokens = 512
    start_on = "human"
    include_system = True
    allow_partial = False

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        system_prompt: str | None = None,
        trim_strategy: str | None = None,
        token_counter: str | Any | None = None,
        max_tokens: int | None = None,
        start_on: str | None = None,
        include_system: bool | None = None,
        allow_partial: bool | None = None,
    ) -> None:
        self.model = model
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if trim_strategy is not None:
            self.trim_strategy = trim_strategy
        if token_counter is not None:
            self.token_counter = token_counter
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if start_on is not None:
            self.start_on = start_on
        if include_system is not None:
            self.include_system = include_system
        if allow_partial is not None:
            self.allow_partial = allow_partial
        resolved_api_key = api_key
        if not resolved_api_key and api_key_env:
            resolved_api_key = os.getenv(api_key_env)
        llm_kwargs: dict[str, Any] = {"model": model}
        if resolved_api_key:
            llm_kwargs["api_key"] = resolved_api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)

    def get_initial_state(self) -> AgentState:
        return AgentState(
            user_input="",
            should_exit=False,
            verbose=False,
            skip_input=False,
            messages=[SystemMessage(content=self.system_prompt)],
        )

    def run_turn(self, state: AgentState, user_input: str) -> TurnResult:
        current_state = state or self.get_initial_state()
        debug_log: list[str] = []
        model_context: list[dict[str, str]] = []

        def log(message: str) -> None:
            debug_log.append(message)

        def get_user_input_node(_: AgentState) -> dict[str, Any]:
            normalized = (user_input or "").strip()
            lowered = normalized.lower()
            log(f"get_user_input: received text length={len(user_input or '')}")

            if lowered in {"quit", "exit", "q"}:
                log("get_user_input: quit command detected")
                return {
                    "user_input": user_input,
                    "should_exit": True,
                    "skip_input": False,
                }

            if normalized == "":
                log("get_user_input: empty input detected; skipping model call")
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                }

            if lowered == "verbose":
                log("get_user_input: enabling verbose mode")
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                    "verbose": True,
                }

            if lowered == "quiet":
                log("get_user_input: disabling verbose mode")
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                    "verbose": False,
                }

            log("get_user_input: appending human message to state")
            return {
                "user_input": user_input,
                "should_exit": False,
                "skip_input": False,
                "messages": [HumanMessage(content=user_input)],
            }

        def call_llm_node(node_state: AgentState) -> dict[str, Any]:
            log("call_llm: preparing trimmed message context")
            trimmed_messages = trim_messages(
                node_state["messages"],
                strategy=self.trim_strategy,
                token_counter=self.token_counter,
                max_tokens=self.max_tokens,
                start_on=self.start_on,
                include_system=self.include_system,
                allow_partial=self.allow_partial,
            )

            model_context.clear()
            model_context.extend(_messages_to_debug_records(trimmed_messages))
            log(
                "call_llm: invoking model with "
                f"{len(trimmed_messages)} messages after trimming"
            )

            response = self.llm.invoke(trimmed_messages)
            log("call_llm: model invocation completed")
            return {"messages": [response]}

        def route_after_input(node_state: AgentState) -> str:
            if node_state["should_exit"]:
                log("route_after_input: routing to END")
                return END
            if node_state["skip_input"]:
                log("route_after_input: skipping model call")
                return "finish_turn"
            log("route_after_input: routing to call_llm")
            return "call_llm"

        def finish_turn_node(node_state: AgentState) -> dict[str, Any]:
            log("finish_turn: completing graph turn")
            return {}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("get_user_input", get_user_input_node)
        graph_builder.add_node("call_llm", call_llm_node)
        graph_builder.add_node("finish_turn", finish_turn_node)
        graph_builder.add_edge(START, "get_user_input")
        graph_builder.add_conditional_edges(
            "get_user_input",
            route_after_input,
            {
                "call_llm": "call_llm",
                "finish_turn": "finish_turn",
                END: END,
            },
        )
        graph_builder.add_edge("call_llm", "finish_turn")
        graph_builder.add_edge("finish_turn", END)
        graph = graph_builder.compile()

        log("graph: invoking turn graph")
        next_state = graph.invoke(current_state)
        log("graph: turn graph finished")

        reply_text = _extract_last_ai_message(next_state["messages"])
        full_history = _messages_to_chat_history(next_state["messages"])
        skipped = next_state["skip_input"]
        session_closed = next_state["should_exit"]

        return TurnResult(
            state=next_state,
            debug_log=debug_log,
            model_context=model_context,
            full_history=full_history,
            reply_text=reply_text,
            session_closed=session_closed,
            skipped=skipped,
        )
def _extract_last_ai_message(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _stringify_content(message.content)
    return ""


def _messages_to_chat_history(messages: list[AnyMessage]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        role = "assistant" if isinstance(message, AIMessage) else "user"
        history.append({"role": role, "content": _stringify_content(message.content)})
    return history


def _messages_to_debug_records(messages: list[AnyMessage]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for message in messages:
        records.append(
            {
                "type": message.type,
                "content": _stringify_content(message.content),
            }
        )
    return records


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)
