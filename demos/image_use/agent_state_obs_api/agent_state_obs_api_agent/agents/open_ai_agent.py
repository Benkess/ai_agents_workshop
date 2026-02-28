"""OpenAI-backed observation agent."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .base_obs_agent import BaseObsAgent
from ..tools.tool_obs_result import obs_result_tool


class AgentState(TypedDict):
    """Single-request graph state."""

    messages: list


class OpenAIObsAgent(BaseObsAgent):
    """Observation agent that forces structured tool output from an OpenAI model."""

    system_prompt = (
        "You are a precise visual observation assistant for a robotics system. "
        "You must always use the provided tool to record your observation. "
        "Never respond with plain text."
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        llm_kwargs = {"model": model}
        if api_key is not None:
            llm_kwargs["api_key"] = api_key
        if base_url is not None:
            llm_kwargs["base_url"] = base_url

        llm = ChatOpenAI(**llm_kwargs)
        self.llm_with_tools = llm.bind_tools([obs_result_tool], tool_choice="required")
        self.graph = self._create_graph()

    def _create_graph(self):
        def call_llm(state: AgentState) -> dict:
            response = self.llm_with_tools.invoke(state["messages"])
            return {"messages": state["messages"] + [response]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("call_llm", call_llm)
        graph_builder.add_edge(START, "call_llm")
        graph_builder.add_edge("call_llm", END)
        return graph_builder.compile()

    def query(self, image_b64: str, mime_type: str, prompt: str) -> dict:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content_blocks=[
                    {"type": "text", "text": prompt},
                    {"type": "image", "base64": image_b64, "mime_type": mime_type},
                ]
            ),
        ]

        result = self.graph.invoke({"messages": messages})
        final_message = result["messages"][-1]
        if not isinstance(final_message, AIMessage):
            raise RuntimeError("Expected final response to be an AIMessage")

        if not final_message.tool_calls:
            raise RuntimeError("No tool call found in model response")

        tool_call = final_message.tool_calls[0]
        args = tool_call.get("args", {})
        return {
            "value": args["value"],
            "failure_mode": args["failure_mode"],
            "reason": args["reason"],
        }
