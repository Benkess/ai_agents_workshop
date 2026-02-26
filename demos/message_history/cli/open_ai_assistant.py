# open_ai_assistant.py

# Import necessary libraries
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver # pip install langgraph-checkpoint-sqlite
from langchain_core.messages.utils import trim_messages

# Library Imports
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__)))) # Adjust path for imports
# Import core Assistant base class
from assistant import Assistant

# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - user_input: The text entered by the user (set by get_user_input node)
    - should_exit: Boolean flag indicating if user wants to quit (set by get_user_input node)
    - verbose: Boolean flag to enable/disable verbose tracing (set by get_user_input node)
    - skip_input: Boolean flag to skip LLM call (set by get_user_input node)
    - messages: Chat history (list of AnyMessage)
    """
    user_input: str
    should_exit: bool
    verbose: bool
    skip_input: bool
    messages: Annotated[list[AnyMessage], add_messages]

# =============================================================================
# OPENAI ASSISTANT IMPLEMENTATION
# =============================================================================

class OpenAIAssistantTrimMessages(Assistant):
    """
    OpenAI Assistant using LangGraph with message trimming functionality to manage context window limits.
    """

    """ System prompt to guide the assistant's behavior. Can be customized as needed. """
    system_prompt = "You are a helpful assistant."

    """ Model """
    model = "gpt-4o-mini"

    """ Trim Messages Parameters: """
    trim_strategy = "last" # Strategy for trimming messages (e.g., 'last', 'first')
    token_counter = "approximate" # Method for counting tokens (e.g., "approximate"=count_tokens_approximately, ChatOpenAI(model="gpt-4o"))
    max_tokens = 128 # Maximum tokens for the trimmed messages (gpt-4o-mini has a context window of 128k tokens and an output limit of 16,384 tokens, but we set a lower limit for testing)
    start_on = "human" # Message type to start trimming from (e.g., "human", "ai")
    include_system = True # Whether to always include the system prompt in the messages
    allow_partial = False # Whether to allow partial messages when trimming (if False, messages that exceed max_tokens will be dropped entirely)
    # end_on = ("human", "tool") # Message types that can end the trimming (e.g., ("human", "ai", "tool"))

    def __init__(self):

        # ============================================
        # Create LLM
        # ============================================

        # Create LLM
        self.llm = ChatOpenAI(model=self.model)

        self.initial_state = self.get_initial_state()

    def get_initial_state(self) -> AgentState:
        """
        Define the initial state of the assistant agent.
        {
            "user_input": "",
            "should_exit": False,
            "verbose": False,
            "skip_input": False,
            "messages": [SystemMessage(content=self.system_prompt)],
        }
        """
        return AgentState(
            user_input="",
            should_exit=False,
            verbose=False,
            skip_input=False,
            messages=[SystemMessage(content=self.system_prompt)],
        )

    # =============================================================================
    # GRAPH CREATION
    # =============================================================================

    def create_graph(self, checkpointer=None):
        return self.create_graph_from_llm(self.llm, checkpointer=checkpointer)

    def create_graph_from_llm(self, llm, checkpointer=None):
        """
        Create the LangGraph state graph with nodes for conversation.
        1. get_user_input: Reads input from stdin
        2. call_llm: Calls the LLM
        3. print_response: Prints the final response

        Graph structure:
            START -> get_user_input -> [conditional] -> call_llm -> print_response -> get_user_input
                                |
                                +-> END (if user wants to quit)
        """

        # =========================================================================
        # NODE 1: get_user_input
        # =========================================================================
        def get_user_input(state: AgentState) -> dict:
            """
            Node that prompts the user for input via stdin.
            Updates state with user input and adds to messages.
            """
            print("\n" + "=" * 50)
            print("Enter your text (or 'quit' to exit):")
            print("=" * 50)
            print("\n> ", end="")
            user_input = input()

            lc = user_input.strip().lower()
            if lc in ['quit', 'exit', 'q']:
                return {
                    "user_input": user_input,
                    "should_exit": True,
                    "skip_input": False,
                }
            
            if lc == '':
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                }

            if lc == 'verbose':
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                    "verbose": True,
                }

            if lc == 'quiet':
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

        # =========================================================================
        # NODE: call_llm
        # =========================================================================
        def call_llm(state: AgentState) -> dict:
            if state.get("verbose", False):
                print("Calling LLM with messages...")
            messages = state["messages"]
            # Trim messages to avoid exceeding context window
            messages = trim_messages(  
                messages,
                strategy=self.trim_strategy,
                token_counter=self.token_counter,
                max_tokens=self.max_tokens,
                start_on=self.start_on,
                include_system=self.include_system,
                allow_partial=self.allow_partial,
            )
            if state["verbose"]:
                print(f"Trimmed messages (max_tokens={self.max_tokens}):")
                for msg in messages:
                    print(f"- {msg.type}: {msg.content}")
            response = llm.invoke(messages)
            return {"messages": [response]} # This will be appended to the existing messages in the state by the graph's message handling

        # =========================================================================
        # NODE: print_response
        # =========================================================================
        def print_response(state: AgentState) -> dict:
            # Find the last AIMessage
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    print(f"Assistant: {msg.content}\n")
                    break
            return {}

        # =========================================================================
        # ROUTING FUNCTION
        # =========================================================================
        def route_after_input(state: AgentState) -> str:
            if state["should_exit"]:
                return END
            elif state["skip_input"]:
                return "get_user_input"
            else:
                return "call_llm"

        # =========================================================================
        # GRAPH CONSTRUCTION
        # =========================================================================
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
                END: END
            }
        )
        graph_builder.add_edge("call_llm", "print_response")
        graph_builder.add_edge("print_response", "get_user_input")

        graph = graph_builder.compile(checkpointer=checkpointer)
        return graph

class OpenAIAssistant(Assistant):
    """
    OpenAI Assistant using LangGraph. No message history managment.
    """

    """ System prompt to guide the assistant's behavior. Can be customized as needed. """
    system_prompt = "You are a helpful assistant."

    def __init__(self):

        # ============================================
        # Create LLM
        # ============================================

        # Create LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        self.initial_state = self.get_initial_state()

    def get_initial_state(self) -> AgentState:
        """
        Define the initial state of the assistant agent.
        {
            "user_input": "",
            "should_exit": False,
            "verbose": False,
            "skip_input": False,
            "messages": [SystemMessage(content=self.system_prompt)],
        }
        """
        return AgentState(
            user_input="",
            should_exit=False,
            verbose=False,
            skip_input=False,
            messages=[SystemMessage(content=self.system_prompt)],
        )

    # =============================================================================
    # GRAPH CREATION
    # =============================================================================

    def create_graph(self, checkpointer=None):
        return self.create_graph_from_llm(self.llm, checkpointer=checkpointer)

    def create_graph_from_llm(self, llm, checkpointer=None):
        """
        Create the LangGraph state graph with nodes for conversation.
        1. get_user_input: Reads input from stdin
        2. call_llm: Calls the LLM
        3. print_response: Prints the final response

        Graph structure:
            START -> get_user_input -> [conditional] -> call_llm -> print_response -> get_user_input
                                |
                                +-> END (if user wants to quit)
        """

        # =========================================================================
        # NODE 1: get_user_input
        # =========================================================================
        def get_user_input(state: AgentState) -> dict:
            """
            Node that prompts the user for input via stdin.
            Updates state with user input and adds to messages.
            """
            print("\n" + "=" * 50)
            print("Enter your text (or 'quit' to exit):")
            print("=" * 50)
            print("\n> ", end="")
            user_input = input()

            lc = user_input.strip().lower()
            if lc in ['quit', 'exit', 'q']:
                return {
                    "user_input": user_input,
                    "should_exit": True,
                    "skip_input": False,
                }
            
            if lc == '':
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                }

            if lc == 'verbose':
                return {
                    "user_input": user_input,
                    "should_exit": False,
                    "skip_input": True,
                    "verbose": True,
                }

            if lc == 'quiet':
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

        # =========================================================================
        # NODE: call_llm
        # =========================================================================
        def call_llm(state: AgentState) -> dict:
            if state.get("verbose", False):
                print("Calling LLM with messages...")
            messages = state["messages"]
            if state["verbose"]:
                print(f"Full Message History:")
                for msg in messages:
                    print(f"- {msg.type}: {msg.content}")
            response = llm.invoke(messages)
            return {"messages": [response]} # This will be appended to the existing messages in the state by the graph's message handling


        # =========================================================================
        # NODE: print_response
        # =========================================================================
        def print_response(state: AgentState) -> dict:
            # Find the last AIMessage
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    print(f"Assistant: {msg.content}\n")
                    break
            return {}

        # =========================================================================
        # ROUTING FUNCTION
        # =========================================================================
        def route_after_input(state: AgentState) -> str:
            if state["should_exit"]:
                return END
            elif state["skip_input"]:
                return "get_user_input"
            else:
                return "call_llm"

        # =========================================================================
        # GRAPH CONSTRUCTION
        # =========================================================================
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
                END: END
            }
        )
        graph_builder.add_edge("call_llm", "print_response")
        graph_builder.add_edge("print_response", "get_user_input")

        graph = graph_builder.compile(checkpointer=checkpointer)
        return graph

def main():
    """
    Main function for the LangGraph agent.
    """
    print("=" * 50)
    print("LangGraph Agent")
    print("=" * 50)
    print()

    thread_id = "chat-1"
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        # Create OpenAI Assistant
        agent = OpenAIAssistantTrimMessages()

        print("Creating LangGraph...")
        graph = agent.create_graph(checkpointer=checkpointer)
        print("Graph created successfully!")

        print("Saving graph visualization...")
        def save_graph_image(graph, filename="lg_graph.png"):
            """
            Generate a Mermaid diagram of the graph and save it as a PNG image.
            Uses the graph's built-in Mermaid export functionality.
            """
            try:
                png_data = graph.get_graph(xray=True).draw_mermaid_png()
                with open(filename, "wb") as f:
                    f.write(png_data)
                print(f"Graph image saved to {filename}")
            except Exception as e:
                print(f"Could not save graph image: {e}")
                print("You may need to install additional dependencies: pip install grandalf")
        # save_graph_image(graph)

        state = graph.get_state(config)
        if state.next:
            print("\n🔄 Resuming from checkpoint...")
            graph.invoke(None, config=config)
        else:
            print("\n▶️ Starting new chat...")
            graph.invoke(agent.initial_state, config=config)


if __name__ == "__main__":
    main()