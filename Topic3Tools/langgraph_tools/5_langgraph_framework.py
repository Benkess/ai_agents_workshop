# langgraph_tools_agent.py
# Program demonstrates use of LangGraph for an agent with tool calling.
# It maintains conversation context and supports checkpointing for recovery.
# Uses GPT-4o-mini with tools for weather, calculator, area, letter counting, and text-to-speech.

# Import necessary libraries
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import json
import ast
import math

# ============================================
# PART 1: Define Your Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def _safe_eval(expr: str):
    """Safely evaluate a simple arithmetic expression using AST traversal.

    Supports numeric literals, parentheses, binary ops + - * / // % ** and unary +/-.
    """
    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only int/float constants are allowed")
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.Pow):
                return left ** right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.FloorDiv):
                return left // right
            raise ValueError(f"Unsupported binary operator: {type(op)}")
        if isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        raise ValueError(f"Unsupported expression: {type(node)}")

    parsed = ast.parse(expr, mode="eval")
    return _eval_node(parsed)


@tool
def calculator(payload: str) -> str:
    """Evaluate an arithmetic expression.

    Expects `payload` to be a JSON string with key `expression`, e.g.
      {"expression": "2 + 3*(4-1)"}

    Returns a JSON string: {"success": true, "result": <number>} or
    {"success": false, "error": "message"}.

    Notes for agents: send a JSON string in `payload` exactly as above. The
    evaluator accepts whitespace in the expression and supports + - * / // % ** and unary +/-.
    """
    try:
        params = json.loads(payload)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Invalid JSON payload: {e}"})
    try:
        expr = params.get("expression")
        if expr is None:
            raise ValueError("Missing 'expression' field")
        # Do not alter whitespace; AST parsing allows it.
        value = _safe_eval(expr)
        return json.dumps({"success": True, "result": value})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def count_letter(payload: str) -> str:
    """Count occurrences of a letter (or substring) in a text.

    Expects `payload` to be a JSON string with keys:
      - `letter`: the character or substring to count
      - `text`: the text to search
      - `case_sensitive` (optional, default False)

    Returns a JSON string: {"success": true, "result": <count>} or
    {"success": false, "error": "message"}.
    """
    try:
        params = json.loads(payload)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Invalid JSON payload: {e}"})
    try:
        letter = params.get("letter")
        text = params.get("text")
        if letter is None or text is None:
            raise ValueError("Both 'letter' and 'text' fields are required")
        case_sensitive = bool(params.get("case_sensitive", False))
        if not case_sensitive:
            letter = letter.lower()
            text = text.lower()
        count = text.count(letter)
        return json.dumps({"success": True, "result": count})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def area(payload: str) -> str:
    """Compute area for simple shapes.

    Expects `payload` to be a JSON string describing the shape:
      Circle:    {"shape": "circle", "radius": 3}
      Rectangle: {"shape": "rectangle", "width": 3, "height": 4}
      Triangle:  {"shape": "triangle", "base": 3, "height": 4}

    Returns a JSON string: {"success": true, "result": <area>} or
    {"success": false, "error": "message"}.

    Notes for agents: send a JSON string in `payload` exactly as shown above.
    """
    try:
        params = json.loads(payload)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Invalid JSON payload: {e}"})
    try:
        shape = (params.get("shape") or "").lower()
        if shape == "circle":
            r = float(params.get("radius"))
            value = math.pi * r * r
            return json.dumps({"success": True, "result": value})
        if shape == "rectangle":
            w = float(params.get("width"))
            h = float(params.get("height"))
            value = w * h
            return json.dumps({"success": True, "result": value})
        if shape == "triangle":
            b = float(params.get("base"))
            h = float(params.get("height"))
            value = 0.5 * b * h
            return json.dumps({"success": True, "result": value})
        raise ValueError(f"Unsupported shape for area: {shape}")
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@tool
def text_to_speech(payload: str) -> str:
    """Convert text to speech and play it through speakers.
    
    Expects `payload` to be a JSON string with key `text`, e.g.
      {'payload': '{"text": "Hello, I am speaking!"}'}
    
    Returns a JSON string: {"success": true} or {"success": false, "error": "message"}.
    """
    try:
        params = json.loads(payload)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Invalid JSON payload: {e}"})
    
    try:
        import pyttsx3
        
        text = params.get("text")
        if not text:
            raise ValueError("Missing 'text' field")
        
        # Initialize text-to-speech engine
        engine = pyttsx3.init()
        
        # Optional: adjust speech rate (default is 200)
        engine.setProperty('rate', 150)
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# ============================================
# Tool Mapping
# ============================================

tools = [get_weather, calculator, area, count_letter, text_to_speech]
tool_map = {tool.name: tool for tool in tools}

# ============================================
# Create LLM with Tools
# ============================================

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

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

def create_graph(llm_with_tools, checkpointer=None):
    """
    Create the LangGraph state graph with nodes for conversation and tool calling.
    1. get_user_input: Reads input from stdin
    2. call_llm: Calls the LLM with tools, handles tool calls
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

        # Agent loop for tool calling
        messages = state["messages"]
        for iteration in range(5):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if response.tool_calls:
                if state.get("verbose", False):
                    print(f"LLM wants to call {len(response.tool_calls)} tool(s)")
                
                for tool_call in response.tool_calls:
                    function_name = tool_call["name"]
                    function_args = tool_call["args"]
                    
                    if state.get("verbose", False):
                        print(f"  Tool: {function_name}")
                        print(f"  Args: {function_args}")
                    
                    if function_name in tool_map:
                        result = tool_map[function_name].invoke(function_args)
                    else:
                        result = f"Error: Unknown function {function_name}"
                    
                    if state.get("verbose", False):
                        print(f"  Result: {result}")
                    
                    messages.append(ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    ))
            else:
                # No more tool calls, final answer
                break
        
        return {"messages": messages}

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
    def call_llama(state: AgentState) -> dict:
        if state.get("verbose", False):
            print(f"[TRACE] call_llama received user_input={state['user_input']!r}")

        chat = messages_to_chat_dicts(state["llama_messages"])
        prompt = llm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        if state.get("verbose", False):
            print(f"[TRACE] call_llama prompt=\n{prompt}")

        print("\nProcessing input with Llama...")
        try:
            resp = str(llm.invoke(prompt))
        except Exception as e:
            resp = f"<llama error: {e}>"
            raise e
        
        # If the model echoed the prompt, strip it.
        if resp.startswith(prompt):
            resp = resp[len(prompt):].lstrip()

        return {"llama_response": resp}


    # =========================================================================
    # NODE: call_qwen
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        if state.get("verbose", False):
            print(f"[TRACE] call_qwen received user_input={state['user_input']!r}")

        chat = messages_to_chat_dicts(state["qwen_messages"])
        prompt = qwen_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        if state.get("verbose", False):
            print(f"[TRACE] call_qwen prompt=\n{prompt}")

        print("\nProcessing input with Qwen...")
        try:
            resp = str(qwen_llm.invoke(prompt))
        except Exception as e:
            resp = f"<qwen error: {e}>"
            raise e
        
        # If the model echoed the prompt, strip it.
        if resp.startswith(prompt):
            resp = resp[len(prompt):].lstrip()

        return {"qwen_response": resp}

    # =========================================================================
    # NODE: print_both (join)
    # =========================================================================
    # This node prints both `llama_response` and `qwen_response`. It is
    # attached as a downstream node of both model nodes so it runs after
    # both branches complete and the state merges their outputs.
    def print_both(state: AgentState) -> dict:
        if state.get("verbose", False):
            print(f"[TRACE] print_both starting (len llama={len(str(state.get('llama_response','')))}, qwen={len(str(state.get('qwen_response','')))})")
        # Print whichever model produced a response. Only one model runs per input.
        has_llama = bool(state.get("llama_response"))
        has_qwen = bool(state.get("qwen_response"))

        if not has_llama and not has_qwen:
            if state.get("verbose", False):
                print("[TRACE] print_both found no responses; skipping print")
            return {}

        print("\n" + "=" * 50)
        print("Model Output:")
        print("=" * 50)

        if has_llama:
            print("\n-- Llama Response --\n")
            resp = state.get("llama_response", "<no response>")
            print(resp)

            # Modify the response to add the prefix
            modified_resp = ThreeWayChatPromptTemplate.modify_llama_response(resp)
            return {
                "llama_response": "", 
                "qwen_response": "",
                "llama_messages": [AIMessage(content=str(modified_resp))],
                "qwen_messages": [HumanMessage(content=str(modified_resp))]
            }
        elif has_qwen:
            print("\n-- Qwen Response --\n")
            resp = state.get("qwen_response", "<no response>")
            print(resp)

            # Modify the response to add the prefix
            modified_resp = ThreeWayChatPromptTemplate.modify_qwen_response(resp)
            return {
                "llama_response": "", 
                "qwen_response": "",
                "qwen_messages": [AIMessage(content=str(modified_resp))],
                "llama_messages": [HumanMessage(content=str(modified_resp))]
            }
        else:
            raise RuntimeError("print_both reached unexpected state with no responses")

    # (old single-LLM print node removed; use print_both as the join/printer)

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    # This function examines the state and determines which node to go to next.
    # It's used for conditional edges after get_user_input.
    # Two possible routes:
    #   1. User wants to quit -> END
    #   2. User entered any input -> proceed to call_llm
    def route_after_input(state: AgentState) -> str:
        """
        Routing function that determines the next node based on state.

        Logic:
          - If `should_exit` => END
          - If `skip_input` => loop back to `get_user_input`
          - If `user_input` begins with 'hey qwen' (case-insensitive) => call_qwen
          - Otherwise => call_llama
        """
        if state.get("should_exit", False):
            nxt = END
        elif state.get("skip_input", False):
            nxt = "get_user_input"
        else:
            ui = state.get("user_input", "") or ""
            if ui.strip().lower().startswith("hey qwen"):
                nxt = "call_qwen"
                # nxt = "call_llama" # TODO: Qwen temporarily disabled for testing, undo this line later
            else:
                nxt = "call_llama"

        if state.get("verbose", False):
            print(f"[TRACE] router -> {nxt}")

        return nxt

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    # Create a StateGraph with our defined state structure
    graph_builder = StateGraph(AgentState)

    # Add nodes to the graph
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_both", print_both)

    # Define edges:
    # 1. START -> get_user_input (always start by getting user input)
    graph_builder.add_edge(START, "get_user_input")

    # 2. get_user_input -> [conditional] -> call_llm OR END
    #    Uses route_after_input to decide based on state.should_exit
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            "get_user_input": "get_user_input",
            END: END
        }
    )

    # 4. call_llama -> print_both and call_qwen -> print_both (join)
    graph_builder.add_edge("call_llama", "print_both")
    graph_builder.add_edge("call_qwen", "print_both")

    # 5. print_both -> get_user_input (loop back for next input)
    graph_builder.add_edge("print_both", "get_user_input")

    # Compile the graph into an executable form
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph

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

def main():
    """
    Main function for the LangGraph agent with tools.
    """
    print("=" * 50)
    print("LangGraph Agent with Tools")
    print("=" * 50)
    print()

    thread_id = "chat-1"
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string("tools_checkpoints.db") as checkpointer:
        print("Creating LangGraph...")
        graph = create_graph(llm_with_tools, checkpointer=checkpointer)
        print("Graph created successfully!")

        print("Saving graph visualization...")
        save_graph_image(graph)

        initial_state: AgentState = {
            "user_input": "",
            "should_exit": False,
            "verbose": False,
            "skip_input": False,
            "messages": [SystemMessage(content="You are a helpful assistant with access to various tools.")],
        }

        state = graph.get_state(config)
        if state.next:
            print("\n🔄 Resuming from checkpoint...")
            graph.invoke(None, config=config)
        else:
            print("\n▶️ Starting new chat...")
            graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    main()