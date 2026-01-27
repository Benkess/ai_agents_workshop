"""
Tool Calling with LangChain
Shows how LangChain abstracts tool calling.
"""

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import json
# from openai import OpenAI
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

# ============================================
# Tool Mapping
# ============================================

tools = [get_weather, calculator, area]
tool_map = {tool.name: tool for tool in tools}

# ============================================
# PART 2: Create LLM with Tools
# ============================================

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """
    
    # Start conversation with user query
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided tools when needed."),
        HumanMessage(content=user_query)
    ]
    
    print(f"User: {user_query}\n")
    
    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Call the LLM
        response = llm_with_tools.invoke(messages)
        
        # Check if the LLM wants to call a tool
        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")
            
            # Add the assistant's response to messages
            messages.append(response)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]
                
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")
                
                # Execute the tool
                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"
                
                print(f"  Result: {result}")
                
                # Add the tool result back to the conversation
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
            
            print()
            # Loop continues - LLM will see the tool results
            
        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {response.content}\n")
            return response.content
    
    return "Max iterations reached"


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Test query that requires tool use
    # print("="*60)
    # print("TEST 1: Query requiring tool")
    # print("="*60)
    # run_agent("What's the weather like in San Francisco?")
    
    # print("\n" + "="*60)
    # print("TEST 2: Query not requiring tool")
    # print("="*60)
    # run_agent("Say hello!")
    
    # print("\n" + "="*60)
    # print("TEST 3: Multiple tool calls")
    # print("="*60)
    # run_agent("What's the weather in New York and London?")

    print("="*60)
    print("TEST 4: Test expression calculation")
    print("="*60)
    run_agent("Calculate the expression 2 + 2 * 2")
    
    print("\n" + "="*60)
    print("TEST 5: Test order of operations")
    print("="*60)
    run_agent("Calculate the expression 2 * (2 + 2)")
    
    print("\n" + "="*60)
    print("TEST 6: Test Area calculations")
    print("="*60)
    run_agent("Calculate the area of a circle with radius 5")