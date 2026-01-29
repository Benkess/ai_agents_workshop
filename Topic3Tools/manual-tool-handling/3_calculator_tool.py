"""
Manual Tool Calling Exercise
Students will see how tool calling works under the hood.
"""

import json
from openai import OpenAI
import ast
import math

# ============================================
# PART 1: Define Your Tools
# ============================================

def get_weather(location: str) -> str:
    """Get the current weather for a location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def _safe_eval(expr: str):
    """Safely evaluate a simple arithmetic expression using AST traversal."""
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
        if isinstance(node, ast.Tuple):
            return tuple(_eval_node(elt) for elt in node.elts)
        raise ValueError(f"Unsupported expression: {type(node)}")

    parsed = ast.parse(expr, mode="eval")
    return _eval_node(parsed)


def calculator(payload: str) -> str:
    """Evaluate an arithmetic expression safely.

    Payload (string): JSON object with key `expression`, e.g.
      {"expression": "2 + 3*(4 - 1)"}

    The evaluator accepts numeric literals, parentheses, binary operators
    `+ - * / // % **` and unary `+`/`-`. Whitespace in the expression is allowed.

    Returns a JSON string: {"success": True, "result": <number>} or
    {"success": False, "error": "message"}.
    """
    try:
        params = json.loads(payload)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Invalid JSON payload: {e}"})

    try:
        expr = params.get("expression")
        if expr is None:
            raise ValueError("Missing 'expression' field")
        # Allow whitespace; AST parsing will handle it. Don't modify expression.
        value = _safe_eval(expr)
        return json.dumps({"success": True, "result": value})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def area_tool(payload: str) -> str:
    """Compute area for simple shapes.

    Payload (string): JSON object with key `shape` and shape-specific params.
      Circle:    {"shape": "circle", "radius": 3}
      Rectangle: {"shape": "rectangle", "width": 3, "height": 4}
      Triangle:  {"shape": "triangle", "base": 3, "height": 4}

    Returns JSON string: {"success": True, "result": <area>} or
    {"success": False, "error": "message"}.
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
# PART 2: Describe Tools to the LLM
# ============================================

# This is the JSON schema that tells the LLM what tools exist
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    }
                },
                "required": ["location"]
            }
        }
    }
    ,
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic expressions. Expects a JSON string in parameter 'payload' containing {\"expression\": \"2 + 3*(4-1)\"}. Supports + - * / // % ** unary +/- and parentheses. Whitespace is allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "string",
                        "description": "A JSON string with key 'expression' containing the arithmetic expression to evaluate"
                    }
                },
                "required": ["payload"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "area",
            "description": "Compute area for simple shapes. Expects a JSON string in parameter 'payload' describing the shape: circle {\"shape\":\"circle\",\"radius\":3}, rectangle {\"shape\":\"rectangle\",\"width\":3,\"height\":4}, triangle {\"shape\":\"triangle\",\"base\":3,\"height\":4}. Returns area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "string",
                        "description": "A JSON string describing the shape and its dimensions"
                    }
                },
                "required": ["payload"]
            }
        }
    }
]


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Start conversation with user query
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": user_query}
    ]
    
    print(f"User: {user_query}\n")
    
    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Call the LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,  # ← This tells the LLM what tools are available
            tool_choice="auto"  # Let the model decide whether to use tools
        )
        
        assistant_message = response.choices[0].message
        
        # Check if the LLM wants to call a tool
        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")
            
            # Add the assistant's response to messages
            messages.append(assistant_message)
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")
                
                # THIS IS THE MANUAL DISPATCH
                # In a real system, you'd use a dictionary lookup
                if function_name == "get_weather":
                    result = get_weather(**function_args)
                elif function_name == "calculator":
                    # The `calculator` tool expects a `payload` string (JSON) containing an `expression`.
                    result = calculator(**function_args)
                elif function_name == "area":
                    # The `area` tool expects a `payload` string (JSON) describing the shape and dimensions.
                    result = area_tool(**function_args)
                else:
                    result = f"Error: Unknown function {function_name}"
                
                print(f"  Result: {result}")
                
                # Add the tool result back to the conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })
            
            print()
            # Loop continues - LLM will see the tool results
            
        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {assistant_message.content}\n")
            return assistant_message.content
    
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