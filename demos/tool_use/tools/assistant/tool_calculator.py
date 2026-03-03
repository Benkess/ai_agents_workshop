# tool_calculator.py

# Import necessary libraries
from langchain.tools import tool
import json
import ast
from pydantic import BaseModel, Field


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

class CalculatorArgs(BaseModel):
    expression: str = Field(description="The arithmetic expression to evaluate, e.g. 2 + 3*(4-1). The evaluator accepts whitespace in the expression and supports + - * / // % ** and unary +/-.")


@tool(args_schema=CalculatorArgs)
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression.

    Args:
        expression (str): The arithmetic expression to evaluate, e.g. 2 + 3*(4-1). The evaluator accepts whitespace in the expression and supports + - * / // % ** and unary +/-."

    Returns:
        output (str): a JSON string, "output":  {"success": true, "result": <number>} or {"success": false, "error": "message"}.
    """
    try:
        if not expression:
            raise ValueError("Missing 'expression' field")
        # Do not alter whitespace; AST parsing allows it.
        value = _safe_eval(expression)
        return json.dumps({"success": True, "result": value})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})