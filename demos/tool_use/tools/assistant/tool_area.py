# tool_area.py

# Import necessary libraries
from langchain.tools import tool
import json
import math
from typing import Optional
from pydantic import BaseModel, Field


class AreaArgs(BaseModel):
    shape: str = Field(..., description="The shape to compute area for: circle, rectangle, triangle.")
    radius: Optional[float] = Field(None, description="Radius of the circle (required if shape is circle)")
    width: Optional[float] = Field(None, description="Width of the rectangle (required if shape is rectangle)")
    height: Optional[float] = Field(None, description="Height of the rectangle or triangle (required if shape is rectangle or triangle)")
    base: Optional[float] = Field(None, description="Base of the triangle (required if shape is triangle)")


@tool(args_schema=AreaArgs)
def area(shape: str, radius: Optional[float] = None, width: Optional[float] = None, height: Optional[float] = None, base: Optional[float] = None) -> str:
    """Compute area for simple shapes.

    Returns a JSON string: {"success": true, "result": <area>} or
    {"success": false, "error": "message"}.
    """
    try:
        s = (shape or "").lower()
        if s == "circle":
            if radius is None:
                raise ValueError("Missing 'radius' for circle")
            value = math.pi * radius * radius
            return json.dumps({"success": True, "result": value})
        if s == "rectangle":
            if width is None or height is None:
                raise ValueError("Missing 'width' or 'height' for rectangle")
            value = width * height
            return json.dumps({"success": True, "result": value})
        if s == "triangle":
            if base is None or height is None:
                raise ValueError("Missing 'base' or 'height' for triangle")
            value = 0.5 * base * height
            return json.dumps({"success": True, "result": value})
        raise ValueError(f"Unsupported shape for area: {shape}")
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})