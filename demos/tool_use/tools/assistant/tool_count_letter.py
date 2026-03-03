# tool_count_letter.py

# Import necessary libraries
from langchain.tools import tool
import json
from pydantic import BaseModel, Field


class CountLetterArgs(BaseModel):
    letter: str = Field(..., description="The character or substring to count")
    text: str = Field(..., description="The text to search")
    case_sensitive: bool = Field(False, description="Whether the count should be case sensitive")


@tool(args_schema=CountLetterArgs)
def count_letter(letter: str, text: str, case_sensitive: bool = False) -> str:
    """Count occurrences of a letter (or substring) in a text.

    Returns a JSON string: {"success": true, "result": <count>} or
    {"success": false, "error": "message"}.
    """
    try:
        if letter is None or text is None:
            raise ValueError("Both 'letter' and 'text' fields are required")
        if not case_sensitive:
            letter = letter.lower()
            text = text.lower()
        count = text.count(letter)
        return json.dumps({"success": True, "result": count})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})