# tool_text_to_speech.py

# Import necessary libraries
from langchain.tools import tool
import json
from pydantic import BaseModel, Field


class TTSArgs(BaseModel):
    text: str = Field(..., description="The text to convert to speech")


@tool(args_schema=TTSArgs)
def text_to_speech(text: str) -> str:
    """Convert text to speech and play it through speakers.

    Args:
        text: text to speak

    Returns JSON string: {"success": true} or {"success": false, "error": "message"}.
    """
    try:
        import pyttsx3

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