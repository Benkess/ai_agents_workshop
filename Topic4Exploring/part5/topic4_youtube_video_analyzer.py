import os
import re
import sys
from typing import Optional

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)

# -------- Helpers --------

YOUTUBE_ID_RE = re.compile(
    r"""(?x)
    (?:v=|\/)              # v= or a slash
    ([0-9A-Za-z_-]{11})    # capture the 11-char video id
    """
)

def extract_video_id(user_input: str) -> Optional[str]:
    """Extract YouTube video id from a URL or return the input if it already looks like an id."""
    s = user_input.strip()

    # If user already passed a bare 11-char id
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", s):
        return s

    m = YOUTUBE_ID_RE.search(s)
    if m:
        return m.group(1)
    return None


# -------- Tool --------
@tool
def get_youtube_transcript(video_id: str) -> str:
    """
    Fetch the transcript of a YouTube video by video ID.
    Returns a single string (all transcript snippets joined).
    """
    # The library’s README emphasizes passing the ID (not the URL) and shows .fetch(video_id).
    # Default language behavior: tries English unless you specify languages. :contentReference[oaicite:4]{index=4}
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        # fetched is iterable; each snippet has .text (see README examples). :contentReference[oaicite:5]{index=5}
        return " ".join(snippet.text for snippet in fetched)
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        return f"ERROR: No transcripts available for video_id={video_id}. Details: {e}"
    except (VideoUnavailable, CouldNotRetrieveTranscript) as e:
        return f"ERROR: Could not retrieve transcript for video_id={video_id}. Details: {e}"
    except Exception as e:
        # Includes possible IP blocks; README notes YouTube may block cloud IPs and mentions proxies. :contentReference[oaicite:6]{index=6}
        return f"ERROR: Unexpected failure fetching transcript for video_id={video_id}. Details: {e}"


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY env var.")
        return 2

    # Basic CLI usage:
    #   python topic4_youtube_video_analyzer.py "<youtube url or id>" "optional question"
    if len(sys.argv) < 2:
        print('Usage: python topic4_youtube_video_analyzer.py "<youtube url or id>" ["optional question"]')
        return 2

    video_input = sys.argv[1]
    video_id = extract_video_id(video_input)
    if not video_id:
        print("Could not parse a YouTube video id from your input.")
        return 2

    user_question = sys.argv[2] if len(sys.argv) >= 3 else ""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_react_agent(llm, tools=[get_youtube_transcript])

    prompt = f"""
You are an educational assistant.

1) Use the tool to fetch the transcript for video id: {video_id}
2) Produce:
   - A concise summary (5-10 bullets)
   - Key concepts (with 1-2 sentence explanations each)
   - 8 quiz questions with answers (mix factual + conceptual)

If transcript tool returns an ERROR, explain what likely happened and suggest next steps.
""".strip()

    if user_question.strip():
        prompt += f"\n\nUser’s extra request:\n{user_question.strip()}"

    result = agent.invoke({"messages": [("user", prompt)]})

    # LangGraph returns messages; print last assistant message content.
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        # Works for both tuples and message objects depending on versions
        content = getattr(last, "content", None) or (last[1] if isinstance(last, tuple) else str(last))
        print(content)
    else:
        print("No output messages returned by agent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
