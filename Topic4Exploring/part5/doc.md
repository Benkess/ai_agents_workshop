## YouTube Transcript API
Description: Fetch transcripts/captions from YouTube videos. No API key required. Useful for video content analysis.

Documentation: https://github.com/jdepoix/youtube-transcript-api

Installation:

pip install youtube-transcript-api
LangChain Integration: ❌ Custom tool definition required

Example Tool Definition:

from langchain.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi

@tool
def get_youtube_transcript(video_id: str) -> str:
    """Get transcript from a YouTube video given its video ID."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])

## Task
Option 3: YouTube Transcript
Project: "Educational Video Analyzer"

Student provides educational video URL

Agent extracts transcript

Generates: summary, key concepts, quiz questions

See below for more details and starter code!

Why this works:

Reinforces custom tool creation (builds on calculator exercise)

High student engagement (analyze videos they actually watch)

Multiple LLM calls in sequence (fetch → summarize → generate questions)

Visible value (helps with studying!)

Learning outcomes:

Custom tool definition patterns

Multi-step agent workflows

Content transformation pipelines

Implementation Template (Works for Any of These)
Here's a 2-hour timeline that works for all three options:

Hour 1: Setup & Basic Tool

0-15 min: Install packages, get API keys (if needed)

15-45 min: Define custom tool OR use pre-built tool

45-60 min: Test tool in isolation (not in agent yet)

Hour 2: Agent Integration

60-90 min: Integrate tool with simple ReAct agent

90-110 min: Test with various prompts, debug

110-120 min: Optional extension or demo preparation

Detailed Recommendation: Start with YouTube
If I had to pick one for your next project after calculator, I'd choose YouTube Transcript.

Reasoning:

Builds directly on calculator pattern - they just wrote a custom tool, now they write another

No API key friction - can start coding immediately

High engagement - students love analyzing YouTube content

Clear progression - shows how custom tools enable new capabilities

Teaches real skill - most LangChain tools will be custom in production

Starter code for students:

python

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

@tool
def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video by video ID."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])

# Create agent with tool
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [get_youtube_transcript])

# Test it
result = agent.invoke({
    "messages": [("user", "Get the transcript for video dQw4w9WgXcQ and summarize it")]
})
Extensions for faster students:

Add summary formatting (bullet points, key quotes)

Extract chapter timestamps

Answer specific questions about video content

Compare transcripts from multiple videos

This gives you a solid 2-hour project that reinforces custom tool creation while introducing practical information retrieval patterns.

 