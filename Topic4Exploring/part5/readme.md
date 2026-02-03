## YouTube Educational Video Analyzer (Agent + Tool Use)

### Project Overview
For this Topic 4 project, I built an **agentic AI system** that analyzes educational YouTube videos using tool calling. The agent accepts a YouTube URL (or video ID), retrieves the video transcript using an external API, and then reasons over the content to generate structured educational outputs.

This project demonstrates:
- LLM tool use via the ReAct pattern
- Integration of a third-party API as a custom LangChain tool
- An agent that decides *when* and *how* to call tools
- Multi-step reasoning over retrieved real-world data

---

### How It Works
1. The user provides a YouTube URL or video ID.
2. The agent extracts the video ID from the input.
3. The agent calls a custom tool (`get_youtube_transcript`) built using the `youtube-transcript-api`.
4. The tool returns the full transcript as a single text string.
5. The agent analyzes the transcript and produces:
   - A concise summary
   - A list of key concepts
   - A set of quiz questions with answers

If no transcript is available, the agent explains the failure and suggests next steps.

---

### Tools Used
- **YouTube Transcript API**  
  Used to fetch captions/transcripts for YouTube videos by video ID.

- **Custom LangChain Tool**  
  The transcript fetcher is wrapped as a tool so the agent can invoke it dynamically.

- **LangGraph ReAct Agent**  
  The agent uses the ReAct pattern to reason about when tool use is necessary and how to incorporate the results into its response.

---

### How to Run

```bash
pip install youtube-transcript-api langchain langgraph langchain-openai
```

Export API Key
```bash
export OPENAI_API_KEY="your_key_here"
```

Run
```bash
python topic4_youtube_video_analyzer.py "https://www.youtube.com/watch?v=VIDEO_ID"
```
```bash
python topic4_youtube_video_analyzer.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```
