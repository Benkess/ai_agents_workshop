# Gradio Chatbot Demo

This demo wraps a simple LangGraph OpenAI chat agent in a Gradio GUI.

## Features

- Gradio chat window with full visible message history
- LangGraph-based agent with sliding-window message trimming
- No SQLite checkpointing; each app launch starts fresh
- Text-only chat flow
- `Quit` button that sends a quit command and closes the current session
- `Internals` toggle with:
  - readable graph/debug flow output
  - exact trimmed model context shown as JSON before each model invocation
- Two launch styles:
  - direct launch with `model`, `api_key`, and `base_url` passed as arguments
  - start screen that collects provider settings before entering chat
- Saved-agent configs loaded from `demos/gradio/chatbot/configs`

## Files

- `agent.py`: LangGraph agent and turn execution logic
- `app.py`: Gradio UI and launch helpers
- `__init__.py`: convenience exports

## Dependencies

Install the packages used by this demo in your Python environment:

```bash
pip install gradio langgraph langchain-openai langchain-core
```

## Launch

Start with the provider config passed directly:

```python
from demos.gradio.chatbot import launch_app

launch_app(
    model="gpt-4o-mini",
    api_key="your_api_key",
    base_url="",
    inbrowser=True,
)
```

Start with a setup page that asks for provider settings:

```python
from demos.gradio.chatbot import launch_start_app

launch_start_app(inbrowser=True)
```

Run the module file directly:

```bash
python demos/gradio/chatbot/app.py
```

Launch directly into chat from the CLI:

```bash
python demos/gradio/chatbot/app.py --mode direct --model gpt-4o-mini --api-key your_api_key
```

Launch the start screen on a specific port without opening the browser:

```bash
python demos/gradio/chatbot/app.py --mode start --server-port 7861 --no-inbrowser
```

Use an environment variable name instead of passing the API key directly:

```bash
python demos/gradio/chatbot/app.py --mode direct --api-key-env OPENAI_API_KEY
```

## Notes

- Empty messages are blocked in the UI by disabling `Send` until text is entered.
- If an empty message still reaches the agent, it is skipped cleanly.
- The conversation shown in the chat window is separate from the trimmed context sent to the model.
- Entering `verbose` or `quiet` as a user message toggles the agent's verbose flag to mirror the CLI example.
- The start screen loads saved agents from `configs/*.json` and falls back to a built-in default if none are present.
