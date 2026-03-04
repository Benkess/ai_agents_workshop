# Vision-Language LangGraph Chat App

This exercise extends the starter Gradio chatbot into a multimodal chat app that carries on a multi-turn conversation about a single uploaded image.

The app uses a LangGraph agent with two message collections:

- `base_messages`: fixed session context containing the system prompt and the uploaded image
- `chat_messages`: ongoing conversational turns managed with a sliding window

Before each model call, the agent sends:

```text
base_messages + trimmed(chat_messages)
```

The Gradio `Exact Model Context` panel shows that combined context in a debug-friendly JSON format with image payloads redacted.

## Features

- Three-stage UI flow:
  - provider/config start screen
  - required image upload screen
  - chat screen
- Multimodal base context built from the uploaded image
- Sliding-window trimming applied only to chat history
- `Internals` panel with:
  - state/debug log
  - exact combined model context
- Saved-agent presets for:
  - OpenAI `gpt-5.2`
  - Ollama `llava`
  - Ollama `qwen3-vl:4b`
- Direct launch mode that still requires image upload before chat
- `quit`, `exit`, `q`, `verbose`, and `quiet` command handling inside chat

## Files

- `agent.py`: LangGraph agent and multimodal context handling
- `app.py`: Gradio UI, session flow, and saved-agent loading
- `configs/`: saved provider presets for this exercise
- `__init__.py`: convenience exports

## Dependencies

Install the packages used by this exercise in your Python environment:

```bash
pip install gradio langgraph langchain-openai langchain-core
```

If you want to use the Ollama presets, make sure Ollama is running locally and the required models are available.

## Saved Agents

The app loads saved agents from `Topic6VLM/exercise_1/configs/*.json`.

Included presets:

- `OpenAI GPT-5.2`
- `Ollama LLaVA`
- `Ollama Qwen3-VL 4B`

The Ollama presets use:

```text
base_url = http://localhost:11434/v1
api_key = ollama
```

## Launch

Start with the setup screen:

```python
from Topic6VLM.exercise_1 import launch_start_app

launch_start_app(inbrowser=True)
```

Launch with provider settings passed directly. This skips the config screen but still requires image upload:

```python
from Topic6VLM.exercise_1 import launch_app

launch_app(
    model="gpt-5.2",
    api_key="your_api_key",
    base_url="",
    inbrowser=True,
)
```

Run the module file directly:

```bash
python3 Topic6VLM/exercise_1/app.py
```

Launch directly into the image-upload flow from the CLI:

```bash
python3 Topic6VLM/exercise_1/app.py --mode direct --model gpt-5.2 --api-key-env OPENAI_API_KEY
```

Launch the full start screen on a specific port without opening the browser:

```bash
python3 Topic6VLM/exercise_1/app.py --mode start --server-port 7861 --no-inbrowser
```

## Session Flow

1. Choose a saved agent or enter provider settings.
2. Continue to the image upload step.
3. Upload the image for the session.
4. Open the chat screen and ask follow-up questions about that image.
5. Use `Quit` to return to setup and start over with a new image.

The uploaded image is fixed for the session. To discuss a different image, return to the earlier screen and start a new session.

## Notes

- Empty messages are blocked in the UI and skipped safely if submitted anyway.
- The uploaded image is always included in model context through `base_messages`.
- The visible chat history does not show the auto-generated base multimodal prompt.
- The `Exact Model Context` panel redacts image base64 while preserving the message structure.
- Advanced settings remain available from the start screen for experimentation with trimming behavior.
