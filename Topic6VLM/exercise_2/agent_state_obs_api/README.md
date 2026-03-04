# Agent State Obs API

## Overview

This exercise now supports two ways to query the observation agent:

- direct in-process calls through `agent_state_obs_api.ObservationAgent`
- the original HTTP server/client split through `agent_state_obs_api_server` and `agent_state_obs_api_client`

Both paths return the same structured observation fields: `value`, `failure_mode`, and `reason`.

The exercise also includes a separate video workflow in `Topic6VLM/exercise_2/video_surveillance_agent` that samples frames from a video, asks the observation agent whether a person is present, and reports enter/exit timestamps.

## Packages

- `agent_state_obs_api`: shared local wrapper and config helpers
- `agent_state_obs_api_server`: Flask server package
- `agent_state_obs_api_client`: HTTP client package
- `agent_state_obs_api_agent`: LangGraph/OpenAI-compatible agent implementation

## Dependencies

Install the server-side dependencies from `agent_state_obs_api_server/`:

```bash
cd agent_state_obs_api_server
python -m pip install -e .
```

For the video surveillance workflow, install OpenCV:

```bash
python -m pip install opencv-python
```

## Direct-call usage

The new local wrapper defaults to the LLaVA config in [llava_agent.json](/home/user/projects/school/ai_agents_workshop/Topic6VLM/exercise_2/agent_state_obs_api/config/llava_agent.json):

```json
{
  "agent": {
    "implementation": "openai",
    "model": "llava",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "api_key_env": null
  }
}
```

Example:

```python
from agent_state_obs_api import ObservationAgent

agent = ObservationAgent()
result = agent.observe_image_path("example.jpg", "Is there a person visible in this frame?")
print(result)
```

Pass `config_path` if you want to use a different provider config.

## HTTP server setup

The original HTTP server and client remain available.

Server configs:

- `agent_state_obs_api_server/config/server.json` for cloud OpenAI
- `agent_state_obs_api_server/config/qwen_server.json` for local Ollama + qwen

Start the server:

```bash
cd agent_state_obs_api_server
python -m agent_state_obs_api_server.server --config config/server.json
```

Or with local Ollama:

```bash
cd agent_state_obs_api_server
python -m agent_state_obs_api_server.server --config config/qwen_server.json
```

## HTTP client usage

Install the client package from `agent_state_obs_api_client/`:

```bash
cd agent_state_obs_api_client
python -m pip install -e .
```

Then call:

```python
from agent_state_obs_api_client import observe

result = observe("example.jpg", "is the red cube on the table")
print(result)
```

## API reference

Endpoint:

```text
POST /observe
```

Request body:

```json
{
  "image_b64": "<base64 encoded image string>",
  "mime_type": "image/jpeg",
  "prompt": "is the red cube on the table"
}
```

Success response:

```json
{
  "success": true,
  "value": "TRUE",
  "failure_mode": "CONFIDENT",
  "reason": "The red cube is clearly visible on the table surface."
}
```

## Video surveillance CLI

Run the surveillance workflow from the repo root:

```bash
python -m Topic6VLM.exercise_2.video_surveillance_agent.cli --video-path path/to/video.mp4
```

Optional arguments:

- `--config` to point at a different agent config
- `--interval-seconds` to change frame sampling cadence

The CLI prints each sampled frame response as it is analyzed, then prints the final ordered list of `enter` and `exit` events.

## Running the existing HTTP test

From `agent_state_obs_api/`:

```bash
python test/test_observe.py path/to/image.jpg --prompt "Describe what you observe in this image." --host http://localhost:5000
```

## Extending

To add another backend, subclass `agent_state_obs_api_agent/agents/base_obs_agent.py` and implement `query(image_b64, mime_type, prompt) -> dict`. Then update the shared factory in `agent_state_obs_api/factory.py` and any relevant config files.
