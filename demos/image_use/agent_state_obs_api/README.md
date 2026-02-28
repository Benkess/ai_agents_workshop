# Agent State Obs API

## Overview

This system exposes a vision-language observation agent behind a small HTTP API. A client sends an image plus a prompt such as `"is the red cube on the table"`, and the server returns a structured observation result with `value`, `failure_mode`, and `reason`.

The project is split into two installable packages:

- `agent_state_obs_api_server`: Flask server plus the LangGraph/OpenAI agent implementation
- `agent_state_obs_api_client`: minimal client package that only depends on `requests`

## Note on environments

The client is designed to run in a separate Python environment from the server. The client package only requires `requests`, so it can be installed without LangChain, LangGraph, or OpenAI-related dependencies.

## Server setup

Install the server package from `agent_state_obs_api_server/`:

```bash
cd agent_state_obs_api_server
python -m pip install -e .
```

Set your OpenAI API key before starting the server:

```bash
export OPENAI_API_KEY=your_key_here
```

The default server config file is `agent_state_obs_api_server/config/server.json`.

Start the server:

```bash
cd agent_state_obs_api_server
python -m agent_state_obs_api_server.server --config config/server.json
```

Or override host and port directly:

```bash
cd agent_state_obs_api_server
python -m agent_state_obs_api_server.server --config config/server.json --host 0.0.0.0 --port 5000
```

The server currently supports a single agent implementation configured in `agent_state_obs_api_server/config/server.json`. The default and only supported implementation is `openai`.

## Client setup

Install the client package from `agent_state_obs_api_client/`:

```bash
cd agent_state_obs_api_client
python -m pip install -e .
```

Import and call the client:

```python
from agent_state_obs_api_client import observe

result = observe("example.jpg", "is the red cube on the table")
print(result)
```

This editable install works for external scripts outside this repository. After `python -m pip install -e .` from `agent_state_obs_api_client/`, any script in that Python environment can import:

```python
from agent_state_obs_api_client import observe
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

Success response (`200`):

```json
{
  "success": true,
  "value": "TRUE",
  "failure_mode": "CONFIDENT",
  "reason": "The red cube is clearly visible on the table surface."
}
```

Busy response (`503`):

```json
{
  "busy": true
}
```

Error response (`400` or `500`):

```json
{
  "success": false,
  "error": "Missing required field: prompt"
}
```

The server handles only one request at a time. Concurrent requests are rejected immediately with `503`; they are not queued.

## Running the test

The test script lives at `test/test_observe.py`, which is outside the client package directory.

Run it from `agent_state_obs_api/`:

```bash
python test/test_observe.py path/to/image.jpg --prompt "Describe what you observe in this image." --host http://localhost:5000
```

Or, if you are currently inside `agent_state_obs_api_client/`:

```bash
python ../test/test_observe.py path/to/image.jpg --prompt "Describe what you observe in this image." --host http://localhost:5000
```

## Extending

To add another backend, subclass `agent_state_obs_api_agent/agents/base_obs_agent.py` and implement `query(image_b64, mime_type, prompt) -> dict`. Then update `agent_state_obs_api_server/config/server.json` and the server agent factory to select the new implementation.
