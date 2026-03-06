# Exercise 2: Video Surveillance Agent

This exercise analyzes a video by sampling frames every few seconds, sending each sampled frame to a vision-language model, and reporting the timestamps where a person enters or exits the scene.

## Files

- `record_webcam_mp4.py`: optional helper to record a local `.mp4` video from your webcam
- `video_surveillance_agent/cli.py`: command-line entrypoint for the exercise
- `agent_state_obs_api/`: shared observation-agent code, configs, and HTTP starter packages
- `test_exercise_2.py`: local tests for the direct-call wrapper and video logic

## Requirements

Install the exercise dependencies from this directory:

```bash
cd Topic6VLM/exercise_2
python3 -m pip install -r requirements.txt
```

If you are using Ollama locally, make sure Ollama is running and the `llava` model is available.

## Default model setup

The exercise defaults to the local config in [llava_agent.json](/home/user/projects/school/ai_agents_workshop/Topic6VLM/exercise_2/agent_state_obs_api/config/llava_agent.json):

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

That means the default workflow expects:

- Ollama running locally
- the `llava` model installed in Ollama
- the OpenAI-compatible Ollama endpoint available at `http://localhost:11434/v1`

## Running the exercise

### 1. Create or choose a video

You need a video of an empty space where a person enters and exits at some point.

You can record one with the helper script:

```bash
cd Topic6VLM/exercise_2
python3 record_webcam_mp4.py
```

This writes `webcam_recording.mp4` in the current directory. Press `q` to stop recording.

### 2. Run the surveillance agent

From the repository root:

```bash
python3 -m Topic6VLM.exercise_2.video_surveillance_agent.cli --video-path Topic6VLM/exercise_2/webcam_recording.mp4
```

Or from inside `Topic6VLM/exercise_2`:

```bash
python3 -m video_surveillance_agent.cli --video-path webcam_recording.mp4
```

Optional arguments:

- `--config`: path to an alternate agent config JSON file
- `--interval-seconds`: frame sampling interval in seconds, default `2.0`

Example with explicit config and faster sampling:

```bash
python3 -m Topic6VLM.exercise_2.video_surveillance_agent.cli \
  --video-path Topic6VLM/exercise_2/webcam_recording.mp4 \
  --config Topic6VLM/exercise_2/agent_state_obs_api/config/llava_agent.json \
  --interval-seconds 1.0
```

## What the program prints

As each sampled frame is analyzed, the CLI prints the raw observation result dict for that frame.

At the end, it prints an ordered event list like:

```python
{'timestamp_seconds': 24.0, 'event': 'enter'}
{'timestamp_seconds': 58.0, 'event': 'exit'}
```

Non-confident model results are ignored for transition detection, so they do not create false enter/exit events by themselves.

If the provider returns a `does not support tools` error (common with Ollama `llava`), the observation wrapper automatically switches to a text-only JSON fallback mode and continues.

## Using a different model config

If you want to use a different OpenAI-compatible model, create another JSON config with the same schema as `agent_state_obs_api/config/llava_agent.json` and pass it with `--config`.

## Running tests

From the repository root:

```bash
python3 -m pytest -q Topic6VLM/exercise_2/test_exercise_2.py
```

## Troubleshooting

- `ModuleNotFoundError`: run commands from the repository root when using `python3 -m Topic6VLM...`
- `Could not open video`: verify the file path and confirm OpenCV can read the file format
- model connection failures: confirm Ollama is running and that `llava` is installed
- no events detected: try a smaller `--interval-seconds` value so more frames are sampled
