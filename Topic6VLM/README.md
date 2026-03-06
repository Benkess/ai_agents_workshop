# Topic 6: Vision-Language Models (VLM)

This directory is my portfolio submission for [Topic 6: Vision-Language Models (VLM)](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic6VLM/vlm.html).

## Environment Setup

If `python` on your machine points to Python 2, replace `python` with `python3` in the commands below.

```bash
cd path/to/ai_agents_workshop
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install dependencies per exercise:

- Exercise 1: follow the dependency section in [exercise_1/README.md](exercise_1/README.md)
- Exercise 2: `python -m pip install -r Topic6VLM/exercise_2/requirements.txt`

Deactivate the environment when done:

```bash
deactivate
```

## Table of Contents

- [exercise_1/](exercise_1/)
- [exercise_2/](exercise_2/)
- [instructions/](instructions/)
- [context/](context/)

## Exercise 1: Vision-Language LangGraph Chat Agent

### Launch

Run the app directly from the repository root:

```bash
python3 Topic6VLM/exercise_1/app.py
```

Optional direct-launch mode:

```bash
python3 Topic6VLM/exercise_1/app.py --mode direct --model gpt-5.2 --api-key-env OPENAI_API_KEY
```

### Key Files

- [exercise_1/app.py](exercise_1/app.py)
- [exercise_1/agent.py](exercise_1/agent.py)
- [exercise_1/configs/](exercise_1/configs/)
- [exercise_1/README.md](exercise_1/README.md)

### Outputs

- [exercise_1/outputs/test_1.png](exercise_1/outputs/test_1.png)
- [exercise_1/outputs/test_2.png](exercise_1/outputs/test_2.png)
- [exercise_1/outputs/test_3.png](exercise_1/outputs/test_3.png)

### Known Issues

- You must set something as your `OPENAI_API_KEY` enviroment variable even if not using openai. It will not be used unless using a gpt model.
- The first time you start a conversation it will not work and you must hit the back button then try again.

## Exercise 2: Video-Surveillance Agent

### Launch

Run from the repository root:

```bash
python3 -m Topic6VLM.exercise_2.video_surveillance_agent.cli --video-path /path/to/video.mp4
```

Optional run from inside `Topic6VLM/exercise_2`:

```bash
python3 -m video_surveillance_agent.cli --video-path /path/to/video.mp4
```

### Key Files

- [exercise_2/video_surveillance_agent/](exercise_2/video_surveillance_agent/)
- [exercise_2/record_webcam_mp4.py](exercise_2/record_webcam_mp4.py)
- [exercise_2/agent_state_obs_api/](exercise_2/agent_state_obs_api/)
- [exercise_2/README.md](exercise_2/README.md)

### Outputs

- [exercise_2/outputs/example.txt](exercise_2/outputs/example.txt)

## Resources

- [Topic instructions: `vlm.md`](instructions/vlm.md)
- [VLM guide](instructions/vlm_guide.md)
- [Image generation guide](instructions/image_generation_guide.md)
