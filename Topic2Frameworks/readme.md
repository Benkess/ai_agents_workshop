# Topic 2: Agent Orchestration Frameworks
This repo is part of my portfolio for the UVA Agentic AI course. This is my submission for [Topic 2: Agent Orchestration Frameworks](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic2Frameworks/Agent%20Control%20Flow%20Frameworks.html).

## Enviroment Setup
> **Note:** if `python` does not default to `python3` on your system, then substitute `python3` for `python` in the following commands.

Navigate to the root directory:
```bash
cd path/to/Topic2Frameworks
```
Use the venv module to create a new virtual environment:
```bash
python -m venv .venv
```

Activate the virtual environment:
```bash
# On macOS/Linux.
source .venv/bin/activate
```
```bash
# On Windows.
.venv\Scripts\activate
```

Upgrade pip:
```bash
python -m pip install --upgrade pip
```
```bash
# Also upgrade these for TensorFlow or PyTorch
python -m pip install --upgrade pip wheel setuptools
```

Use pip to install all the packages listed in your requirements.txt:
```bash
pip install -r requirements.txt
```

Install the correct PyTorch version:
```bash
pip3 install torch torchvision
#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# Install PyTorch first (CUDA 13.0)
# python -m pip install torch torchvision `
#   --index-url https://download.pytorch.org/whl/cu130 `
#   --extra-index-url https://pypi.org/simple `
#   --timeout 120
```
> **Note:** 
> For specific PyTorch installs see [get-started](https://pytorch.org/get-started/locally/).
> See your CUDA version with `nvidia-smi`.

When you are done working in the virtual environment, you can deactivate it.
```bash
deactivate
```

## Resorces:
- [Lang Graph](https://docs.langchain.com/oss/python/langgraph/graph-api)

---

## Tree

```text
Topic2Frameworks/
├─ .gitignore
├─ requirements.txt
├─ langgraph_simple_llama_agent.py
├─ step1_verbose_quiet.py
├─ step2_empty_input_branch.py
├─ step3_parallel_llama_qwen.py
├─ step4_router_llama_or_qwen.py
├─ step5_chat_history_llama_only.py
├─ step6_chat_history_llama_qwen.py
├─ step7_checkpointing.py
├─ tree.txt
└─ output/
   ├─ step1.txt
   ├─ step2_empty_response.txt
   ├─ step2_skip_empty.txt
   ├─ step3.txt
   ├─ step4.txt
   ├─ step5.txt
   ├─ step6_1.txt
   ├─ step6_2.txt
   ├─ step7.txt
   └─ step7_updated_part_6.txt
```

---

## Table of contents

### Starter Code

* `langgraph_simple_llama_agent.py`
  Unmodified starter program from the assignment. Baseline LangGraph single-agent chat with a HuggingFace Llama model.

### Step 1 — Verbose / Quiet tracing

* `step1_verbose_quiet.py`
  Adds a `verbose` flag to the LangGraph state. If the user types `verbose`, nodes print trace messages; if the user types `quiet`, tracing is disabled.
* `output/step1.txt`
  Terminal log showing tracing toggled on/off.

### Step 2 — Empty input behavior + 3-way branch

* `step2_empty_input_branch.py`
  Demonstrates the model behavior when given empty input, then fixes it by adding a 3-way conditional branch out of `get_user_input` so empty input routes back to itself (never passing empty input to the LLM).
* `output/step2_empty_response.txt`
  Terminal log showing what happens when the program is given an empty input (baseline behavior recorded).
* `output/step2_skip_empty.txt`
  Terminal log showing the corrected behavior where empty input is routed back to `get_user_input`.

### Step 3 — Parallel Llama + Qwen execution

* `step3_parallel_llama_qwen.py`
  Modifies the graph so that after user input, both Llama and a Qwen model run in parallel. A join node prints both results.
* `output/step3.txt`
  Terminal log demonstrating both model outputs for the same prompt.

### Step 4 — Router: Llama *or* Qwen

* `step4_router_llama_or_qwen.py`
  Removes parallel fanout. Adds a router: if user input begins with `Hey Qwen`, route to Qwen; otherwise route to Llama.
* `output/step4.txt`
  Terminal log demonstrating routing behavior.

### Step 5 — Add chat history (Message API), Llama only

* `step5_chat_history_llama_only.py`
  Adds a persistent chat history to the state using the Message API (`SystemMessage`, `HumanMessage`, `AIMessage`). Qwen routing is disabled to validate chat-history works.
* `output/step5.txt`
  Terminal log demonstrating multi-turn conversation context being preserved.

### Step 6 — Multi-agent chat history (Llama + Qwen)

* `step6_chat_history_llama_qwen.py`
  Integrates history with agent switching between Llama and Qwen. Maintains two message histories so each model receives the correct “view” of the conversation. Uses the “other model as user role + name prefixes” pattern (e.g., `Human:`, `Llama:`, `Qwen:`), and adds model-specific system prompts describing the participants.
* `output/step6_1.txt`
  Example multi-turn conversation demonstrating switching.
* `output/step6_2.txt`
  Change system prompt so llama answers as its self instead of qwen.

### Step 7 — Checkpointing + crash recovery

* `step7_checkpointing.py`
  Adds SQLite checkpointing so the graph automatically saves state after every node. Demonstrates crash recovery by restarting the program with the same `thread_id` and resuming conversation without losing history.
* `output/step7.txt`
  Terminal log showing a run and resume behavior.
* `output/step7_updated_part_6.txt`
  Change system prompt so llama answers as its self instead of qwen.

---
