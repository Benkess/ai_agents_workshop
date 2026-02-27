# Qwen3-VL Transformers Manual Tool-Use Demo

This demo adds a clean manual Transformers path for Qwen3-VL tool use without `vLLM` and without an OpenAI-compatible HTTP layer.

## Why this exists

- It keeps the same LangGraph tool loop semantics as the clean OpenAI and vLLM demos.
- It isolates model-specific prompting and parsing inside one adapter module.
- It keeps `messages` as the single conversation state.

## What is different from the vLLM demo

- The model runs locally in-process through Hugging Face Transformers.
- Tool-call parsing is manual inside the adapter.
- There is no HTTP server and no `bind_tools(...)` provider integration.

## Install

```bash
pip install -U torch transformers accelerate safetensors
pip install -U bitsandbytes
```

## Environment variables

- `QWEN_MODEL`: defaults to `Qwen/Qwen3-VL-4B-Instruct`
- `QWEN_DEVICE`: defaults to `cuda` if available else `cpu`
- `QWEN_DTYPE`: defaults to `bfloat16` or `float16` on GPU, `float32` on CPU

## Run

```bash
python demos/tool_use/cli/qwen3_transformers_assistant.py
```

## Design notes

- LangGraph `messages` is the only source of truth.
- The manual adapter owns prompt construction, generation, and tool-call parsing.
- Tool results are appended as `ToolMessage(tool_call_id=...)`, never as user messages.
- Parsing supports multiple `<tool_call>{...}</tool_call>` blocks safely.
- This first version is text-only even though it uses the Qwen3-VL model family.
