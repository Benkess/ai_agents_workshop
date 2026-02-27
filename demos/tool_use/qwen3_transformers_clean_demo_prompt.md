# Prompt: Implement a clean Qwen3 manual (Transformers) tool-use demo + local run instructions

You are a coding agent working in this repo. Your task is to add a **manual Transformers-based** Qwen3 tool-use demo that is as clean, safe, and LangGraph/LangChain-aligned as possible.

This demo is explicitly **NOT** the vLLM/OpenAI-compatible route. It must run with **Hugging Face Transformers** (`model.generate(...)`) in a single Python process, while still preserving clean abstractions (single message state, correct tool message semantics, robust parsing).

---

## Goal

Create a new CLI demo that runs Qwen3 locally via Transformers and supports tool use in a clean LangGraph loop.

### Deliverables (new files)

1. `demos/tool_use/cli/qwen3_transformers_assistant.py`
2. `demos/tool_use/cli/_qwen_transformers_adapter.py`

Optional (nice-to-have):
- `demos/tool_use/docs/qwen3_transformers_manual_demo.md` (short usage + design notes)

---

## Non-goals / strict prohibitions

- **NO vLLM** and **NO OpenAI-compatible HTTP** in this demo.
- **NO** parallel history like `conversation_history` dict-lists. LangGraph `messages` is the **only** source of truth.
- **NO** “tool result as user message” (never append tool output as `HumanMessage`).
- **NO** regex that only handles one tool call. Parsing must handle multiple tool calls and malformed JSON safely.
- Do not reimplement tool logic; reuse existing LangChain tools from `demos/tool_use/tools/assistant/`.

---

## Clean architecture contract

### State
Use LangGraph state with a single field:
- `messages: list[AnyMessage]` (LangChain messages)

CLI flags like `verbose`, `should_exit`, etc. are fine, but **messages** must remain canonical.

### Agent loop shape (must match the “clean” pattern)
Repeat up to `max_tool_calling_iterations` (e.g., 5):
1. Call model adapter: `response = adapter.invoke(messages)`
2. Append response `AIMessage` to `messages`
3. If `response.tool_calls` exists:
   - execute each tool (allow-list by name)
   - append `ToolMessage(content=result, tool_call_id=<id>)` for each tool call
   - continue loop
4. Else break and print assistant response

This is the same semantic pattern used by the OpenAI/vLLM demos; only the model invocation differs.

---

## The adapter: `_qwen_transformers_adapter.py`

Implement a small class that looks like a chat model from the outside:

```python
class QwenTransformersAdapter:
    def __init__(...):
        ...

    def invoke(self, messages: list[AnyMessage]) -> AIMessage:
        ...
```

### Adapter responsibilities
1. Convert LangChain messages → Qwen/Transformers chat-template input
2. Inject tool definitions into the prompt (function calling protocol)
3. Run `model.generate()`
4. Decode output text
5. Parse tool calls from output text into structured `tool_calls`
6. Return `AIMessage(content=<raw_model_text>, tool_calls=<parsed_calls>)`

#### Parsing requirements
- Support **multiple tool calls** in one generation.
- If a tool call JSON is malformed:
  - do **not** crash
  - treat it as plain text (no tool call) OR return a tool call with an error, but keep the loop safe.
- Generate stable IDs for tool calls (e.g., `call_<uuid4>`).

#### Message conversion requirements
You must round-trip tool results correctly:

- In LangGraph, tool results are `ToolMessage(content=..., tool_call_id=...)`.
- When converting to the model prompt, map `ToolMessage` to whatever Qwen expects for tool output (often role `"function"`).

**Do not** map tool results to user messages.

### Tool schema injection
Reuse LangChain tools as the source of truth. Create a single helper that converts each tool into a JSON-schema-like spec for the model prompt. Keep it centralized so it doesn’t drift across demos.

If auto-schema extraction is messy, define a minimal schema per tool in one place (still single source of truth in the demo), and keep it consistent with tool signatures.

---

## The demo CLI: `qwen3_transformers_assistant.py`

- Provide a simple interactive CLI (similar to other CLI demos):
  - `verbose` / `quiet` toggles
  - `quit` exits
- Print when tools are called (name + args) in verbose mode.
- Keep tool output sizes bounded (truncate long outputs).

---

## Safety / robustness checklist (must implement)

1. **Allow-list** tool names (only those you registered).
2. **Max iterations** for tool calling.
3. **Argument validation**:
   - if args are not a dict, try to parse JSON; if still bad, tool error ToolMessage.
4. **Tool exception handling**: tool exceptions become ToolMessage error strings.
5. **Tool output truncation** (e.g., 2–4k chars max).
6. **No dual state**: never maintain a second history list.

---

## Local run instructions (must include in docstring + README note)

### Recommended environment
- Python 3.10+
- A CUDA-capable NVIDIA GPU (strongly recommended for Qwen3-VL; CPU will be very slow / may OOM)
- If on Windows, WSL2 Ubuntu + CUDA is often the smoothest path for Transformers GPU.

### Dependencies
Add (or document) these pip installs:

```bash
pip install -U torch transformers accelerate safetensors
# optional, if using quantization:
pip install -U bitsandbytes
```

### Model loading
Use the Hugging Face model repo name:
- `Qwen/Qwen3-VL-4B-Instruct` (default)
Make it configurable via env vars:
- `QWEN_MODEL` (default above)
- `QWEN_DEVICE` (default `"cuda"` if available else `"cpu"`)
- `QWEN_DTYPE` (default `"bfloat16"` or `"float16"` on GPU)

### Example run
From repo root (or correct folder):

```bash
python demos/tool_use/cli/qwen3_transformers_assistant.py
```

---

## Acceptance tests (you must run mentally / ensure code supports)

1. Prompt that triggers calculator:
   - “What is 19*23?” (should call `calculator` tool if your system prompt encourages tool use)
2. Prompt that triggers count_letter:
   - “Count the letter a in banana.”
3. Prompt that should NOT call tools:
   - “Explain what vLLM is.”

Ensure `ToolMessage(tool_call_id=...)` gets appended, and the assistant final response incorporates tool outputs correctly.

---

## Notes on prompting for tool calls

Your system prompt should strongly encourage structured tool calling, e.g.:

- “If a tool can answer faster/more reliably, call it.”
- “Only emit tool call markup when calling a tool; otherwise respond normally.”
- “When calling a tool, emit one or more <tool_call>{JSON}</tool_call> blocks.”

BUT: keep the protocol inside the adapter and parsing; do not leak a second history format.

---

## Output requirements

When you respond, provide:
- The full contents of the two new files
- Any small shared helper code you introduced
- Any minimal changes needed elsewhere (imports, cli launcher registration), but keep changes tight and safe.
