# Review of Qwen Tool-Use Examples

This review is not just about whether the examples can produce a tool call. The goal is to judge whether they are good reference implementations for future coding agents. The main criteria are abstraction fidelity, model-format correctness, maintainability, and safety. The clean OpenAI assistant in [open_ai_assistant.py](/c:/Users/benpk/School/CSFiles/AI_Agent/ai_agents_workshop/demos/tool_use/cli/open_ai_assistant.py) is the baseline pattern to preserve where possible.

## Evaluation Criteria

- Alignment with LangChain and LangGraph abstractions
- Correctness of the tool-calling protocol and message types
- Maintainability and duplication cost
- Practicality for local Qwen runtimes
- Safety, observability, and failure handling

## Baseline Expectations from the Docs

The copied docs support a standard tool-calling loop:

- [langchain_integrations_qwen.md](/c:/Users/benpk/School/CSFiles/AI_Agent/ai_agents_workshop/demos/tool_use/docs/langchain_integrations_qwen.md) shows that Qwen can support `bind_tools(...)` through LangChain integration.
- [langchain_tool_calling.md](/c:/Users/benpk/School/CSFiles/AI_Agent/ai_agents_workshop/demos/tool_use/docs/langchain_tool_calling.md) shows the expected client-side flow: invoke the model, inspect `AIMessage.tool_calls`, run the tool, and return a matching `ToolMessage(tool_call_id=...)`.
- [langchain_tools.md](/c:/Users/benpk/School/CSFiles/AI_Agent/ai_agents_workshop/demos/tool_use/docs/langchain_tools.md) reinforces that tool definitions should normally come from LangChain tools rather than being manually re-described in parallel JSON schemas.

Those expectations matter because the demos are intended to become prompt context for future coding agents. If the reference examples teach custom message formats and manual parsing as the default, future agents are likely to copy those same patterns.

## Review: `qwen25vl_assistant.py`

Reference: [qwen25vl_assistant.py](/c:/Users/benpk/School/CSFiles/AI_Agent/ai_agents_workshop/examples/agents/core/assistant/qwen25vl_assistant.py)

### What it does well

- It still uses LangGraph for the outer orchestration loop instead of collapsing everything into a single procedural script.
- It defines tool schemas explicitly, which at least makes the model contract visible.
- It limits tool-calling iterations to prevent obvious infinite loops.
- It is aimed at direct local `transformers` inference, which is a legitimate compatibility target.

### What it does poorly

- It rebuilds tool schemas manually instead of reusing the existing LangChain tool objects as the source of truth.
- It introduces a second message history, `conversation_history`, alongside LangGraph `messages`, which creates two parallel state systems.
- It parses tool calls with a regex over `<tool_call>...</tool_call>` blocks, which is brittle and model-format-specific.
- It feeds tool results back as a `"user"` message instead of a proper tool message. The file even contains a TODO acknowledging that this is wrong.
- It relies on Qwen-specific prompt preprocessing rather than preserving the higher-level LangChain tool-calling contract.
- It increases maintenance risk because any model formatting drift now has to be handled manually.

### Why the coding agent likely chose this design

- A raw local `transformers` model does not naturally return LangChain-native `tool_calls` metadata.
- The coding agent appears to have optimized for "make local Qwen run directly" rather than "preserve the same abstraction boundary as the OpenAI demo."
- The `NousFnCallPrompt` path was likely used as a coercion layer to make Qwen emit a parseable function-call format.

### Verdict

This makes sense as a low-level experiment for raw local inference. It does not make sense as a clean teaching example for future coding agents if your goal is to preserve LangChain and LangGraph abstractions.

## Review: `qwen3vl_assistant.py`

Reference: [qwen3vl_assistant.py](/c:/Users/benpk/School/CSFiles/AI_Agent/ai_agents_workshop/examples/agents/core/assistant/qwen3vl_assistant.py)

### What it does well

- It keeps the same basic LangGraph orchestration shape as the other assistants.
- It preserves a bounded tool loop.
- It shows pragmatic fallback loading logic for a model family with uneven support across libraries and versions.
- It targets direct local execution, which is useful if the runtime environment cannot rely on a serving layer.

### What it does poorly

- It inherits almost all of the architectural problems from `qwen25vl_assistant.py`.
- It still duplicates tool definitions instead of reusing LangChain tool metadata.
- It still uses custom `conversation_history` rather than a single LangGraph message history.
- It still relies on regex parsing and nonstandard tool-result injection.
- The multi-branch model loading path makes the example even harder to reason about as a canonical reference.
- The fallback logic makes behavior less predictable, because the same file may run through materially different model classes depending on the environment.

### Why the coding agent likely chose this design

- Qwen3-VL support is less mature and more variable, so the coding agent was probably optimizing for compatibility first.
- Manual integration provides an escape hatch when first-class tool-calling support is inconsistent.
- The design suggests "get the local model working by any means necessary" rather than "teach the cleanest long-term pattern."

### Verdict

This is a reasonable compatibility probe. It is a poor canonical example if the goal is to teach future agents the right abstractions and message protocol.

## Review: `qwen_assistant_clean.py`

Reference: [qwen_assistant_clean.py](/c:/Users/benpk/School/CSFiles/AI_Agent/ai_agents_workshop/examples/agents/core/assistant/qwen_assistant_clean.py)

### What it does well

- It keeps the same LangChain tool-binding pattern as the OpenAI demo.
- It uses `response.tool_calls` and `ToolMessage(tool_call_id=...)` correctly.
- It keeps `messages` as the central conversation state instead of inventing a parallel message format.
- It minimizes custom integration code and therefore reduces formatting risk.
- It is the easiest example to maintain, compare, and reuse as prompt context.
- It best matches your stated goal of using proper LangChain and LangGraph abstractions.

### What it does poorly

- It depends on an OpenAI-compatible serving layer such as `vLLM`, not raw local `transformers` inference.
- Some comments overstate the conclusion and imply that this path is universally correct, when it is really the preferred path under a specific runtime assumption.
- It should state compatibility assumptions more explicitly, especially around which local runtimes actually preserve tool-calling semantics correctly.
- It should mention that server-side tool-calling support varies by model and runtime, even if the API surface looks OpenAI-compatible.

### Why the coding agent likely chose this design

- This design preserves the highest-value abstraction boundary in the codebase.
- If the local runtime can emulate the OpenAI tool-calling protocol, then manual parsing becomes unnecessary.
- It intentionally mirrors the existing clean OpenAI assistant, which is exactly what you want if these demos will be reused as context for future coding agents.

### Verdict

This is the preferred foundation for future clean Qwen demos. It is the best canonical example of the three, with the caveat that the runtime assumptions should be documented more carefully.

## Does the Manual Local Approach Make Sense?

Yes, but only under a narrower goal: "run raw local Qwen through `transformers` with tool use by any means necessary."

No, if the goal is to preserve LangChain and LangGraph abstractions and create clean future examples for coding agents.

That is the core tradeoff. If fully direct local inference is a hard requirement, some amount of custom bridging may be unavoidable. If a local adapter service such as `vLLM` is acceptable, the cleaner and safer choice is to preserve the OpenAI-style LangChain tool-calling loop.

## Suggestions for Cleaner and Safer Future Demos

### Primary recommendation

Make the OpenAI-style LangChain and LangGraph message loop the default Qwen pattern:

- use an OpenAI-compatible local serving layer where possible
- keep `messages` as the single source of truth
- bind tools with `llm.bind_tools(tools)`
- consume tool requests through `AIMessage.tool_calls`
- return tool results through `ToolMessage(tool_call_id=...)`

This is the path that best preserves your abstractions and gives future coding agents the right pattern to copy.

### Secondary recommendation

If you must support raw local `transformers`, keep that as a separate advanced compatibility example, not the canonical one:

- label it clearly as a fallback for runtimes that cannot surface standard tool-calling metadata
- explain that it is a compatibility bridge, not the preferred design
- isolate the model-specific formatting logic so it does not contaminate the default examples

### Concrete improvements

- Reuse existing LangChain tool objects instead of rebuilding parallel JSON tool schemas by hand.
- Prefer one state model. In practice that should be LangGraph `messages`.
- Avoid regex parsing as the primary protocol mechanism.
- Never feed tool outputs back as fake `"user"` messages.
- Centralize constants such as system prompt and max tool iterations.
- Add explicit handling for malformed tool arguments, unknown tool names, and tool exceptions.
- Document the exact runtime assumptions for each demo.
- Separate "clean example" files from "compatibility experiment" files in both naming and directory placement.

## Recommended Future `demos/tool_use/cli` Layout

- One canonical Qwen demo that mirrors [open_ai_assistant.py](/c:/Users/benpk/School/CSFiles/AI_Agent/ai_agents_workshop/demos/tool_use/cli/open_ai_assistant.py) as closely as possible while using a local OpenAI-compatible runtime.
- One optional raw-local compatibility demo with strong caveat labeling.
- Shared tool definitions reused across OpenAI and Qwen demos rather than redefined per example.

## Normative Interface Expectations for Future Implementations

Future clean demos should treat these interfaces as the default contract:

- LangChain tool binding via `llm.bind_tools(tools)`
- Model output consumed through `AIMessage.tool_calls`
- Tool results returned via `ToolMessage(tool_call_id=...)`
- LangGraph state centered on a single message history, ideally `messages`
- Any raw-local fallback should still translate back into the same conceptual tool-call lifecycle, even if the adapter code is model-specific

## Conclusion

`qwen_assistant_clean.py` is the right baseline to evolve. The two manual examples are useful as experiments and compatibility probes, but they should not be treated as the main reference examples for future coding agents. If the purpose of these demos is to teach clean patterns, they should optimize for protocol correctness and abstraction consistency over "pure local" minimal dependencies.
