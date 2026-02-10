# Computer Use Agent - Design Documentation

## Overview

This agent controls a computer using vision-language understanding and mouse/keyboard actions. It maintains proper message history using LangChain message types and generates actions through a single tool call per step.

**Key Design Principle**: Everything that goes to the model must be properly formatted with explicit roles (system, user, assistant, tool) to avoid confusing the agent about who said what.

---

## Architecture

### Message History (LangChain Format)

We use LangChain's message types to maintain a proper conversation history:

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

messages = [
    SystemMessage(content="You are controlling a computer..."),
    HumanMessage(content="User goal: Complete the quiz"),
    HumanMessage(content=[{"type": "text", "text": "Screenshot"}, {"type": "image_url", ...}]),
    AIMessage(content="", tool_calls=[{"name": "computer_use", "args": {...}}]),
    ToolMessage(content="Clicked at (x,y)", tool_call_id="..."),
    # ... continues
]
```

**Why this matters**:
- Each message has an explicit `role` (system/user/assistant/tool)
- The model knows who said what (no self-talk)
- History is properly formatted by `apply_chat_template`
- We can serialize/deserialize cleanly

**Alternative we rejected**: Custom types like `ContextMessage` with types like "thought", "action", "result" - these aren't standard roles and confuse the model about conversation structure.

---

## Design Decisions

### 1. Separate System and User Goal Messages

**What we do**:
```python
messages = [
    SystemMessage(content="You are controlling a computer. Left side is messaging..."),
    HumanMessage(content="User goal: {task_description}")
]
```

**Why**:
- System message establishes context and rules
- User goal is a separate instruction (proper role separation)
- Models are trained to treat system vs user messages differently

**Alternative we rejected**: Combining them (`f"System: ...\nUser goal: ..."`) - this loses role information and creates ambiguity.

---

### 2. Thought Inside Tool Call

**What we do**:
```python
{
    "name": "computer_use",
    "parameters": {
        "thought": {  # <-- Reasoning comes first
            "type": "string",
            "description": "Your reasoning: what you see, what to do, any mistakes",
            "required": true
        },
        "action": {
            "type": "string",
            "enum": ["left_click", "type", ...]
        },
        # ... other params
    }
}
```

**Why**:
- Combines ReAct-style reasoning with action in one generation
- Forces model to think before acting
- Eliminates two-phase generation (thought → action)
- Simpler loop: generate once → execute → repeat

**Alternative we rejected**: Separate thought and action phases - this doubled generation time, required manual prompt switching, and the model sometimes generated actions during "thought" phase anyway.

---

### 3. Image Handling (Custom Layer Required)

**The challenge**: We run Qwen2.5-VL locally via Hugging Face Transformers, not through LangChain's ChatQwen API.

**What LangChain expects**:
```python
HumanMessage(content=[
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
])
```

**What Qwen's processor expects**:
```python
messages = [{"role": "user", "content": [{"type": "image", "image": <PIL.Image>}]}]
inputs = processor(text=[...], images=[screenshot], ...)
```

**Our solution**: Hybrid approach
1. Store images in `HumanMessage` using LangChain's format (for portability)
2. When generating, extract PIL images and pass to processor separately
3. Document this as a "local HF adaptation layer"

```python
# In message history (LangChain format)
messages.append(HumanMessage(content=[
    {"type": "text", "text": "Current screenshot"},
    {"type": "image_url", "image_url": {"url": f"temp://screenshot_{step}.png"}}
]))

# When generating (extract for HF processor)
images = [self._extract_image(msg) for msg in messages if has_image(msg)]
inputs = processor(text=[formatted_text], images=images, ...)
```

**Why this matters**:
- Message history remains standard LangChain format
- We can switch to ChatQwen API later without rewriting history logic
- Clear separation between "state" (messages) and "execution" (model invocation)

**Alternative we rejected**: Storing PIL images directly in message content - this isn't LangChain standard and makes serialization impossible.

---

### 4. Tool Call Parsing (Custom Layer Required)

**The challenge**: Local Qwen2.5-VL outputs tool calls as XML text, not structured objects.

**What the model outputs**:
```xml
<tool_call>
{"name": "computer_use", "arguments": {"thought": "I see...", "action": "left_click", ...}}
</tool_call>
```

**What LangChain expects in AIMessage**:
```python
AIMessage(
    content="",
    tool_calls=[{
        "name": "computer_use",
        "args": {"thought": "...", "action": "left_click", ...},
        "id": "call_abc123"
    }]
)
```

**Our solution**: Parse and convert
```python
def parse_tool_call(response_text: str) -> Optional[dict]:
    # 1. Extract XML with regex
    match = re.search(r'<tool_call>\s*(\{.+?\})\s*</tool_call>', response_text, re.DOTALL)
    if not match:
        return None
    
    # 2. Parse JSON
    tool_data = json.loads(match.group(1))
    
    # 3. Convert to LangChain format
    return {
        "name": tool_data["name"],
        "args": tool_data["arguments"],
        "id": f"call_{uuid.uuid4().hex[:8]}"
    }
```

**Why this matters**:
- Tool calls follow LangChain standard format
- Rest of agent loop is provider-agnostic
- We can add better parsing later (e.g., handle multiple calls)

**Alternative we rejected**: Keeping raw XML in context - this doesn't work with ToolMessage format and breaks LangChain tools.

---

### 5. Single Generation Per Step

**What we do**:
```python
def step(self):
    # 1. Screenshot
    screenshot = capture_screenshot()
    messages.append(HumanMessage(content=[...screenshot...]))
    
    # 2. Generate ONCE (model returns thought + action)
    ai_message = generate_response()
    messages.append(ai_message)
    
    # 3. Execute and add result
    if ai_message.tool_calls:
        result = execute_action(ai_message.tool_calls[0]['args'])
        messages.append(ToolMessage(content=result, tool_call_id=...))
```

**Why**:
- One generation = one complete thought-action pair
- Faster (half the inference time)
- Model can't generate actions during "thought" phase anymore
- Cleaner message history

**Alternative we rejected**: Two-phase (thought → action) - this was slower, required phase-switching prompts, and the model often violated phase boundaries.

---

### 6. Handling Non-Tool Responses

**What we do**:
```python
if not ai_message.tool_calls:
    print(f"[Warning] Model generated text instead of tool: {ai_message.content}")
    # DO NOT add to message history
    return True  # Continue loop
```

**Why**:
- System prompt says "always use tool"
- If model violates this, it's a mistake
- Adding non-tool responses to history confuses future steps
- Log for debugging but don't persist

**Alternative we rejected**: Adding all responses to history - this allows the model to "escape" the tool interface and generates unstructured text.

---

### 7. Context Window Management

**The challenge**: Keep recent history while preserving system context.

**What we do**:
```python
def trim_messages(messages: List[BaseMessage], max_pairs: int = 12) -> List[BaseMessage]:
    """
    Keep:
    - First system message
    - User goal message
    - Last N pairs of (HumanMessage + AIMessage + ToolMessage)
    """
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    user_goal = messages[1] if len(messages) > 1 and isinstance(messages[1], HumanMessage) else None
    
    # Get last N interaction triplets
    recent = messages[2:]  # Everything after system+goal
    # Group into triplets and keep last max_pairs
    trimmed = recent[-(max_pairs * 3):]
    
    return system_msgs + ([user_goal] if user_goal else []) + trimmed
```

**Why this works**:
- System message always present (model knows its role)
- User goal always present (model knows its task)
- Recent interactions provide context
- Prevents context overflow

**Parameters**:
- `max_pairs=12` means keep last 12 screenshot-action-result cycles
- Each cycle = 3 messages (HumanMessage with image, AIMessage with tool_call, ToolMessage with result)
- Total ≈ 38 messages (2 system + 36 recent)

**Alternative we rejected**: Hard cap at N messages - this could cut off mid-interaction and lose system context.

---

### 8. Message Formatting for Model

**What we do**:
```python
def generate_response(self) -> AIMessage:
    # 1. Trim context
    trimmed = trim_messages(self.messages, max_pairs=12)
    
    # 2. Convert to dicts
    message_dicts = [msg.dict() for msg in trimmed]
    
    # 3. Apply chat template (model-specific formatting)
    text = self.processor.apply_chat_template(
        message_dicts,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 4. Extract images for processor
    images = [self._extract_image(m) for m in trimmed if self._has_image(m)]
    
    # 5. Generate
    inputs = self.processor(text=[text], images=images, ...)
    output = self.model.generate(**inputs)
    response = self.processor.decode(output)
    
    # 6. Parse and return as AIMessage
    tool_call = parse_tool_call(response)
    return AIMessage(content="", tool_calls=[tool_call] if tool_call else [])
```

**Why each step**:
1. **Trim first** - Prevents context overflow
2. **Convert to dicts** - Required by `apply_chat_template`
3. **Apply template** - Model-specific formatting with special tokens
4. **Extract images** - HF processor needs separate images list
5. **Generate** - Standard HF generation
6. **Parse and convert** - Back to LangChain format

**What happens inside `apply_chat_template`**:
- Adds special tokens (e.g., `<|im_start|>`, `<|im_end|>`)
- Formats roles correctly for Qwen
- Handles tool definitions
- Returns properly tokenized text

**Why we don't manually build prompts**:
- Model-specific formatting is complex
- Special tokens vary by model
- Chat template knows the correct format
- Manual formatting causes tokenization issues

---

## Error Handling

### Coordinate Out of Bounds
```python
if not (0 <= x < self.screen_width and 0 <= y < self.screen_height):
    print(f"[Warning] Coordinate ({x}, {y}) out of bounds")
    return "Error: Click outside screen bounds"
```

**Why**: Agent sometimes generates invalid coordinates. We catch and report rather than crash.

### Tool Call Parsing Failure
```python
tool_call = parse_tool_call(response)
if not tool_call:
    print(f"[Warning] Failed to parse tool call from: {response[:200]}")
    return AIMessage(content=response)  # No tool_calls
```

**Why**: Model sometimes generates malformed XML. We log the issue and continue (next step gets screenshot showing nothing changed).

### Image Extraction Failure
```python
try:
    image = self._extract_image(message)
except Exception as e:
    print(f"[Warning] Failed to extract image: {e}")
    image = None
```

**Why**: Graceful degradation if image handling breaks. Better to continue without one image than crash.

---

## Testing Strategy

### Manual Inspection Points

1. **Message history** - After each step, print:
   ```python
   print(f"History: {len(self.messages)} messages")
   print(f"Last 3: {[type(m).__name__ for m in self.messages[-3:]]}")
   ```

2. **Tool calls** - Verify format:
   ```python
   if ai_msg.tool_calls:
       print(f"Tool: {ai_msg.tool_calls[0]['name']}")
       print(f"Thought: {ai_msg.tool_calls[0]['args'].get('thought')}")
   ```

3. **Coordinates** - Check normalization:
   ```python
   print(f"Normalized [{norm_x}, {norm_y}] → Absolute ({abs_x}, {abs_y})")
   ```

### Debug Endpoints

**Context viewer** (`/context`):
- Shows last N messages with roles
- Displays thought+action pairs
- Auto-refreshes

**State dump** (`/state`):
- Full message history as JSON
- Screen dimensions
- Current step count

---

## Deployment Checklist

- [ ] Screen resolution detected correctly (`pyautogui.size()`)
- [ ] Message roles are correct (system/user/assistant/tool)
- [ ] Images handled properly (LangChain format + HF extraction)
- [ ] Tool calls parsed correctly (XML → LangChain format)
- [ ] Context trimming preserves system messages
- [ ] Non-tool responses are logged but not saved
- [ ] Coordinates normalized correctly (0-1000 → screen pixels)
- [ ] Debug server accessible from network

---

## Future Improvements

### Potential Enhancements
1. **Better tool parsing** - Handle multiple tool calls per turn
2. **Smarter trimming** - Summarize very old messages instead of dropping
3. **Checkpointing** - Save/restore message history
4. **Image caching** - Don't re-send identical screenshots
5. **Action validation** - Pre-check coordinates before executing

### Migration Path to LangGraph
If we want full LangGraph integration later:
1. Message history is already LangChain-compatible ✓
2. Add state graph with nodes for (screenshot → generate → execute)
3. Use LangGraph's trimming utilities
4. Add conditional edges for error handling

The current architecture is "LangGraph-ready" - we're just not using the state machine yet.

---

## Common Pitfalls (What We Avoid)

### ❌ Manual Prompt Building
```python
# DON'T DO THIS
prompt = f"System: {system}\nHistory:\n"
for msg in history:
    prompt += f"{msg.type}: {msg.content}\n"
```
**Why bad**: Loses role information, breaks tokenization, model gets confused.

### ❌ Editing Tool Calls
```python
# DON'T DO THIS
tool_call_str = f"Action: {tool_call['name']} with {tool_call['args']}"
messages.append(HumanMessage(content=tool_call_str))
```
**Why bad**: Model didn't generate that text, adding it as HumanMessage makes model think user said it.

### ❌ Mixing Content Types
```python
# DON'T DO THIS
messages.append(HumanMessage(content="Screenshot captured"))  # Where's the image?
```
**Why bad**: Model expects image but doesn't get one, breaks visual reasoning.

### ❌ Ignoring Tool Results
```python
# DON'T DO THIS
execute_action(args)
# (don't add ToolMessage to history)
```
**Why bad**: Model generates action, never learns the result, can't adapt behavior.

---

## Summary

**Core principles**:
1. **Proper message roles** - system/user/assistant/tool, never custom types
2. **Standard formats** - LangChain messages, even when adapting for local models
3. **Single generation** - Thought+action together, not two phases
4. **Clean history** - Only valid interactions, no failed attempts
5. **Transparent adaptation** - Custom layers documented and explained

**What makes this different from naive approaches**:
- We use LangChain message types even though we're running local HF
- We parse model output into standard formats rather than using raw text
- We separate state (messages) from execution (model calls)
- We document every unusual decision

**Result**: An agent that's debuggable, maintainable, and follows best practices even when using local models that don't have native LangChain support.
