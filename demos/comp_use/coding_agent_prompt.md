# Coding Agent Prompt: Computer Use Agent — Environment & Tool Implementation

## Context

You are working inside a Python project for a modular computer use agent. Read the referenced files before writing any code — they define the interfaces, style, and patterns you must follow.

---

## Read These Files First

Before writing anything, read the following files in order:

1. **Environment interface** — [`custom_agent/comp_use_env.py`](/demos/comp_use/custom_agent/comp_use_env.py)
   The abstract base class all environments must implement. Pay attention to the `LangChainToolLike` protocol and the lifecycle contract in the docstring.

2. **Agent implementation** — [`custom_agent/custom_comp_use_agent.py`](/demos/comp_use/custom_agent/custom_comp_use_agent.py)
   Shows how the agent calls `get_computer_use_tool()`, invokes tools, and checks `result.get("terminal", False)` to detect task completion. Your tools must return JSON strings with `"terminal": true` on `terminate`/`fail` or the agent loop will not exit.

3. **Launcher** — [`custom_agent/comp_agent_launch.py`](/demos/comp_use/custom_agent/comp_agent_launch.py)
   Shows exactly how env configs are loaded and how `PlaywrightComputerUseEnv` and `PyAutoGUIComputerUseEnv` are instantiated. Your class constructors must match the params in the config files.

4. **Existing PyAutoGUI tool** — [`tools/qwen_tool_computer_use.py`](/demos/comp_use/tools/qwen_tool_computer_use.py)
   The canonical example of tool style in this project. Match its: Pydantic schema pattern, `@tool` decorator usage, JSON string return format, `thought` field, and field description quality. The Playwright tools must follow this same style.

5. **Reference Playwright action handler** — [`tools/playwrite.py`](/demos/comp_use/tools/playwrite.py)
   Shows the exact action set and Playwright API calls used for GPT-5.4's native computer use. Use this as the baseline action set for `tool_playwright_gpt.py`.

6. **Existing agent configs** — [`custom_agent/config/agent/gpt_agent.json`](custom_agent/config/agent/gpt_agent.json) and [`custom_agent/config/agent/qwen_agent.json`](/demos/comp_use/custom_agent/config/agent/qwen_agent.json)
   Shows the config schema the agent expects. The environment configs you create must follow the same JSON structure (`"type"` + `"params"`).

---

## Project Structure

```
comp_use/
├── custom_agent/
│   │   comp_agent_launch.py        ← exists
│   │   comp_use_env.py             ← exists
│   │   custom_comp_use_agent.py    ← exists
│   │   playwright_env.py           ← CREATE
│   │   pyautogui_env.py            ← CREATE
│   │   README.md                   ← CREATE
│   │
│   └── config/
│       ├── agent/
│       │       gpt_agent.json      ← exists
│       │       qwen_agent.json     ← exists
│       │
│       └── environment/
│               playwright_gpt.json      ← CREATE
│               playwright_qwen.json     ← CREATE
│               pyautogui.json           ← CREATE
│
└── tools/
        tool_computer_use.py        ← exists
        tool_playwright_gpt.py      ← CREATE
        tool_playwright_qwen.py     ← CREATE
        playwrite.py                ← exists, reference only
```

---

## Files to Create

### 1. `tools/tool_playwright_gpt.py`

Export a factory function `build_tool(page) -> LangChainToolLike`. Do **not** use a module-level singleton — all state must be scoped to the closure over `page`.

- Tool name: `"computer_use"`
- Action set: `navigate`, `click`, `double_click`, `scroll`, `type`, `keypress`, `move`, `drag`, `screenshot`, `wait`, `terminate`, `fail` — matching [`tools/playwrite.py`](tools/playwrite.py)
- **Coordinate system: viewport pixel coordinates.** `x`/`y` are raw pixel values within the browser viewport. Do not normalize. Field descriptions must state this clearly so the model knows what values to output.
- `keypress` presses keys **sequentially**, not as a chord
- `screenshot` should call `page.screenshot(type="png")`, encode as base64, and return `"data:image/png;base64,<data>"` in the result field
- `terminate` and `fail` must return `"terminal": true` in the JSON
- Match the Pydantic schema style, `thought` field, and JSON return format from [`tools/qwen_tool_computer_use.py`](tools/qwen_tool_computer_use.py)

### 2. `tools/tool_playwright_qwen.py`

Same factory pattern as above: `build_tool(page) -> LangChainToolLike`.

- Tool name: `"computer_use"`
- Action set: everything in the GPT tool plus `right_click`, `fill`, `get_text`, `back`, `forward`
- **Coordinate system: Qwen3-VL normalized 0–1000.** `x`/`y` are in range 0–1000 where `[0,0]` is top-left and `[1000,1000]` is bottom-right. Convert to viewport pixels before executing:
  ```python
  viewport = page.viewport_size  # {"width": W, "height": H}
  abs_x = max(0, min(int(x / 1000 * viewport["width"]),  viewport["width"]  - 1))
  abs_y = max(0, min(int(y / 1000 * viewport["height"]), viewport["height"] - 1))
  ```
- Field descriptions for `x` and `y` must explicitly state: `"Normalized coordinate in range 0–1000. [0,0] is top-left, [1000,1000] is bottom-right. Do NOT use pixel values."` This is critical — the model reads these descriptions to know what range to output.
- Additional actions:
  - `right_click`: right-click at normalized coords
  - `fill`: `page.fill(selector, text)` — directly sets an input field value (faster than typing; good for forms). Requires `selector` (CSS selector) and `text` fields.
  - `get_text`: `page.inner_text(selector)` — returns element text in the result field. Requires `selector`.
  - `back`: `page.go_back()`
  - `forward`: `page.go_forward()`

### 3. `custom_agent/playwright_env.py`

Implement `PlaywrightComputerUseEnv(ComputerUseEnv)` per the interface in [`custom_agent/comp_use_env.py`](custom_agent/comp_use_env.py).

Constructor signature:
```python
def __init__(
    self,
    model_variant: str = "gpt",      # "gpt" or "qwen"
    headless: bool = False,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    start_url: str | None = None      # Optional URL to navigate to after launch
)
```

- `start_env()`: launch `sync_playwright`, launch Chromium with `chromium_sandbox=True` and `env={}`, create a page with the given viewport, navigate to `start_url` if provided
- `stop_env()`: close page, browser, and playwright instance in a try/finally block; set all to `None`
- `get_computer_use_tool()`: raise `RuntimeError` if `start_env()` hasn't been called; import `build_tool` from `tool_playwright_gpt` or `tool_playwright_qwen` based on `model_variant`; return `build_tool(self._page)`
- `capture_screenshot()`: raise `RuntimeError` if not started; return `(self._page.screenshot(type="png"), "image/png")`
- Add `sys.path.insert` at the top to resolve `tools/` imports, using the same relative path pattern as [`custom_agent/custom_comp_use_agent.py`](custom_agent/custom_comp_use_agent.py) line 17

### 4. `custom_agent/pyautogui_env.py`

Implement `PyAutoGUIComputerUseEnv(ComputerUseEnv)` per the same interface.

- `start_env()`: no-op (PyAutoGUI needs no setup); set `self._started = True`
- `stop_env()`: set `self._started = False`
- `get_computer_use_tool()`: raise `RuntimeError` if not started; import and return `computer_use` from [`tools/qwen_tool_computer_use.py`](tools/qwen_tool_computer_use.py)
- `capture_screenshot()`: use `PIL.ImageGrab.grab()`, convert to PNG bytes via `io.BytesIO`, return `(png_bytes, "image/png")`
- Same `sys.path.insert` pattern for `tools/` resolution

### 5. Environment config files

Follow the `"type"` + `"params"` structure from the existing agent configs in [`custom_agent/config/agent/`](custom_agent/config/agent/). Constructor params map directly to `"params"`.

**`custom_agent/config/environment/playwright_gpt.json`**:
- `type`: `"playwright"`, `model_variant`: `"gpt"`, `headless`: `false`, `viewport_width`: `1280`, `viewport_height`: `720`, `start_url`: `null`

**`custom_agent/config/environment/playwright_qwen.json`**:
- Same as above but `model_variant`: `"qwen"`

**`custom_agent/config/environment/pyautogui.json`**:
- `type`: `"pyautogui"`, `params`: `{}`

### 6. `custom_agent/README.md`

Write a minimal, practical README. Include these sections:

1. **Overview** — One paragraph describing the system as a modular computer use agent pairing any OpenAI-compatible model with a pluggable environment, configured through JSON files.

2. **Architecture** — Describe the three layers: Agent (LangGraph loop in `custom_comp_use_agent.py`), Environment (owns lifecycle, wires tool to agent), Tools (implement actions; Playwright tools use `build_tool(page)` factory pattern).

3. **Supported Configurations** — Table with columns: Model | Environment | Tool File | Coordinate System | Agent Config | Env Config. Rows: GPT-5.4 + Playwright, Qwen3-VL + Playwright, Qwen3-VL + PyAutoGUI.

4. **Prerequisites** — Python packages (`langchain`, `langchain-openai`, `langgraph`, `playwright`, `pyautogui`, `pillow`, `pydantic`), Playwright browser install (`playwright install chromium`), `OPENAI_API_KEY` env var for GPT, Ollama running with `qwen3-vl:4b` pulled for Qwen.

5. **How to Run** — Show the Python call to `launch_computer_use_agent()` with example paths for each config combo. Show how to set `start_url` in the env config for local HTML files, noting it accepts `http://`, `https://`, and `file://` paths.

6. **Configuring the Agent** — Brief note that `user_prompt` in the agent config is the task instruction. Point to `config/agent/` and `config/environment/` for all tunable parameters.

---

## Hard Requirements

- Do not modify any existing files
- Tool name must be `"computer_use"` in both Playwright tool files — the agent looks up tools by name
- All tool return values must be JSON strings (`json.dumps()`) — not dicts
- `terminate` and `fail` must include `"terminal": true` in the JSON — see how the agent checks this in [`custom_agent/custom_comp_use_agent.py`](custom_agent/custom_comp_use_agent.py)
- The PyAutoGUI tool to import in `pyautogui_env.py` is the `computer_use` function from [`tools/qwen_tool_computer_use.py`](tools/qwen_tool_computer_use.py)
- Do not use module-level singletons in Playwright tool files
- Type hints and docstrings on all classes and public methods
- Note: fix any spelling errors in dir and file names.