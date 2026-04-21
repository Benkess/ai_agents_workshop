# Dev Info

`gui_tester` is a self-contained GUI testing agent package. It includes:
- a direct CLI for human use
- a local stdio MCP server for coding-agent use
- a bundled copy of the `comp_use` support code it depends on

> **Note:** This draft lives inside the workshop repo at `final_project/gui-tester`, but it is structured as if this directory were already its own standalone repo. The later repo split should be path-stable: the future `gui-tester` repo root is intended to be exactly the contents of this directory.

## Repository Layout

- `gui_tester/` contains the package code, prompts, config, wrapper, custom tools, CLI, and MCP server
- `comp_use/` contains the bundled computer-use support code copied from the workshop demo
- `pyproject.toml` defines the installable package and console entrypoints

The `comp_use` copy is intentionally kept as a sibling directory to `gui_tester` so the current conceptual split stays visible.

## Package Setup

For MCP use, it is recommended to clone this repo seperatly from any projects. The coding agents will use an MCP config that points to this packages env and install.

### Enviroment Setup
> **Note:** if `python` does not default to `python3` on your system, then substitute `python3` for `python` in the following commands.

Navigate to the root directory:
```bash
cd path/to/gui-tester
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

### Package Install

From this directory:

```powershell
pip install -e .
```

Optional local-dev convenience:

```powershell
pip install -e .[dev]
```

The optional `dev` extra installs `python-dotenv`, which the bundled computer-use agent can use as a local `.env` fallback.

> **Note:** Currently normal install still requires python-dotenv, this will be removed in future updates.

## Direct CLI

After installation:

```powershell
gui-tester `
  --url "http://localhost:3000" `
  --gui-description "A small browser game with a start button, score display, and status messages." `
  --test-instructions "Check that the main controls are visible, verify the start button works, and report any obvious UI problems." `
  --report-dir "C:/path/to/context/reports"
```

Module form also works:

```powershell
python -m gui_tester `
  --url "http://localhost:3000" `
  --gui-description "A small browser game with a start button, score display, and status messages." `
  --test-instructions "Check that the main controls are visible, verify the start button works, and report any obvious UI problems." `
  --report-dir "C:/path/to/context/reports"
```

Required arguments:
- `--url`
- `--gui-description`
- `--test-instructions`
- `--report-dir`

Optional arguments:
- `--config`

## Output

Each run creates a timestamped `run_<timestamp>` directory inside the supplied `--report-dir` parent directory.

Run contents:
- `final_report.md`
- `notes/`
- `screenshots/`
- `tool_calls/`
- `gui_tester_run.log`

The final report links to notes, and notes link to screenshots when attached.

## Secrets and Config

Official package story:
- direct CLI usage should rely on normal environment variables
- MCP hosts should pass environment variables in their MCP config

Important:
- this package preserves the bundled computer-use agent's existing `api_key` and `api_key_env` behavior
- not all configs use `OPENAI_API_KEY`
- for example, some local-model configs use an explicit `api_key` value such as `ollama`

Local-dev convenience:
- if `python-dotenv` is installed, the bundled computer-use agent may still use its existing lower-level `.env` fallback
- this is preserved for compatibility, but it is not the primary package contract

Advanced config:
- the CLI supports `--config`
- the MCP tool schema stays minimal in this draft

## MCP Server

The package includes a local stdio MCP server with:
- server name: `gui_tester`
- tool name: `launch_gui_tester`

Tool inputs:
- `url`
- `gui_description`
- `test_instructions`
- `report_dir`

Tool output:
- `report_path`

Run the MCP server after installation:

```powershell
gui-tester-mcp
```

Module form also works:

```powershell
python -m gui_tester.mcp
```

## Claude Code Setup

There are many ways to setup the MCP depending on your use case. We give examples for local project setups. It is assumed that you have already setup the this package elsewhere.

Example using the installed entrypoint inside the repo venv:

**CLI only** (default local scope, goes into `~/.claude.json`):
```powershell
claude mcp add gui_tester --transport stdio --env YOUR_API_KEY=sk-your-key -- <path-to-venv>\Scripts\python.exe -m gui_tester.mcp
```

**CLI + VS Code extension** (project scope, creates `.mcp.json` in project root):
```powershell
claude mcp add gui_tester --scope project --transport stdio --env YOUR_API_KEY=sk-your-key -- <path-to-venv>\Scripts\python.exe -m gui_tester.mcp
```

Replace `YOUR_API_KEY` with whatever env var name your model config expects (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

Then verify:

In Claude Code use `/mcp` and ensure `gui_tester` shows `Connected`.

Notes:
- After you setup or change the MCP config, start a fresh Claude session and reconnect to the MCP before retesting tool availability.

Example prompt for Claude Code:

```text
Call the launch_gui_tester MCP tool with these arguments:

url = file:///C:/path/to/your/gui/index.html
gui_description = A template for a personal website. It includes a landing page, blog page, and resume page. The sidebar on the landing page contains links to other media accounts.
test_instructions = Check all three pages for functionality and visual layout correctness. Report any issues found including visual, layout and navigation issues. Pay attention to whether the site fits cleanly in the viewport and whether each page looks complete and usable.
report_dir = C:\path\to\your\context\reports
```
