# gui_tester

`gui_tester` is a self-contained GUI testing agent package. It includes:
- a direct CLI for human use
- a local stdio MCP server for coding-agent use
- a bundled copy of the `comp_use` support code it depends on

This draft lives inside the workshop repo at `final_project/gui-tester`, but it is structured as if this directory were already its own standalone repo. The later repo split should be path-stable: the future `gui-tester` repo root is intended to be exactly the contents of this directory.

## Repository Layout

- `gui_tester/` contains the package code, prompts, config, wrapper, custom tools, CLI, and MCP server
- `comp_use/` contains the bundled computer-use support code copied from the workshop demo
- `pyproject.toml` defines the installable package and console entrypoints

The `comp_use` copy is intentionally kept as a sibling directory to `gui_tester` so the current conceptual split stays visible.

## Install

From this directory:

```powershell
pip install -e .
```

Optional local-dev convenience:

```powershell
pip install -e .[dev]
```

The optional `dev` extra installs `python-dotenv`, which the bundled computer-use agent can use as a local `.env` fallback.

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

One working Claude Code setup is to point directly at the package MCP module or installed entrypoint.

Recommended example using the installed entrypoint inside the repo venv:

```powershell
claude mcp add --transport stdio --scope local gui_tester -- `
  "<repo-root>\.venv\Scripts\gui-tester-mcp.exe"
```

Alternative module form:

```powershell
claude mcp add --transport stdio --scope local gui_tester -- `
  "<repo-root>\.venv\Scripts\python.exe" `
  "<repo-root>\gui_tester\mcp\__main__.py"
```

Then verify:

```powershell
claude mcp list
claude mcp get gui_tester
```

Expected result:
- `gui_tester` shows `Connected`

Notes:
- if you change MCP code or config, start a fresh Claude session before retesting tool availability
- `/mcp` may not show much useful text even when the tool is available, so the real check is whether Claude can call `launch_gui_tester`

Example prompt for Claude Code:

```text
Call the launch_gui_tester MCP tool with these arguments:

url = file:///C:/path/to/your/gui/index.html
gui_description = A template for a personal website. It includes a landing page, blog page, and resume page. The sidebar on the landing page contains links to other media accounts.
test_instructions = Check all three pages for functionality and visual layout correctness. Report any issues found including visual, layout and navigation issues. Pay attention to whether the site fits cleanly in the viewport and whether each page looks complete and usable.
report_dir = C:\path\to\your\context\reports
```

## Testing Checklist

1. Install the package in editable mode from this directory.
2. Run one direct CLI test against a known GUI.
3. Confirm the report lands under the supplied parent `report_dir`.
4. Start the MCP server and confirm a client sees `launch_gui_tester`.
5. Run one end-to-end MCP call and confirm it returns `report_path`.
