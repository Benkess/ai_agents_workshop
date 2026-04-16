# GUI Tester V1

This project adds a minimal GUI testing agent on top of the existing Playwright + GPT custom computer-use demo without modifying any demo files. It can be used directly from the CLI or exposed to a coding agent through a local stdio MCP server.

## What It Does

- launches a browser against a provided URL
- runs a GUI testing agent with a testing-specific system prompt
- lets the agent save notes and optional screenshots during testing
- requires the agent to finish by submitting a final markdown report
- returns the full path to the main report file

## Direct Usage

Run the GUI tester directly with either command:

```powershell
python final_project_gui_tester/run_gui_tester.py `
  --url "http://localhost:3000" `
  --gui-description "A small browser game with a start button, score display, and status messages." `
  --test-instructions "Check that the main controls are visible, verify the start button works, and report any obvious UI problems." `
  --report-dir "C:/path/to/context/reports"
```

```powershell
python -m final_project_gui_tester `
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

Each run creates a run directory containing:
- `final_report.md`
- `notes/`
- `screenshots/`
- `tool_calls/`
- `gui_tester_run.log`

The final report links to the note files, and notes link to screenshots when attached.

Report location:
- `--report-dir` is required
- the caller chooses the parent directory where artifacts should be written
- the tool creates a timestamped subdirectory inside that parent directory for each run
- a repo-local ignored artifacts folder such as `context/reports` is a good default workflow

## Direct Testing Workflow

1. Make sure the GUI is already reachable by URL.
2. Run the CLI with the URL, a short GUI description, and focused testing instructions.
3. Wait for the agent to finish and print the report path.
4. Open `final_report.md` first, then inspect linked notes and screenshots.

Example with a local file URL:

```powershell
python final_project_gui_tester/run_gui_tester.py `
  --url "file:///C:/path/to/your/broken_gui/index.html" `
  --gui-description "A single-page web GUI for a small class project." `
  --test-instructions "Check whether labels are visible, buttons do the right thing, and any error or feedback messages appear when expected." `
  --report-dir "C:/path/to/context/reports"
```

## Requirements

- Python environment with the same dependencies used by the demo
- Playwright Chromium installed
- `OPENAI_API_KEY` set if using the default config
- `mcp` installed if you want to run the local MCP server

## Current Status

- The GUI tester works directly from the CLI.
- The project includes a local stdio MCP server for coding-agent integration.
- Claude Code has been tested successfully against the MCP tool.
- The tester always ends by submitting a final report. There is no separate terminate/fail tool path in this project.
- The current tester behavior has been validated across several broken and mostly-working personal-website GUI variants.

## MCP Server

This package includes a local stdio MCP server with:
- server name: `gui_tester`
- tool name: `launch_gui_tester`

Tool inputs:
- `url`
- `gui_description`
- `test_instructions`
- `report_dir`

Tool output:
- `report_path`

Run the MCP server locally from the repo root:

```powershell
python -m final_project_gui_tester.mcp
```

Or use the repo venv interpreter explicitly:

```powershell
<repo-root>\.venv\Scripts\python.exe -m final_project_gui_tester.mcp
```

Example stdio MCP client config:

```json
{
  "mcpServers": {
    "gui_tester": {
      "command": "<repo-root>\\.venv\\Scripts\\python.exe",
      "args": ["-m", "final_project_gui_tester.mcp"],
      "cwd": "<repo-root>"
    }
  }
}
```

## Claude Code Setup

The working Claude Code setup used the repo venv interpreter and the MCP `__main__.py` script directly.

From the repo root:

```powershell
claude mcp add --transport stdio --scope local gui_tester -- `
  "<repo-root>\.venv\Scripts\python.exe" `
  "<repo-root>\final_project_gui_tester\mcp\__main__.py"
```

Then verify:

```powershell
claude mcp list
claude mcp get gui_tester
```

Expected result:
- `gui_tester` shows `Connected`

Notes:
- If you change the MCP server code or config, start a fresh Claude session before retesting tool availability.
- In Claude, `/mcp` may not show much useful text even when the tool is available. The more reliable check is whether Claude can call `launch_gui_tester`.

Example prompt for Claude Code:

```text
Call the launch_gui_tester MCP tool with these arguments:

url = file:///C:/path/to/your/gui/index.html
gui_description = A template for a personal website. It includes a landing page, blog page, and resume page. The sidebar on the landing page contains links to other media accounts.
test_instructions = Check all three pages for functionality and visual layout correctness. Report any issues found including visual, layout and navigation issues. Pay attention to whether the site fits cleanly in the viewport and whether each page looks complete and usable.
report_dir = C:\path\to\your\context\reports
```

## MCP Testing

1. Start or register the MCP server.
2. Confirm your MCP client sees one tool named `launch_gui_tester`.
3. Call the tool with:
   - a working `url`
   - a short `gui_description`
   - focused `test_instructions`
   - `report_dir` set to a parent folder such as `C:\path\to\your\context\reports`
4. Wait for the tool to finish and return `report_path`.
5. Confirm a new `run_<timestamp>` directory was created under the supplied `report_dir`.
6. Open the returned `final_report.md` first, then inspect linked notes, screenshots, and the agent log if needed.

The direct CLI and package entrypoint still work exactly the same:
- `python final_project_gui_tester/run_gui_tester.py ...`
- `python -m final_project_gui_tester ...`
