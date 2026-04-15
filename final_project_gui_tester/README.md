# GUI Tester V1

This project adds a minimal GUI testing agent on top of the existing Playwright + GPT custom computer-use demo without modifying any demo files.

## What It Does

- launches a browser against a provided URL
- runs a GUI testing agent with a testing-specific system prompt
- lets the agent save notes and optional screenshots during testing
- requires the agent to finish by submitting a final markdown report
- returns the full path to the main report file

## Main Entry Point

Run the GUI tester directly with:

```powershell
python final_project_gui_tester/run_gui_tester.py `
  --url "http://localhost:3000" `
  --gui-description "A small browser game with a start button, score display, and status messages." `
  --test-instructions "Check that the main controls are visible, verify the start button works, and report any obvious UI problems."
```

Required arguments:
- `--url`
- `--gui-description`
- `--test-instructions`

Optional arguments:
- `--report-dir`
- `--config`

## Output

Each run creates a run directory containing:
- `final_report.md`
- `notes/`
- `screenshots/`
- `tool_calls/`
- `gui_tester_run.log`

The final report links to the note files, and notes link to screenshots when attached.

## Testing A Broken GUI Directly

1. Make sure the GUI is already reachable by URL.
2. Run the CLI with the URL, a short GUI description, and focused testing instructions.
3. Wait for the agent to finish and print the report path.
4. Open `final_report.md` first, then inspect linked notes and screenshots.

Example with a local file URL:

```powershell
python final_project_gui_tester/run_gui_tester.py `
  --url "file:///C:/path/to/your/broken_gui/index.html" `
  --gui-description "A single-page web GUI for a small class project." `
  --test-instructions "Check whether labels are visible, buttons do the right thing, and any error or feedback messages appear when expected."
```

## Requirements

- Python environment with the same dependencies used by the demo
- Playwright Chromium installed
- `OPENAI_API_KEY` set if using the default config

## Notes

- V1 is focused on direct testing of the GUI tester itself.
- The parent-facing launcher function already exists in code for later integration, but this README documents direct CLI usage only.
- The tester always ends by submitting a final report. There is no separate terminate/fail tool path in this project.

