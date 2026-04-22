# OpenAI Playwright Computer Agent (REPL)

This script launches a Playwright Chromium browser once and keeps it open while you type tasks into a terminal REPL.

## File

- `openai_playwright_computer_agent_repl.py`

## Install

```bash
pip install -U openai playwright
playwright install chromium
```

Set your API key:

```bash
export OPENAI_API_KEY=YOUR_KEY
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

## Launch

```bash
python openai_playwright_computer_agent_repl.py \
  --start-url https://example.com
```

Optional flags:

```bash
--headless
--manual-approve
--max-steps 50
--timeout-ms 30000
--pause-after-action-ms 400
```

## Basic usage

After launch, type tasks at the prompt:

```text
agent> summarize this page
agent> click the first article and tell me what it says
agent> go back and open the second result
agent> quit
```

Each task starts a new model request, but the same browser session stays open. That means the agent can keep using the current page state from earlier tasks.

## Built-in REPL commands

```text
help                Show commands
reset               Go back to the original --start-url
goto <url>          Navigate manually
where               Show current page title and URL
screenshot          Save a screenshot in the current directory
quit / exit         End the session
```

## Example

```bash
python openai_playwright_computer_agent_repl.py \
  --start-url https://news.ycombinator.com \
  --manual-approve
```

Then:

```text
agent> open the top story in a new tab
agent> summarize the article
agent> where
agent> screenshot
agent> quit
```

## Notes

- `--manual-approve` is useful while testing because it lets you review each action batch before execution.
- This is a minimal local starter, not a production-safe browser agent.
- For high-impact actions like logins, purchases, or destructive changes, keep a human in the loop.
- The browser state persists across tasks, but model conversation history does not. That keeps the loop simpler and avoids unbounded context growth.

## Local HTML files (`file://`)

If you test against a local page like:

```bash
--start-url "file:///C:/.../index.html"
```

the browser may need local file access. The REPL launcher now enables that automatically for `file://` URLs, and you can also force it with:

```bash
--allow-local-files
```

For more reliable behavior, it is often better to serve the page over a local web server instead of `file://`.

Example:

```bash
python -m http.server 8000
```

Then open:

```bash
http://127.0.0.1:8000/index.html
```
