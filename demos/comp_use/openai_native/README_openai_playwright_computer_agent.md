# OpenAI Playwright Computer Agent

Minimal Python starter for running `gpt-5.4` as a computer-use agent with a local Playwright browser.

## Requirements

- Python 3.10+
- An OpenAI API key
- Playwright Chromium installed

## Install

```bash
pip install -U openai playwright
playwright install chromium
```

## Set your API key

Linux/macOS:

```bash
export OPENAI_API_KEY=YOUR_KEY
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

## Run

```bash
python openai_playwright_computer_agent.py \
  --start-url https://example.com \
  --task "Find the page title and tell me what it is."
```

## Useful flags

```bash
--headless         # run browser without showing the window
--manual-approve   # ask before each action batch
--max-steps 50     # cap the number of agent steps
```

## Example

```bash
python openai_playwright_computer_agent.py \
  --start-url https://news.ycombinator.com \
  --task "Tell me the title of the first post."
```

## Notes

- The browser starts at the page you pass with `--start-url`.
- The task is given with `--task`.
- `--manual-approve` is a good idea while testing.
- Avoid using this on anything sensitive until you trust the setup.

## Local HTML files (`file://`)

If you test against a local page like:

```bash
--start-url "file:///C:/.../index.html"
```

the browser may need local file access. The launcher now enables that automatically for `file://` URLs, and you can also force it with:

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
