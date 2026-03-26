#!/usr/bin/env python3
"""
Persistent Playwright + OpenAI Computer Use REPL for gpt-5.4.

What it does:
- launches a local Chromium browser with Playwright
- opens a user-specified starting page
- keeps the browser session alive across multiple tasks
- lets you type new tasks in a terminal REPL
- executes returned computer actions in the browser
- captures screenshots and continues until the model stops calling the tool

Example:
    python openai_playwright_computer_agent_repl.py \
        --start-url https://example.com

Then type tasks like:
    agent> summarize this page
    agent> click the login button
    agent> reset
    agent> quit

Install:
    pip install -U openai playwright
    playwright install chromium

Environment:
    export OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import base64
import sys
import textwrap
import time
from typing import Any, Iterable

from openai import OpenAI
from playwright.sync_api import Page, sync_playwright


DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
REPL_HELP = """\
Commands:
  <text>              Run a new task against the current browser state
  help                Show this help text
  reset               Navigate the current page back to --start-url
  goto <url>          Navigate to a URL manually
  screenshot          Save a screenshot to the current directory
  where               Print the current page URL and title
  quit / exit         End the session
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gpt-5.4 as a persistent computer-use agent against a Playwright browser."
    )
    parser.add_argument(
        "--start-url",
        required=True,
        help="Initial page to open when the browser launches.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="OpenAI model to use. Default: gpt-5.4",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headless. Default is headed for easier debugging.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum number of computer-call turns per task. Default: 50",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Playwright navigation timeout in milliseconds. Default: 30000",
    )
    parser.add_argument(
        "--pause-after-action-ms",
        type=int,
        default=400,
        help="Small delay after each executed action. Default: 400",
    )
    parser.add_argument(
        "--manual-approve",
        action="store_true",
        help="Prompt before each action batch is executed.",
    )
    parser.add_argument(
        "--screenshot-prefix",
        default="computer_agent",
        help="Prefix for screenshots saved by the 'screenshot' command. Default: computer_agent",
    )
    return parser.parse_args()


def response_items(response: Any) -> Iterable[Any]:
    return getattr(response, "output", []) or []


def first_computer_call(response: Any) -> Any | None:
    for item in response_items(response):
        if getattr(item, "type", None) == "computer_call":
            return item
    return None


def final_text_from_response(response: Any) -> str:
    parts: list[str] = []
    for item in response_items(response):
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) in {"output_text", "text"}:
                    text = getattr(content, "text", "")
                    if text:
                        parts.append(text)
        elif item_type in {"output_text", "text"}:
            text = getattr(item, "text", "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def capture_screenshot(page: Page) -> bytes:
    return page.screenshot(type="png", full_page=False)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def describe_action(action: Any) -> str:
    action_type = _get(action, "type", "unknown")
    if action_type in {"click", "double_click", "move"}:
        return f"{action_type}(x={_get(action, 'x')}, y={_get(action, 'y')})"
    if action_type == "scroll":
        return (
            f"scroll(x={_get(action, 'x')}, y={_get(action, 'y')}, "
            f"scrollX={_get(action, 'scrollX', 0)}, scrollY={_get(action, 'scrollY', 0)})"
        )
    if action_type == "keypress":
        return f"keypress(keys={_get(action, 'keys', [])})"
    if action_type == "type":
        text = _get(action, "text", "")
        preview = text if len(text) <= 60 else text[:57] + "..."
        return f"type(text={preview!r})"
    if action_type == "drag":
        path = _get(action, "path", [])
        return f"drag(path={path})"
    return action_type


def handle_computer_actions(page: Page, actions: Iterable[Any], pause_after_action_ms: int) -> None:
    for action in actions:
        action_type = _get(action, "type")

        if action_type == "click":
            page.mouse.click(
                _get(action, "x"),
                _get(action, "y"),
                button=_get(action, "button", "left"),
            )
        elif action_type == "double_click":
            page.mouse.dblclick(
                _get(action, "x"),
                _get(action, "y"),
                button=_get(action, "button", "left"),
            )
        elif action_type == "scroll":
            page.mouse.move(_get(action, "x"), _get(action, "y"))
            page.mouse.wheel(_get(action, "scrollX", 0), _get(action, "scrollY", 0))
        elif action_type == "keypress":
            for key in _get(action, "keys", []):
                page.keyboard.press(" " if key == "SPACE" else key)
        elif action_type == "type":
            page.keyboard.type(_get(action, "text", ""))
        elif action_type == "wait":
            time.sleep(2)
        elif action_type == "move":
            page.mouse.move(_get(action, "x"), _get(action, "y"))
        elif action_type == "drag":
            path = _get(action, "path", [])
            if len(path) < 2:
                raise ValueError(f"Drag action requires at least 2 path points, got: {path!r}")
            start = path[0]
            page.mouse.move(_get(start, "x"), _get(start, "y"))
            page.mouse.down()
            for point in path[1:]:
                page.mouse.move(_get(point, "x"), _get(point, "y"))
            page.mouse.up()
        elif action_type == "screenshot":
            pass
        else:
            raise ValueError(f"Unsupported action type: {action_type!r}")

        if pause_after_action_ms > 0:
            page.wait_for_timeout(pause_after_action_ms)


def send_initial_request(client: OpenAI, model: str, start_url: str, task: str) -> Any:
    prompt = (
        f"You are controlling a Playwright browser that was initially opened on: {start_url}\n"
        f"The browser is currently on this page: {task_current_url_placeholder()}\n"
        "Use the computer tool for UI interaction.\n"
        "Work only inside the browser unless the user explicitly asks otherwise.\n"
        "Treat the current browser state as authoritative.\n"
        "If the task appears blocked by login, CAPTCHA, payments, or another high-impact step, stop and explain.\n\n"
        f"Task: {task}"
    )
    return client.responses.create(
        model=model,
        tools=[{"type": "computer"}],
        input=prompt,
    )


def task_current_url_placeholder() -> str:
    return "<provided separately by the harness>"


def send_task_request(client: OpenAI, model: str, start_url: str, current_url: str, task: str) -> Any:
    prompt = (
        f"You are controlling a Playwright browser that was initially opened on: {start_url}\n"
        f"The browser is currently on this page: {current_url}\n"
        "Use the computer tool for UI interaction.\n"
        "Work only inside the browser unless the user explicitly asks otherwise.\n"
        "Treat the current browser state as authoritative.\n"
        "If the task appears blocked by login, CAPTCHA, payments, or another high-impact step, stop and explain.\n\n"
        f"Task: {task}"
    )
    return client.responses.create(
        model=model,
        tools=[{"type": "computer"}],
        input=prompt,
    )


def send_computer_screenshot(
    client: OpenAI,
    model: str,
    response: Any,
    call_id: str,
    screenshot_bytes: bytes,
) -> Any:
    screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
    return client.responses.create(
        model=model,
        tools=[{"type": "computer"}],
        previous_response_id=response.id,
        input=[
            {
                "type": "computer_call_output",
                "call_id": call_id,
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                    "detail": "original",
                },
            }
        ],
    )


def computer_use_loop(
    client: OpenAI,
    page: Page,
    initial_response: Any,
    model: str,
    max_steps: int,
    pause_after_action_ms: int,
    manual_approve: bool,
) -> Any:
    response = initial_response

    for step in range(1, max_steps + 1):
        computer_call = first_computer_call(response)
        if computer_call is None:
            return response

        actions = list(_get(computer_call, "actions", []))
        print(f"\n=== Step {step} ===")
        for idx, action in enumerate(actions, start=1):
            print(f"  {idx}. {describe_action(action)}")

        if manual_approve:
            approval = input("Execute this action batch? [y/N]: ").strip().lower()
            if approval not in {"y", "yes"}:
                raise RuntimeError("Stopped because the action batch was not approved.")

        handle_computer_actions(page, actions, pause_after_action_ms=pause_after_action_ms)

        screenshot = capture_screenshot(page)
        response = send_computer_screenshot(
            client=client,
            model=model,
            response=response,
            call_id=_get(computer_call, "call_id"),
            screenshot_bytes=screenshot,
        )

    raise RuntimeError(f"Stopped after reaching max_steps={max_steps}.")


def print_current_location(page: Page) -> None:
    title = page.title()
    url = page.url
    print("\nCurrent page")
    print(f"  Title: {title}")
    print(f"  URL:   {url}")


def save_screenshot(page: Page, prefix: str, counter: int) -> str:
    filename = f"{prefix}_{counter:03d}.png"
    page.screenshot(path=filename, type="png", full_page=False)
    return filename


def repl(client: OpenAI, page: Page, args: argparse.Namespace) -> None:
    screenshot_counter = 1

    print("\nBrowser session is ready.")
    print_current_location(page)
    print()
    print(textwrap.dedent(REPL_HELP).strip())

    while True:
        try:
            raw = input("\nagent> ").strip()
        except EOFError:
            print("\nEOF received. Exiting.")
            break

        if not raw:
            continue

        lowered = raw.lower()
        if lowered in {"quit", "exit"}:
            print("Exiting.")
            break
        if lowered == "help":
            print(textwrap.dedent(REPL_HELP).strip())
            continue
        if lowered == "reset":
            print(f"Navigating back to start URL: {args.start_url}")
            page.goto(args.start_url, wait_until="domcontentloaded")
            print_current_location(page)
            continue
        if lowered == "where":
            print_current_location(page)
            continue
        if lowered == "screenshot":
            filename = save_screenshot(page, args.screenshot_prefix, screenshot_counter)
            screenshot_counter += 1
            print(f"Saved screenshot: {filename}")
            continue
        if lowered.startswith("goto "):
            url = raw[5:].strip()
            if not url:
                print("Usage: goto <url>")
                continue
            print(f"Navigating to: {url}")
            page.goto(url, wait_until="domcontentloaded")
            print_current_location(page)
            continue

        try:
            initial_response = send_task_request(
                client=client,
                model=args.model,
                start_url=args.start_url,
                current_url=page.url,
                task=raw,
            )
            final_response = computer_use_loop(
                client=client,
                page=page,
                initial_response=initial_response,
                model=args.model,
                max_steps=args.max_steps,
                pause_after_action_ms=args.pause_after_action_ms,
                manual_approve=args.manual_approve,
            )
            final_text = final_text_from_response(final_response)
            print("\n=== Final response ===")
            if final_text:
                print(final_text)
            else:
                print("The model stopped calling the computer tool, but no final text was returned.")
        except Exception as exc:
            print(f"\nTask failed: {exc}")

        print_current_location(page)


def main() -> int:
    args = parse_args()
    client = OpenAI()

    with sync_playwright() as p:
        chromium_args = ["--disable-extensions"]
        if not (args.allow_local_files or args.start_url.startswith("file://")):
            chromium_args.append("--disable-file-system")
        else:
            print("Local file access enabled for this session.")

        browser = p.chromium.launch(
            headless=args.headless,
            chromium_sandbox=True,
            env={},
            args=chromium_args,
        )
        page = browser.new_page(viewport=DEFAULT_VIEWPORT)
        page.set_default_timeout(args.timeout_ms)

        try:
            print(f"Opening: {args.start_url}")
            page.goto(args.start_url, wait_until="domcontentloaded")
            repl(client=client, page=page, args=args)
        finally:
            browser.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
