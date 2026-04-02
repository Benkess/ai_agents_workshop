"""
Playwright-based computer use environment.

Manages the full lifecycle of a Chromium browser instance via Playwright,
wiring it to the appropriate computer use tool for the chosen model variant.
"""

import os
import sys
from typing import Optional, Tuple

# Resolve tools/ imports regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from comp_use_env import ComputerUseEnv, LangChainToolLike


class PlaywrightComputerUseEnv(ComputerUseEnv):
    """
    Computer use environment backed by a Playwright Chromium browser.

    Supports two model variants:
    - "gpt": raw viewport pixel coordinates (tool_playwright_gpt)
    - "qwen": Qwen3-VL normalized 0–1000 coordinates (tool_playwright_qwen)

    Lifecycle::

        env = PlaywrightComputerUseEnv(model_variant="gpt")
        env.start_env()
        tool = env.get_computer_use_tool()
        # ... run agent ...
        env.stop_env()
    """

    def __init__(
        self,
        model_variant: str = "gpt",
        headless: bool = False,
        viewport_width: Optional[int] = 1280,
        viewport_height: Optional[int] = 720,
        start_url: Optional[str] = None,
        allow_local_files: bool = False,
        allow_extensions: bool = False,
        show_cursor_overlay: bool = True,
    ) -> None:
        """
        Args:
            model_variant: Which tool coordinate system to use. "gpt" for raw pixels,
                           "qwen" for Qwen3-VL normalized 0–1000 space.
            headless: Run the browser without a visible window.
            viewport_width: Browser viewport width in pixels. Defaults to 1280.
            viewport_height: Browser viewport height in pixels. Defaults to 720.
            start_url: Optional URL to navigate to immediately after launch.
                       Accepts http://, https://, and file:// schemes.
            allow_local_files: Allow Chromium access to local files. Also
                               auto-enabled when start_url begins with file://.
            allow_extensions: Allow Chromium browser extensions to run.
            show_cursor_overlay: Show a cursor dot overlay in the browser for
                                 debugging. Disable to avoid obstructing the
                                 model's view.
        """
        self._model_variant = model_variant
        self._headless = headless
        self._viewport_width = viewport_width or 1280
        self._viewport_height = viewport_height or 720
        self._start_url = start_url
        self._allow_local_files = allow_local_files
        self._allow_extensions = allow_extensions
        self._show_cursor_overlay = show_cursor_overlay

        self._playwright = None
        self._browser = None
        self._page = None

    def start_env(self) -> None:
        """
        Launch the Playwright Chromium browser and open a new page.

        Navigates to start_url if one was provided at construction time.
        """
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        launch_args = []
        if not self._allow_extensions:
            launch_args.append("--disable-extensions")
        if not self._allow_local_files and not (
            self._start_url and self._start_url.startswith("file://")
        ):
            launch_args.append("--disable-file-system")
        launch_args.append(f"--window-size={self._viewport_width},{self._viewport_height}")

        self._browser = self._playwright.chromium.launch(
            headless=self._headless,
            chromium_sandbox=True,
            env={},
            args=launch_args,
        )

        self._page = self._browser.new_page(
            viewport={"width": self._viewport_width, "height": self._viewport_height}
        )
        
        if self._show_cursor_overlay:
            # Add a custom cursor for better visibility during demos.
            # The tool updates cursor position by calling window.__setCursor(x, y)
            # and window.__flashCursor() via page.evaluate() after each action, since
            # Playwright's CDP mouse events do not fire DOM mousemove events.
            self._page.add_init_script("""
                const cursor = document.createElement('div');
                cursor.style.cssText = `
                    position: fixed; pointer-events: none; z-index: 999999;
                    width: 8px; height: 8px; border-radius: 50%;
                    background: rgba(255, 0, 0, 0.9); border: 1px solid white;
                    transform: translate(-50%, -50%);
                    left: -100px; top: -100px;
                `;
                window.__setCursor = (x, y) => {
                    cursor.style.left = x + 'px';
                    cursor.style.top = y + 'px';
                };
                window.__flashCursor = () => {
                    cursor.style.background = 'rgba(0, 255, 0, 0.8)';
                    setTimeout(() => cursor.style.background = 'rgba(255, 0, 0, 0.5)', 300);
                };
                document.addEventListener('DOMContentLoaded', () => {
                    document.body.appendChild(cursor);
                    // Also track real mouse for manual inspection
                    document.addEventListener('mousemove', e => {
                        cursor.style.left = e.clientX + 'px';
                        cursor.style.top = e.clientY + 'px';
                    });
                });
            """)
        if self._start_url:
            self._page.goto(self._start_url)

    def stop_env(self) -> None:
        """
        Close the browser page, browser, and Playwright instance.

        Safe to call even if start_env() was never called or raised an error.
        """
        try:
            if self._page:
                self._page.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        finally:
            self._page = None
            self._browser = None
            self._playwright = None

    def get_computer_use_tool(self) -> LangChainToolLike:
        """
        Return a computer use tool bound to the current browser page.

        Each call returns a fresh tool instance wrapping the live page object.

        Raises:
            RuntimeError: If start_env() has not been called.
            ValueError: If model_variant is not "gpt" or "qwen".
        """
        if self._page is None:
            raise RuntimeError(
                "Environment is not started. Call start_env() before get_computer_use_tool()."
            )

        if self._model_variant == "gpt":
            from tool_playwright_gpt import build_tool
        elif self._model_variant == "qwen":
            from tool_playwright_qwen import build_tool
        else:
            raise ValueError(
                f"Unknown model_variant: {self._model_variant!r}. Expected 'gpt' or 'qwen'."
            )

        return build_tool(self._page)

    def capture_screenshot(self) -> Tuple[bytes, str]:
        """
        Capture a PNG screenshot of the current browser viewport.

        Returns:
            A tuple of (png_bytes, "image/png").

        Raises:
            RuntimeError: If start_env() has not been called.
        """
        if self._page is None:
            raise RuntimeError(
                "Environment is not started. Call start_env() before capture_screenshot()."
            )
        return (self._page.screenshot(type="png"), "image/png")
