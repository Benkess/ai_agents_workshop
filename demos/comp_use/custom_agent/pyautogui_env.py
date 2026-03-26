"""
PyAutoGUI-based computer use environment.

Wraps the existing qwen_tool_computer_use tool to provide a ComputerUseEnv
implementation that operates directly against the host desktop via PyAutoGUI.
No browser or external process is needed — start_env() is a no-op.
"""

import io
import os
import sys
from typing import Tuple

# Resolve tools/ imports regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from comp_use_env import ComputerUseEnv, LangChainToolLike


class PyAutoGUIComputerUseEnv(ComputerUseEnv):
    """
    Computer use environment backed by PyAutoGUI for direct desktop control.

    Uses the Qwen3-VL normalized 0–1000 coordinate system via the
    qwen_tool_computer_use tool.

    Lifecycle::

        env = PyAutoGUIComputerUseEnv()
        env.start_env()
        tool = env.get_computer_use_tool()
        # ... run agent ...
        env.stop_env()
    """

    def __init__(self) -> None:
        self._started = False

    def start_env(self) -> None:
        """
        Mark the environment as started.

        PyAutoGUI operates directly against the running desktop and requires
        no setup beyond this flag.
        """
        self._started = True

    def stop_env(self) -> None:
        """Mark the environment as stopped."""
        self._started = False

    def get_computer_use_tool(self) -> LangChainToolLike:
        """
        Return the PyAutoGUI computer use tool.

        Returns:
            The module-level computer_use tool from qwen_tool_computer_use.

        Raises:
            RuntimeError: If start_env() has not been called.
        """
        if not self._started:
            raise RuntimeError(
                "Environment is not started. Call start_env() before get_computer_use_tool()."
            )
        from qwen_tool_computer_use import computer_use
        return computer_use

    def capture_screenshot(self) -> Tuple[bytes, str]:
        """
        Capture a PNG screenshot of the entire desktop.

        Returns:
            A tuple of (png_bytes, "image/png").

        Raises:
            RuntimeError: If start_env() has not been called.
        """
        if not self._started:
            raise RuntimeError(
                "Environment is not started. Call start_env() before capture_screenshot()."
            )
        from PIL import ImageGrab
        img = ImageGrab.grab()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return (buf.getvalue(), "image/png")
