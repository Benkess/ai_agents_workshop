"""HTTP client for the observation API."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import requests


def _result_template() -> dict:
    return {
        "success": False,
        "value": None,
        "failure_mode": None,
        "reason": None,
        "busy": False,
        "error": None,
    }


def observe(image_path: str, prompt: str, host: str = "http://localhost:5000") -> dict:
    """
    Send an image file and prompt to the obs API server.
    Returns a dict with success, observation fields, busy, and error status.
    """
    try:
        result = _result_template()
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        mime_type = mime_type or "application/octet-stream"
        image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")

        response = requests.post(
            f"{host.rstrip('/')}/observe",
            json={
                "image_b64": image_b64,
                "mime_type": mime_type,
                "prompt": prompt,
            },
            timeout=60.0,
        )

        if response.status_code == 503:
            result["busy"] = True
            result["error"] = "Server busy"
            return result

        if response.status_code != 200:
            try:
                data = response.json()
                error = str(data.get("error", f"Server returned {response.status_code}"))
            except Exception:
                error = f"Server returned {response.status_code}"
            result["error"] = error
            return result

        data = response.json()
        result.update(data)
        result["busy"] = False
        return result
    except requests.Timeout:
        result = _result_template()
        result["error"] = "Request timeout"
        return result
    except requests.ConnectionError:
        result = _result_template()
        result["error"] = f"Could not connect to server at {host.rstrip('/')}"
        return result
    except Exception as exc:
        result = _result_template()
        result["error"] = str(exc)
        return result
