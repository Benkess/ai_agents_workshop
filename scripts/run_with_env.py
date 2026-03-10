from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print(
        "Error: python-dotenv is not installed.\n"
        "Install it with: pip install python-dotenv",
        file=sys.stderr,
    )
    sys.exit(1)


def find_repo_root(start: Path) -> Path:
    """Walk upward to find the repo root, using .git or .venv as a hint."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / ".venv").exists():
            return candidate
    return start.resolve()


def get_venv_python(repo_root: Path) -> Path | None:
    """Return the local venv interpreter if it exists."""
    if os.name == "nt":
        py = repo_root / ".venv" / "Scripts" / "python.exe"
    else:
        py = repo_root / ".venv" / "bin" / "python"

    return py if py.exists() else None


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)

    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python scripts/run_with_env.py path/to/script.py [args ...]",
            file=sys.stderr,
        )
        return 2

    target = sys.argv[1]
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = (repo_root / target_path).resolve()

    if not target_path.exists():
        print(f"Error: target script not found: {target_path}", file=sys.stderr)
        return 1

    python_exe = get_venv_python(repo_root)
    if python_exe is None:
        python_exe = Path(sys.executable)

    cmd = [str(python_exe), str(target_path), *sys.argv[2:]]

    print(f"Using Python: {python_exe}")
    if env_file.exists():
        print(f"Loaded env file: {env_file}")
    else:
        print("No .env file found; continuing without it.")

    completed = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=os.environ.copy(),
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())    # Return the exit code from the script execution