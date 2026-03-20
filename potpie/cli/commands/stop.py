"""potpie stop – shut down the local Potpie server."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml (project root)."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def stop_server() -> None:
    """Stop the Potpie server using the project's stop script."""
    project_root = _find_project_root()
    stop_script = project_root / "scripts" / "stop.sh"

    if stop_script.exists():
        if not stop_script.is_file():
            print(f"Error: {stop_script} is not a regular file.", file=sys.stderr)
            sys.exit(1)
        print("Stopping Potpie server…")
        try:
            subprocess.run(  # noqa: S603 — trusted script from project tree
                ["bash", str(stop_script)],
                cwd=str(project_root),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Error: stop script exited with code {exc.returncode}", file=sys.stderr)
            sys.exit(exc.returncode)
    else:
        _stop_directly()


_ALLOWED_PROCESS_NAMES = frozenset({"gunicorn", "celery"})


def _stop_directly() -> None:
    """Stop gunicorn and celery processes directly."""
    print("Stopping Potpie services…")
    for process_name in _ALLOWED_PROCESS_NAMES:
        result = subprocess.run(  # noqa: S603 — hardcoded allowlist, no user input
            ["pkill", "-f", process_name],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"  Stopped {process_name}")
        else:
            print(f"  No {process_name} process found")
    print("Done.")
