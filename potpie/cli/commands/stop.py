"""potpie stop – shut down the local Potpie server."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

PIDFILE = Path.home() / ".potpie" / "potpie.pids"


def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml (project root)."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not find pyproject.toml — run potpie from inside the project directory."
    )


def stop_server() -> None:
    """Stop the Potpie server using the project's stop script."""
    try:
        project_root = _find_project_root()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
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


def _stop_directly() -> None:
    """Stop gunicorn and celery processes using saved PIDs."""
    print("Stopping Potpie services…")
    if not PIDFILE.exists():
        print("No PID file found — services may not be running.", file=sys.stderr)
        return

    try:
        pids = json.loads(PIDFILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error reading PID file: {exc}", file=sys.stderr)
        return

    for name, pid in pids.items():
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"  Stopped {name} (PID {pid})")
        except ProcessLookupError:
            print(f"  No {name} process found (PID {pid})")
        except OSError as exc:
            print(f"  Error stopping {name} (PID {pid}): {exc}", file=sys.stderr)

    PIDFILE.unlink(missing_ok=True)
    print("Done.")
