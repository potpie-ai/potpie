"""potpie stop – shut down the local Potpie server."""

from __future__ import annotations

import shutil
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


def _get_bash() -> str:
    """Return the absolute path to bash, or exit if not found."""
    bash = shutil.which("bash")
    if not bash:
        print("Error: bash not found on PATH.", file=sys.stderr)
        sys.exit(1)
    return bash


def stop_server() -> None:
    """Stop the Potpie server using the project's stop script."""
    try:
        project_root = _find_project_root()
    except FileNotFoundError:
        _stop_directly()
        return
    stop_script = project_root / "scripts" / "stop.sh"

    if stop_script.exists():
        if not stop_script.is_file():
            print(f"Error: {stop_script} is not a regular file.", file=sys.stderr)
            sys.exit(1)
        try:
            stop_script.resolve().relative_to(project_root.resolve())
        except ValueError:
            print("Error: stop script is outside the project root.", file=sys.stderr)
            sys.exit(1)
        print("Stopping Potpie server…")
        bash = _get_bash()
        try:
            subprocess.run(  # noqa: S603 # NOSONAR
                [bash, str(stop_script)],
                cwd=str(project_root),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Error: stop script exited with code {exc.returncode}", file=sys.stderr)
            sys.exit(exc.returncode)
    else:
        _stop_directly()


def _stop_directly() -> None:
    """Stop gunicorn and celery processes using pkill."""
    print("Stopping Potpie services…")
    pkill = shutil.which("pkill") or "pkill"
    for service in ("gunicorn", "celery"):
        try:
            subprocess.run(  # noqa: S603 # NOSONAR
                [pkill, "-f", service],
                check=False,
            )
            print(f"  Sent SIGTERM to {service} processes.")
        except OSError as exc:
            print(f"  Error stopping {service}: {exc}", file=sys.stderr)
    PIDFILE.unlink(missing_ok=True)
    print("Done.")
