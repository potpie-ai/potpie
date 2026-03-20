"""potpie start – launch the local Potpie server."""

from __future__ import annotations

import json
import os
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


def start_server() -> None:
    """Start the Potpie server using the project's start script."""
    try:
        project_root = _find_project_root()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    start_script = project_root / "scripts" / "start.sh"

    if start_script.exists():
        if not start_script.is_file():
            print(f"Error: {start_script} is not a regular file.", file=sys.stderr)
            sys.exit(1)
        print("Starting Potpie server…")
        try:
            proc = subprocess.Popen(  # noqa: S603
                ["bash", str(start_script)],
                cwd=str(project_root),
            )
            PIDFILE.parent.mkdir(parents=True, exist_ok=True)
            PIDFILE.write_text(json.dumps({"shell": proc.pid}))
            ret = proc.wait()
            if ret != 0:
                print(f"Error: start script exited with code {ret}", file=sys.stderr)
                PIDFILE.unlink(missing_ok=True)
                sys.exit(ret)
        except OSError as exc:
            print(f"Error: failed to run start script: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        # Fallback: start gunicorn and celery directly
        _start_directly(project_root)


def _start_directly(project_root: Path) -> None:
    """Start gunicorn and celery directly without the shell script."""
    env = {**os.environ, "isDevelopmentMode": os.getenv("isDevelopmentMode", "enabled")}

    bind_address = os.getenv("POTPIE_BIND_ADDRESS", "127.0.0.1:8001")
    gunicorn_cmd = [
        sys.executable, "-m", "gunicorn",
        "--worker-class", "uvicorn.workers.UvicornWorker",
        "--workers", "1",
        "--timeout", "1800",
        "--bind", bind_address,
        "--log-level", "info",
        "app.main:app",
    ]
    celery_queue = os.getenv("CELERY_QUEUE_NAME", "staging")
    celery_cmd = [
        sys.executable, "-m", "celery",
        "-A", "app.celery.celery_app", "worker",
        "--loglevel=info",
        "-Q", (
            f"{celery_queue}_process_repository,"
            f"{celery_queue}_agent_tasks,"
            f"{celery_queue}_external-event"
        ),
        "-E", "--concurrency=1", "--pool=solo",
    ]

    print("Starting Potpie server (gunicorn)…")
    gunicorn_proc = subprocess.Popen(gunicorn_cmd, cwd=str(project_root), env=env)  # noqa: S603

    print("Starting Celery worker…")
    try:
        celery_proc = subprocess.Popen(celery_cmd, cwd=str(project_root), env=env)  # noqa: S603
    except Exception as exc:
        print(f"Error starting Celery worker: {exc}", file=sys.stderr)
        gunicorn_proc.kill()
        sys.exit(1)

    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(
        json.dumps({"gunicorn": gunicorn_proc.pid, "celery": celery_proc.pid})
    )

    print("Potpie services started. Use 'potpie stop' to stop them.")
