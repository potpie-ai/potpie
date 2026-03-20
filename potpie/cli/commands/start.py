"""potpie start – launch the local Potpie server."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml (project root)."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: cwd
    return Path.cwd()


def start_server() -> None:
    """Start the Potpie server using the project's start script."""
    project_root = _find_project_root()
    start_script = project_root / "scripts" / "start.sh"

    if start_script.exists():
        if not start_script.is_file():
            print(f"Error: {start_script} is not a regular file.", file=sys.stderr)
            sys.exit(1)
        print("Starting Potpie server…")
        try:
            subprocess.run(  # noqa: S603 — trusted script from project tree
                ["bash", str(start_script)],
                cwd=str(project_root),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Error: start script exited with code {exc.returncode}", file=sys.stderr)
            sys.exit(exc.returncode)
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
    celery_queue = os.getenv("CELERY_QUEUE_NAME", "default")
    celery_cmd = [
        sys.executable, "-m", "celery",
        "-A", "app.celery.celery_app", "worker",
        "--loglevel=info",
        f"-Q", f"{celery_queue}_process_repository,{celery_queue}_agent_tasks",
        "-E", "--concurrency=1", "--pool=solo",
    ]

    print("Starting Potpie server (gunicorn)…")
    subprocess.Popen(gunicorn_cmd, cwd=str(project_root), env=env)  # noqa: S603 — fixed command, no user input

    print("Starting Celery worker…")
    subprocess.Popen(celery_cmd, cwd=str(project_root), env=env)  # noqa: S603 — fixed command, no user input

    print("Potpie services started. Use 'potpie stop' to stop them.")
