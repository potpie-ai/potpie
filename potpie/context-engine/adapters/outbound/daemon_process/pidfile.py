"""PID file, discovery file, and signal-based shutdown helpers."""

from __future__ import annotations
import asyncio
import json
import os
import pathlib
import signal


class AlreadyRunning(RuntimeError):
    pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def write_pid_file(path: pathlib.Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = int(path.read_text().strip())
        except ValueError:
            existing = -1
        if existing > 0 and _pid_alive(existing):
            raise AlreadyRunning(f"daemon already running (pid={existing})")
    path.write_text(f"{pid}\n")


def read_pid_file(path: pathlib.Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def remove_pid_file(path: pathlib.Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_discovery(path: pathlib.Path, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))


def read_discovery(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def install_signal_handlers(shutdown: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except (NotImplementedError, RuntimeError):
            # Windows/restricted: ignore; CLI 'stop' uses the PID kill path.
            pass
