"""PID file, discovery file, and signal-based shutdown helpers."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import signal
from contextlib import contextmanager
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


class AlreadyRunning(RuntimeError):
    pass


from potpie.daemon.process.liveness import pid_alive as _pid_alive  # noqa: E402


@contextmanager
def _pid_file_lock(path: pathlib.Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.chmod(lock_path, 0o600)
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:  # pragma: no cover - unsupported Python platform
            raise RuntimeError("PID file locking is unavailable")
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt is not None:  # pragma: no cover - Windows
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)


def write_pid_file(path: pathlib.Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _pid_file_lock(path):
        while True:
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                existing = read_pid_file(path)
                if existing is not None and existing > 0 and _pid_alive(existing):
                    raise AlreadyRunning(
                        f"daemon already running (pid={existing})"
                    ) from None
                remove_pid_file(path)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{pid}\n")
            os.chmod(path, 0o600)
            return


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
    _write_private_text(path, json.dumps(fields))


def read_discovery(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _write_private_text(path: pathlib.Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)
    os.chmod(path, 0o600)


def install_signal_handlers(shutdown: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except (NotImplementedError, RuntimeError):
            # Windows/restricted: ignore; CLI 'stop' uses the PID kill path.
            pass
