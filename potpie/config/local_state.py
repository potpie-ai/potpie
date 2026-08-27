"""Cross-thread and cross-process transactions for root local JSON state."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows uses msvcrt below.
    _fcntl = None  # type: ignore[assignment]

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX uses fcntl above.
    _msvcrt = None  # type: ignore[assignment]


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


@contextmanager
def local_json_transaction(
    path: Path,
    *,
    default_factory: Callable[[], dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Lock, freshly load, mutate, and atomically replace one JSON document."""

    resolved = path.resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved.with_suffix(resolved.suffix + ".lock")
    with _thread_lock(lock_path):
        with lock_path.open("a+b") as lock_file:
            _acquire_process_lock(lock_file)
            try:
                state = _load_json(resolved, default_factory=default_factory)
                yield state
                _atomic_write_json(resolved, state)
            finally:
                _release_process_lock(lock_file)


def _thread_lock(path: Path) -> threading.RLock:
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(path, threading.RLock())


def _load_json(
    path: Path, *, default_factory: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default_factory()
    return data if isinstance(data, dict) else default_factory()


def _atomic_write_json(path: Path, state: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(state, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _acquire_process_lock(lock_file: Any) -> None:
    if _fcntl is not None:
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX)
        return
    if _msvcrt is not None:
        lock_file.seek(0)
        if not lock_file.read(1):
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_LOCK, 1)


def _release_process_lock(lock_file: Any) -> None:
    if _fcntl is not None:
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
        return
    if _msvcrt is not None:
        lock_file.seek(0)
        _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)


__all__ = ["local_json_transaction"]
