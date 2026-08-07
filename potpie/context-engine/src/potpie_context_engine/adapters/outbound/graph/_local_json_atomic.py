"""Process- and thread-safe critical sections for local JSON stores."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import threading

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


def _thread_lock(path: Path) -> threading.RLock:
    resolved = path.resolve(strict=False)
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(resolved, threading.RLock())


def _acquire_process_lock(lock_file) -> None:
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


def _release_process_lock(lock_file) -> None:
    if _fcntl is not None:
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
        return
    if _msvcrt is not None:
        lock_file.seek(0)
        _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)


@contextmanager
def locked_json_store(path: Path) -> Iterator[None]:
    """Serialize read-modify-write operations for one JSON store path."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(lock_path):
        with lock_path.open("a+b") as lock_file:
            _acquire_process_lock(lock_file)
            try:
                yield
            finally:
                _release_process_lock(lock_file)


__all__ = ["locked_json_store"]
