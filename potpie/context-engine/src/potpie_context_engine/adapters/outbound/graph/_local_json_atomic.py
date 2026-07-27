"""Process- and thread-safe critical sections for local JSON stores."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import threading

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows keeps thread-level safety.
    fcntl = None  # type: ignore[assignment]

_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    resolved = path.resolve(strict=False)
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(resolved, threading.RLock())


@contextmanager
def locked_json_store(path: Path) -> Iterator[None]:
    """Serialize read-modify-write operations for one JSON store path."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(lock_path):
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


__all__ = ["locked_json_store"]
