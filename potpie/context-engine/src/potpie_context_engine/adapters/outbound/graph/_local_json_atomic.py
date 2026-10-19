"""Process- and thread-safe critical sections for local JSON stores."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import json
import os
import tempfile
import threading
from typing import Any, TextIO

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


def strict_json_load(stream: TextIO) -> Any:
    """Decode persisted authority without accepting duplicate object keys."""

    def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate storage key: {key}")
            result[key] = value
        return result

    return json.load(stream, object_pairs_hook=unique_object)


def _thread_lock(path: Path) -> threading.RLock:
    resolved = path.resolve(strict=False)
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(resolved, threading.RLock())


def _acquire_process_lock(lock_file) -> None:
    if _fcntl is not None:
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX)
        return
    if _msvcrt is not None:
        # Windows permits locking beyond EOF. Reading or initializing this byte
        # first would fail while another process holds its mandatory range lock.
        lock_file.seek(0)
        _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_LOCK, 1)
        return
    raise RuntimeError("process locking is unsupported on this platform")


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


def durable_json_replace(path: Path, value: object) -> None:
    """Replace one JSON state after synchronizing its bytes and directory entry.

    Callers own the process lock. A failure after replacement is an unknown
    acknowledgement; the complete published file remains authoritative.
    """
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Windows does not expose directory fsync through os.open. The file's
        # contents are flushed above; replace is performed on the same volume.
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


__all__ = ["locked_json_store", "durable_json_replace", "strict_json_load"]
