"""The telemetry spool: what a CLI process writes instead of talking to the network.

A command runs for a few hundred milliseconds. Every telemetry sink it used
to talk to directly — Sentry for one metrics envelope, PostHog for one to
three events — costs a TLS handshake and a round trip on the command's own
wall time, paid after the answer was already printed: ~0.2 s for Sentry and
~0.9 s per PostHog batch, measured. So the process writes one JSON line per
metric or event to a spool file here and, at exit, starts a *detached* child
(``python -m potpie.cli.telemetry.flush``) that ships whatever is spooled.
Nothing waits on the child: a fork costs the parent a few milliseconds and the
child's network time belongs to nobody.

Rules that keep it honest:

- The spool never grows past :data:`SPOOL_MAX_BYTES`; offline, new records
  are dropped rather than accumulated. Telemetry is best-effort by design.
- One flusher at a time, via a lock file holding the flusher's pid. A lock
  whose process is gone (or that is older than five minutes) is stale and
  taken over. On Windows only the age rule applies: ``os.kill(pid, 0)`` there
  would *terminate* the process rather than probe it.
- A flusher never spawns a flusher (:data:`FLUSHER_ENV` marks the child).
- Everything here swallows ``OSError``: a read-only home or a full disk
  costs the telemetry, never the command.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from potpie_context_engine.adapters.outbound.cli_auth.credentials_store import (
    config_dir,
)

SPOOL_MAX_BYTES = 512 * 1024
FLUSHER_ENV = "POTPIE_TELEMETRY_FLUSHER"
LOCK_STALE_SECONDS = 300.0
#: Tests turn this off so an interpreter exit never launches a real flusher.
exit_spawn_enabled = True

_appended = False


def spool_dir() -> Path:
    return config_dir() / "telemetry"


def spool_path() -> Path:
    return spool_dir() / "spool.jsonl"


def lock_path() -> Path:
    return spool_dir() / "spool.lock"


def append(record: Mapping[str, Any]) -> bool:
    """Append one record; ``True`` when it was written.

    Appends of a few hundred bytes through ``O_APPEND`` are atomic on every
    platform this runs on, so concurrent commands never interleave lines.
    """
    global _appended
    path = spool_path()
    line = (
        json.dumps({"ts": time.time(), **record}, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.stat().st_size >= SPOOL_MAX_BYTES:
                return False
        except FileNotFoundError:
            pass
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        return False
    if not _appended:
        _appended = True
        atexit.register(_spawn_flusher_at_exit)
    return True


def pending_count() -> int:
    """How many records wait in the spool (0 when there is no spool)."""
    try:
        with spool_path().open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def take() -> list[dict[str, Any]]:
    """Move the spool aside and return its records; the moved file is removed.

    The rename is what makes this safe against commands still appending: they
    create a fresh spool, which the flusher's next round picks up.
    """
    path = spool_path()
    taken = path.with_name(f"spool.{os.getpid()}.{int(time.time() * 1000)}.jsonl")
    try:
        path.rename(taken)
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    try:
        for raw in taken.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append(record)
    except OSError:
        pass
    try:
        taken.unlink()
    except OSError:
        pass
    return records


def spawn_flusher() -> bool:
    """Start a detached flusher when the spool has content and none is running.

    Never raises: a failure to fork is a failure to ship telemetry, which the
    next command's exit will try again.
    """
    try:
        path = spool_path()
        if not path.exists() or path.stat().st_size == 0:
            return False
        if lock_held():
            return False
        _launch()
        return True
    except Exception:  # noqa: BLE001 - telemetry must never fail CLI work.
        return False


def _spawn_flusher_at_exit() -> None:
    if not exit_spawn_enabled or os.environ.get(FLUSHER_ENV):
        return
    spawn_flusher()


def launch_command() -> tuple[list[str], dict[str, Any]]:
    """The ``Popen`` argv and keyword arguments for a detached flusher."""
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "env": {**os.environ, FLUSHER_ENV: "1"},
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kwargs["start_new_session"] = True
    return [sys.executable, "-m", "potpie.cli.telemetry.flush"], kwargs


def _launch() -> None:
    argv, kwargs = launch_command()
    subprocess.Popen(argv, **kwargs)  # noqa: S603


# --- flusher lock -------------------------------------------------------------


def acquire_lock() -> bool:
    """Take the flusher lock; ``False`` when a live flusher holds it."""
    path = lock_path()
    for _ in range(2):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if lock_held():
                return False
            try:
                path.unlink()
            except OSError:
                return False
            continue
        except OSError:
            return False
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        return True
    return False


def release_lock() -> None:
    try:
        lock_path().unlink()
    except OSError:
        pass


def lock_held() -> bool:
    """Whether a live flusher holds the lock (a stale lock reads as free)."""
    path = lock_path()
    try:
        raw = path.read_text(encoding="ascii").strip()
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    if age > LOCK_STALE_SECONDS:
        return False
    try:
        pid = int(raw)
    except ValueError:
        return False
    return _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # No portable probe without terminating; the age rule bounds a stale
        # lock to LOCK_STALE_SECONDS on Windows.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


__all__ = [
    "FLUSHER_ENV",
    "LOCK_STALE_SECONDS",
    "SPOOL_MAX_BYTES",
    "acquire_lock",
    "append",
    "launch_command",
    "lock_held",
    "lock_path",
    "pending_count",
    "release_lock",
    "spawn_flusher",
    "spool_dir",
    "spool_path",
    "take",
]
