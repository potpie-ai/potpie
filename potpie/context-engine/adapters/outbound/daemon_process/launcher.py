"""Detached daemon launcher: start the daemon as a background process and block until ready, or stop it.

This is the reliable, transport-agnostic start/stop mechanism the ``host.daemon.Daemon``
seam drives. ``start_detached`` is QUIET — it returns ``{"pid","socket","bind"}`` once the
daemon has signalled readiness (discovery file written) or raises ``DaemonStartError``; it
never prints, so it is safe to call from inside a rich/live setup UI.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

from adapters.outbound.daemon_process.pidfile import read_pid_file


class DaemonStartError(Exception):
    """Raised by start_detached() when the daemon does not come up.

    Carries the daemon log path (when available) so callers can surface the cause.
    """

    def __init__(self, message: str, *, log_path: "pathlib.Path | None" = None) -> None:
        super().__init__(message)
        self.log_path = log_path


def start_detached(home: pathlib.Path, *, ready_timeout_s: float = 60.0) -> dict:
    """Start the daemon detached for ``home`` and block until it is fully serving.

    Returns ``{"pid", "socket", "bind"}`` once the daemon signals readiness (discovery file
    written), or raises :class:`DaemonStartError` on any failure (already running, child
    crash, or readiness timeout).
    """
    home = pathlib.Path(home)
    pid_file = home / "daemon.pid"
    disc_file = home / "discovery.json"
    log_path = home / "logs" / "potpied.log"
    if pid_file.exists():
        existing = read_pid_file(pid_file)
        if existing:
            try:
                os.kill(existing, 0)
                raise DaemonStartError(f"daemon already running (pid={existing})")
            except ProcessLookupError:
                pid_file.unlink()  # stale
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("a")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "host.daemon_runtime", "run", "--home", str(home)],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env={**os.environ, "CONTEXT_ENGINE_HOME": str(home)},
        )
    finally:
        log_fp.close()
    deadline = time.time() + ready_timeout_s
    while time.time() < deadline:
        if disc_file.exists():
            try:
                disc = json.loads(disc_file.read_text())
            except (OSError, json.JSONDecodeError):
                disc = {}
            bind = disc.get("bind", "")
            socket_path = bind[len("unix:") :] if bind.startswith("unix:") else bind
            return {"pid": proc.pid, "socket": socket_path, "bind": bind}
        if proc.poll() is not None:
            raise DaemonStartError(
                f"daemon failed to start (exit {proc.returncode})", log_path=log_path
            )
        time.sleep(0.1)
    # Alive but never signalled ready — stop it so we don't leave a half-up daemon.
    try:
        os.kill(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise DaemonStartError(
        f"daemon did not become ready within {int(ready_timeout_s)}s", log_path=log_path
    )


def stop_daemon(home: pathlib.Path) -> str:
    """Stop the daemon if running. Returns a human-readable message; never raises."""
    home = pathlib.Path(home)
    pid_file = home / "daemon.pid"
    pid = read_pid_file(pid_file)
    if not pid:
        return "daemon not running"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _unlink(pid_file)
        return "stale pid file removed"
    deadline = time.time() + 10
    while time.time() < deadline:
        if not pid_file.exists():
            return "daemon stopped"
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _unlink(pid_file)
    return "daemon killed (forced after timeout)"


def _unlink(path: pathlib.Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
