"""Does a pid name a live process? One answer, shared by everything that asks.

Three places used to ask with ``os.kill(pid, 0)``, and on Windows that is not
a liveness check at all. ``signal.CTRL_C_EVENT`` is ``0`` there, so
``os.kill(pid, 0)`` is ``GenerateConsoleCtrlEvent``: it *succeeds* for any pid
whenever the caller owns a console, and raises winerror 5/6/87 whenever it
does not -- neither of which says anything about the pid. The daemon's pid
file therefore read as "running" from a terminal and as "stale" from a
console-less extension host, for the same live daemon, and the stale reading
unlinked the pid file and started a second daemon on the same home.

psutil asks the kernel. It ships with the ``potpie[daemon]`` extra, which is
the only install that has a daemon to ask about; the fallback below keeps the
POSIX semantics for a bare client that somehow reaches this code.
"""

from __future__ import annotations

import os


def pid_alive(pid: int) -> bool:
    """True when ``pid`` is a live process. A zombie is not alive."""
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:  # pragma: no cover - the daemon extra carries psutil
        return _pid_alive_posix(pid)
    try:
        if not psutil.pid_exists(pid):
            return False
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.AccessDenied:
        # Exists, and belongs to someone else. That is still "alive" -- the
        # failure this exists to prevent is treating a live daemon as stale.
        return True
    except psutil.Error:
        return False


def _pid_alive_posix(pid: int) -> bool:
    """The classic probe, correct only where signal 0 means "may I signal you"."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists; not ours to signal
    except (OSError, SystemError):
        return False
    return True


__all__ = ["pid_alive"]
