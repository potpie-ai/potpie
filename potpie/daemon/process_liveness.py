"""Whether a process id names a live process, on every platform the CLI runs on.

``os.kill(pid, 0)`` is a liveness probe only on POSIX. On Windows signal ``0``
is ``CTRL_C_EVENT``, so ``os.kill(pid, 0)`` becomes
``GenerateConsoleCtrlEvent``: it interrupts the console process group ``pid``
when the caller shares a console and fails when it does not. Neither outcome
says anything about whether ``pid`` is alive. ``os.WNOHANG`` does not exist
there either. Windows therefore asks the kernel through ``OpenProcess`` and
``GetExitCodeProcess``. POSIX keeps the reap-then-signal-zero probe.

``ctypes`` keeps this free of a new dependency: ``psutil`` reaches the lock
only through ``falkordblite``, which is not installed on Windows.

Liveness is diagnostic metadata. A PID is never daemon instance identity
(DAEMON-012, DAEMON-046); this only decides whether runtime records look stale
and whether an authenticated handshake is worth attempting.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Final

#: ``GetExitCodeProcess`` reports this exit code while the process runs.
_STILL_ACTIVE: Final[int] = 259

#: The narrowest access right that permits ``GetExitCodeProcess``. Windows
#: grants it for most processes of other users and for elevated processes.
_PROCESS_QUERY_LIMITED_INFORMATION: Final[int] = 0x1000

#: ``OpenProcess`` fails with this when the process exists but this user may
#: not open it. Every other failure means there is no such process.
_ERROR_ACCESS_DENIED: Final[int] = 5


def pid_alive(pid: int) -> bool:
    """Return whether ``pid`` names a live process. An exited child is not live."""

    if pid <= 0:
        return False
    if _windows():
        return _pid_alive_windows(pid)
    return _pid_alive_posix(pid)


def _windows() -> bool:
    """Read at call time so tests can take the Windows branch on any host.

    Patching ``os.name`` instead would also change how ``pathlib`` builds
    paths for the rest of the test.
    """

    return os.name == "nt"


def _pid_alive_posix(pid: int) -> bool:
    try:
        waited_pid, _status = os.waitpid(pid, os.WNOHANG)
    except OSError:
        # A daemon launched by another CLI process is not our child. Keep the
        # signal probe for that normal cross-process observation path.
        pass
    else:
        # ``kill(pid, 0)`` still succeeds for an exited child that is a zombie.
        # Reap that child before deciding whether its PID is live.
        if waited_pid == pid:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM means the process exists but this user cannot signal it. Treat
        # that as live so status/start cannot mistake a cross-user process for
        # a stale PID and attempt runtime-record cleanup.
        return True
    except OSError:
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Same answer as POSIX EPERM: the process exists, it is just not ours
        # to inspect, so its runtime records must not be treated as stale.
        return _last_error() == _ERROR_ACCESS_DENIED
    try:
        exit_code = wintypes.DWORD(0)
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        # An exited process whose object is still referenced (for example by
        # this CLI's own child handle) reports its real exit code here.
        return exit_code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


@functools.cache
def _kernel32() -> Any:
    """A private ``kernel32`` binding with explicit signatures.

    Private so the argument and return types set here do not leak into the
    shared ``ctypes.windll.kernel32`` that other libraries use. ``HANDLE`` must
    be declared: the default ``int`` return type truncates 64-bit handles.
    """

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _last_error() -> int:
    import ctypes

    return int(ctypes.get_last_error())  # type: ignore[attr-defined]


__all__ = ["pid_alive"]
