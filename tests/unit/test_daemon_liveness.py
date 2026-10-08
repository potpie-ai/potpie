"""Daemon liveness asks the kernel on every platform, never ``os.kill(pid, 0)`` on Windows."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from potpie.daemon import process_liveness
from potpie.daemon.discovery import read_daemon_pid, write_daemon_pid
from potpie.daemon.lifecycle import Daemon, _RecordedDaemonProcess
from potpie.daemon.process_liveness import pid_alive

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX reaping semantics"
)


def test_a_live_process_is_alive_and_an_exited_one_is_not() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert pid_alive(child.pid) is True
    finally:
        child.kill()
        child.wait(timeout=10)
    assert pid_alive(child.pid) is False


@_POSIX_ONLY
def test_an_exited_unreaped_child_is_not_alive() -> None:
    """A zombie still owns its pid and ``kill(pid, 0)`` would say yes."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        # Block until the child has exited but leave it unreaped (a zombie).
        # Waiting for stdout EOF is not enough: the child closes its pipes
        # before the kernel marks it exited, and on Linux that window is wide
        # enough for the liveness probe to see a still-running process.
        if hasattr(os, "waitid"):
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        else:  # os.waitid is missing on macOS before Python 3.13: poll ps instead.
            deadline = time.monotonic() + 10
            while (
                "Z"
                not in subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(child.pid)],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
            ):
                assert time.monotonic() < deadline, "child never became a zombie"
                time.sleep(0.01)
        assert pid_alive(child.pid) is False
    finally:
        child.wait(timeout=10)


def test_non_positive_pids_are_never_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        process_liveness.os,
        "kill",
        lambda *_args: pytest.fail("pid 0 and -1 address process groups"),
    )

    assert pid_alive(0) is False
    assert pid_alive(-1) is False


class _FakeKernel32:
    """``OpenProcess``/``GetExitCodeProcess``/``CloseHandle`` without Windows."""

    def __init__(
        self,
        *,
        handle: int = 0x1F4,
        exit_code: int = 259,
        query_ok: bool = True,
    ) -> None:
        self.handle = handle
        self.exit_code = exit_code
        self.query_ok = query_ok
        self.opened: list[tuple[int, bool, int]] = []
        self.closed: list[int] = []

    def OpenProcess(self, access: int, inherit: bool, pid: int) -> int:  # noqa: N802
        self.opened.append((access, inherit, pid))
        return self.handle

    def GetExitCodeProcess(self, handle: int, exit_code: Any) -> bool:  # noqa: N802
        assert handle == self.handle
        if not self.query_ok:
            return False
        exit_code._obj.value = self.exit_code
        return True

    def CloseHandle(self, handle: int) -> bool:  # noqa: N802
        self.closed.append(handle)
        return True


def _as_windows(
    monkeypatch: pytest.MonkeyPatch, kernel32: _FakeKernel32, *, last_error: int = 0
) -> None:
    monkeypatch.setattr(process_liveness, "_windows", lambda: True)
    monkeypatch.setattr(process_liveness, "_kernel32", lambda: kernel32)
    monkeypatch.setattr(process_liveness, "_last_error", lambda: last_error)
    monkeypatch.setattr(
        process_liveness.os,
        "kill",
        lambda *_args: pytest.fail("Windows signal 0 is a console Ctrl+C"),
    )
    monkeypatch.setattr(
        process_liveness.os,
        "waitpid",
        lambda *_args: pytest.fail("Windows has no WNOHANG reaping"),
        raising=False,
    )


def test_windows_running_process_is_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel32 = _FakeKernel32(exit_code=259)
    _as_windows(monkeypatch, kernel32)

    assert pid_alive(4242) is True
    assert kernel32.opened == [(0x1000, False, 4242)]
    assert kernel32.closed == [kernel32.handle]


def test_windows_exited_process_is_not_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel32 = _FakeKernel32(exit_code=0)
    _as_windows(monkeypatch, kernel32)

    assert pid_alive(4242) is False
    assert kernel32.closed == [kernel32.handle]


def test_windows_failed_exit_code_query_is_not_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = _FakeKernel32(query_ok=False)
    _as_windows(monkeypatch, kernel32)

    assert pid_alive(4242) is False
    assert kernel32.closed == [kernel32.handle]


def test_windows_missing_process_is_not_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel32 = _FakeKernel32(handle=0)
    _as_windows(monkeypatch, kernel32, last_error=87)  # ERROR_INVALID_PARAMETER

    assert pid_alive(4242) is False
    assert kernel32.closed == []


def test_windows_access_denied_process_is_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same answer as POSIX EPERM: it exists, so its records are not stale."""
    kernel32 = _FakeKernel32(handle=0)
    _as_windows(monkeypatch, kernel32, last_error=5)  # ERROR_ACCESS_DENIED

    assert pid_alive(4242) is True


def test_status_clears_records_of_a_daemon_whose_process_exited(
    tmp_path: Path,
) -> None:
    """The path that crashed on Windows: status over a stale recorded pid."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    write_daemon_pid(tmp_path, child.pid)

    status = Daemon(home=tmp_path, in_process=False).status()

    assert status["up"] is False
    assert status["pid"] is None
    assert read_daemon_pid(tmp_path) is None


def test_a_recorded_daemon_process_is_never_signalled() -> None:
    """Only a directly owned child may be terminated (DAEMON-052)."""
    recorded = _RecordedDaemonProcess(4242)

    with pytest.raises(PermissionError, match="refusing to signal"):
        recorded.terminate()
    with pytest.raises(PermissionError, match="refusing to signal"):
        recorded.kill()
