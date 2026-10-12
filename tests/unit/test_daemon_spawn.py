"""How the controller creates the daemon child on POSIX and on Windows."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from potpie.daemon import lifecycle
from potpie.daemon.lifecycle import Daemon
from potpie.runtime import controller
from potpie.runtime.controller import (
    WINDOWS_DAEMON_CREATIONFLAGS,
    DaemonLaunchSpec,
    spawn_daemon_process,
)

_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_CREATE_NO_WINDOW = 0x08000000
_DETACHED_PROCESS = 0x00000008


class _Recorder:
    def __init__(self, failures: list[BaseException] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._failures = list(failures or [])

    async def __call__(self, *command: str, **kwargs: Any) -> object:
        self.calls.append({"command": command, **kwargs})
        if self._failures:
            raise self._failures.pop(0)
        return object()


def _access_denied() -> OSError:
    error = OSError("access denied")
    error.winerror = 5  # type: ignore[attr-defined]
    return error


def _launch(flags: int = WINDOWS_DAEMON_CREATIONFLAGS) -> DaemonLaunchSpec:
    return DaemonLaunchSpec(
        command=(sys.executable, "-m", "potpie.daemon"), creationflags=flags
    )


def test_windows_daemon_flags_are_windowless_grouped_and_break_away() -> None:
    assert WINDOWS_DAEMON_CREATIONFLAGS == (
        _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP | _CREATE_BREAKAWAY_FROM_JOB
    )
    # Windows ignores CREATE_NO_WINDOW when DETACHED_PROCESS is also set.
    assert not WINDOWS_DAEMON_CREATIONFLAGS & _DETACHED_PROCESS


async def test_posix_child_starts_a_new_session_without_creation_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(controller, "_windows", lambda: False)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)

    await spawn_daemon_process(_launch(), environment={"A": "1"}, stdout=None)

    (call,) = recorder.calls
    assert call["start_new_session"] is True
    assert "creationflags" not in call
    assert call["stdin"] == asyncio.subprocess.DEVNULL
    assert call["env"] == {"A": "1"}


async def test_windows_child_uses_creation_flags_instead_of_a_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(controller, "_windows", lambda: True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)

    await spawn_daemon_process(_launch(), environment={}, stdout=None)

    (call,) = recorder.calls
    assert call["creationflags"] == WINDOWS_DAEMON_CREATIONFLAGS
    assert "start_new_session" not in call


async def test_windows_job_that_forbids_breakaway_falls_back_to_staying_in_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder(failures=[_access_denied()])
    monkeypatch.setattr(controller, "_windows", lambda: True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)

    await spawn_daemon_process(_launch(), environment={}, stdout=None)

    assert [call["creationflags"] for call in recorder.calls] == [
        WINDOWS_DAEMON_CREATIONFLAGS,
        _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP,
    ]


@pytest.mark.parametrize(
    ("flags", "error"),
    [
        # Access denied without breakaway has nothing left to drop.
        (_CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP, _access_denied()),
        # Any other failure is not about the job object.
        (WINDOWS_DAEMON_CREATIONFLAGS, FileNotFoundError("no python")),
    ],
)
async def test_windows_spawn_failures_other_than_breakaway_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, flags: int, error: OSError
) -> None:
    recorder = _Recorder(failures=[error])
    monkeypatch.setattr(controller, "_windows", lambda: True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)

    with pytest.raises(OSError):
        await spawn_daemon_process(_launch(flags), environment={}, stdout=None)

    assert len(recorder.calls) == 1


async def test_a_real_child_starts_detached_on_this_platform() -> None:
    """Runs the real flags: on a Windows runner this exercises CreateProcess."""
    flags = WINDOWS_DAEMON_CREATIONFLAGS if sys.platform == "win32" else 0
    launch = DaemonLaunchSpec(
        command=(sys.executable, "-c", "raise SystemExit(7)"), creationflags=flags
    )

    process = await spawn_daemon_process(
        launch, environment=dict(os.environ), stdout=None
    )

    assert await asyncio.wait_for(process.wait(), timeout=30) == 7


def test_negative_creation_flags_are_rejected() -> None:
    with pytest.raises(ValueError, match="creation flags"):
        DaemonLaunchSpec(command=(sys.executable,), creationflags=-1)


def test_lifecycle_launch_spec_carries_windows_flags_only_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    daemon = Daemon(home=tmp_path, in_process=False)
    boot = daemon._boot_spec(backend=None)
    try:
        expected = WINDOWS_DAEMON_CREATIONFLAGS if sys.platform == "win32" else 0
        assert boot.launch.creationflags == expected
    finally:
        daemon._run(boot.observer.close())

    # Scoped tightly: a patched ``os.name`` also changes how ``pathlib`` builds
    # paths, so nothing else may run while it is in place.
    with monkeypatch.context() as patch:
        patch.setattr(lifecycle.os, "name", "nt")
        windows_flags = lifecycle._daemon_creationflags()
    with monkeypatch.context() as patch:
        patch.setattr(lifecycle.os, "name", "posix")
        posix_flags = lifecycle._daemon_creationflags()

    assert windows_flags == WINDOWS_DAEMON_CREATIONFLAGS
    assert posix_flags == 0
