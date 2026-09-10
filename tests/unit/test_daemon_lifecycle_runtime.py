"""Unit coverage for detached-daemon lifecycle decisions."""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from potpie.daemon.discovery import credential_path, write_daemon_credential
from potpie.daemon.lifecycle import Daemon, DaemonStopError, _pid_alive
from potpie.runtime import RuntimeOwnershipLock
from potpie_context_engine import Success


def test_pid_alive_reaps_an_exited_child_before_signal_probe(monkeypatch) -> None:
    waited: list[tuple[int, int]] = []

    def waitpid(pid: int, options: int) -> tuple[int, int]:
        waited.append((pid, options))
        return pid, 0

    monkeypatch.setattr("potpie.daemon.lifecycle.os.waitpid", waitpid)
    monkeypatch.setattr(
        "potpie.daemon.lifecycle.os.kill",
        lambda *_args: pytest.fail("an exited child must not reach the signal probe"),
    )

    assert _pid_alive(4242) is False
    assert waited == [(4242, os.WNOHANG)]


def test_pid_alive_uses_signal_probe_for_a_non_child(monkeypatch) -> None:
    signaled: list[tuple[int, int]] = []

    def not_a_child(_pid: int, _options: int) -> tuple[int, int]:
        raise ChildProcessError

    monkeypatch.setattr("potpie.daemon.lifecycle.os.waitpid", not_a_child)
    monkeypatch.setattr(
        "potpie.daemon.lifecycle.os.kill",
        lambda pid, signal: signaled.append((pid, signal)),
    )

    assert _pid_alive(4242) is True
    assert signaled == [(4242, 0)]


def test_pid_alive_treats_permission_denied_signal_probe_as_alive(monkeypatch) -> None:
    def not_a_child(_pid: int, _options: int) -> tuple[int, int]:
        raise ChildProcessError

    def permission_denied(_pid: int, _signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr("potpie.daemon.lifecycle.os.waitpid", not_a_child)
    monkeypatch.setattr("potpie.daemon.lifecycle.os.kill", permission_denied)

    assert _pid_alive(4242) is True


def test_daemon_restart_refuses_when_running_backend_unknown(
    tmp_path: Path, monkeypatch
) -> None:
    daemon = Daemon(home=tmp_path, in_process=False)
    calls: list[str] = []
    monkeypatch.setattr(
        daemon,
        "status",
        lambda: {
            "up": True,
            "mode": "detached",
            "home": str(tmp_path),
            "identity": "ok",
        },
    )
    monkeypatch.setattr(daemon, "stop", lambda: calls.append("stop"))
    monkeypatch.setattr(daemon, "start", lambda **_kwargs: calls.append("start"))

    # A *typed* refusal, not a bare RuntimeError: the CLI renders this as a
    # daemon failure with a next action instead of "Unexpected internal error".
    with pytest.raises(DaemonStopError) as raised:
        daemon.restart()

    assert "cannot determine the running daemon" in str(raised.value)
    assert raised.value.error.code == "daemon_backend_undetermined"
    assert raised.value.error.recommended_next_action
    assert calls == []


def test_daemon_restart_replaces_a_daemon_it_cannot_authenticate(
    tmp_path: Path, monkeypatch
) -> None:
    """An upgrade leaves a running daemon whose record this build cannot read.

    Restart is the recovery the CLI recommends, so it must take that process
    down by signal rather than refusing for an unknown backend.
    """
    daemon = Daemon(home=tmp_path, in_process=False)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        daemon,
        "status",
        lambda: {
            "up": True,
            "mode": "detached",
            "home": str(tmp_path),
            "pid": 4242,
            "identity": "unauthenticated",
        },
    )
    monkeypatch.setattr(daemon, "stop", lambda **kwargs: calls.append(("stop", kwargs)))
    monkeypatch.setattr(
        daemon, "start", lambda **kwargs: calls.append(("start", kwargs)) or {}
    )

    daemon.restart()

    assert calls[0] == ("stop", {"force": True})
    assert calls[1][0] == "start"


def test_force_stop_refuses_a_recorded_pid_that_is_not_a_daemon(
    tmp_path: Path, monkeypatch
) -> None:
    """PIDs are reused; a forced takedown must verify identity first."""
    daemon = Daemon(home=tmp_path, in_process=False)
    monkeypatch.setattr(
        "potpie.daemon.lifecycle._process_command_line",
        lambda _pid: "/usr/bin/vim notes.txt",
    )
    killed: list[int] = []
    monkeypatch.setattr(
        "potpie.daemon.lifecycle.os.kill",
        lambda pid, _sig: killed.append(pid),
    )

    with pytest.raises(DaemonStopError) as raised:
        daemon._force_stop(4242)

    assert raised.value.error.code == "daemon_recorded_pid_not_a_daemon"
    assert killed == []


def test_stop_preserves_credential_while_replacement_boot_owns_runtime(
    tmp_path: Path,
) -> None:
    ownership = RuntimeOwnershipLock((tmp_path / "daemon.runtime.lock").resolve())
    assert isinstance(ownership.acquire(), Success)
    write_daemon_credential(tmp_path, "x" * 43)
    daemon = Daemon(home=tmp_path, in_process=False)

    try:
        with pytest.raises(DaemonStopError) as raised:
            daemon.stop()
    finally:
        ownership.release()

    assert raised.value.error.code == "daemon_cleanup_ownership_conflict"
    assert credential_path(tmp_path).exists()
