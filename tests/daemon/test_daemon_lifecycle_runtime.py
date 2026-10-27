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
        lambda: {"up": True, "mode": "detached", "home": str(tmp_path)},
    )
    monkeypatch.setattr(daemon, "stop", lambda: calls.append("stop"))
    monkeypatch.setattr(daemon, "start", lambda **_kwargs: calls.append("start"))

    with pytest.raises(RuntimeError, match="cannot determine running daemon backend"):
        daemon.restart()

    assert calls == []


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
