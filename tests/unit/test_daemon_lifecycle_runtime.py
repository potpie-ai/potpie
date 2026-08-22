"""Unit coverage for detached-daemon lifecycle decisions."""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from potpie.daemon.lifecycle import Daemon, _pid_alive


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
