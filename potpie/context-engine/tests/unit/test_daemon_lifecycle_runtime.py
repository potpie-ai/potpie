"""Detached daemon lifecycle and daemon-backed setup smoke coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from potpie_context_engine.adapters.inbound.cli import host_cli as cli_main
from potpie_context_engine.domain.lifecycle import DONE, SKIPPED, SetupPlan
from potpie_context_engine.host.daemon import Daemon
from potpie_context_engine.host.daemon_client import DaemonRpcClient, RemoteHostShell


def test_daemon_ensure_starts_and_reuses(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    daemon = Daemon(home=tmp_path, in_process=False, startup_timeout_s=15)
    try:
        first = daemon.ensure(SetupPlan(backend="embedded"))
        second = daemon.ensure(SetupPlan(backend="embedded"))

        assert first.state == DONE
        assert second.state == SKIPPED
        status = daemon.status()
        assert status["up"] is True
        assert status["mode"] == "detached"
        assert status["backend"] == "embedded"
    finally:
        daemon.stop()


def test_daemon_restart_preserves_running_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "falkordb_lite")
    daemon = Daemon(home=tmp_path, in_process=False, startup_timeout_s=15)
    try:
        daemon.ensure(SetupPlan(backend="embedded"))
        restarted = daemon.restart()

        assert restarted["started"]["backend"] == "embedded"
        assert daemon.status()["backend"] == "embedded"
    finally:
        daemon.stop()


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


def test_remote_host_runs_setup_inside_daemon(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    daemon = Daemon(home=tmp_path, in_process=False, startup_timeout_s=15)
    try:
        daemon.ensure(SetupPlan(backend="embedded"))
        host = RemoteHostShell(rpc=DaemonRpcClient(daemon=daemon))

        report = host.setup.run(
            SetupPlan(
                host_mode="daemon",
                backend="embedded",
                repo="potpie",
                pot="default",
                agent="claude",
                assume_yes=True,
            )
        )

        assert report.ok
        assert any(
            step.step == "daemon" and step.state == SKIPPED for step in report.steps
        )
        active = host.pots.active_pot()
        assert active is not None
        assert active.name == "default"
    finally:
        daemon.stop()


def test_cli_setup_starts_detached_daemon(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "daemon")
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "embedded")
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app,
        [
            "--json",
            "setup",
            "--backend",
            "embedded",
            "--repo",
            "potpie",
            "--pot",
            "default",
            "--agent",
            "claude",
            "--yes",
        ],
    )
    try:
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["ok"] is True
        assert payload["plan"]["host_mode"] == "daemon"

        status_result = runner.invoke(cli_main.app, ["--json", "daemon", "status"])
        assert status_result.exit_code == 0, status_result.output
        status = json.loads(status_result.output)
        assert status["up"] is True
        assert status["mode"] == "detached"
    finally:
        runner.invoke(cli_main.app, ["--json", "daemon", "stop"])
