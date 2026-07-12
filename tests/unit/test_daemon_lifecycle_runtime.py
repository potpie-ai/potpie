"""Detached daemon lifecycle and daemon-backed setup smoke coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.contracts import (
    EngineStatusRequest,
    PotCreateRequest,
)
from potpie.setup import DONE, SKIPPED, SetupPlan
from potpie.daemon.client import DaemonEngineClient, DaemonRpcTransport
from potpie.daemon.lifecycle import Daemon


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


@pytest.mark.asyncio
async def test_typed_engine_client_runs_inside_daemon(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_HOME", str(tmp_path))
    daemon = Daemon(home=tmp_path, in_process=False, startup_timeout_s=15)
    try:
        daemon.ensure(SetupPlan(backend="embedded"))
        engine = DaemonEngineClient(DaemonRpcTransport(data_dir=tmp_path))
        pot = await engine.pots.create(PotCreateRequest(name="default", use=True))
        status = await engine.context.status(EngineStatusRequest(pot_id=pot.pot_id))

        assert status.pot_id == pot.pot_id
        assert status.pot_name == "default"
        assert status.backend == "embedded"
    finally:
        daemon.stop()
