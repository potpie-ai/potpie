"""End-to-end: the real detached daemon starts, serves the context ops over UDS, and stops.

Exercises the dissolved daemon path: launcher.start_detached spawns ``host.daemon_runtime``
(which builds a real HostShell), the http transport serves the registered context_graph
component, and stop_daemon tears it down cleanly. Replaces potpied's CLI-app readiness/e2e
tests, which targeted the old standalone ``potpie init/daemon`` typer apps.
"""
from __future__ import annotations

import pathlib

import pytest

from adapters.outbound.daemon_process.launcher import start_detached, stop_daemon, DaemonStartError
from adapters.outbound.daemon_process.ipc_client import client_for


@pytest.mark.integration
def test_detached_daemon_starts_serves_and_stops(tmp_path: pathlib.Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")  # child hosts in-process

    info = start_detached(home, ready_timeout_s=60.0)
    try:
        assert info["pid"] > 0
        assert info["socket"] == str(home / "daemon.sock")
        assert (home / "discovery.json").exists()
        assert (home / "daemon.pid").exists()

        with client_for(home) as c:
            ops = c.get("/op").json()["operations"]
            assert set(ops) == {
                "context.resolve", "context.search", "context.record", "context.status",
            }
            assert c.get("/admin/health").json()["status"] == "ready"
    finally:
        msg = stop_daemon(home)

    assert msg == "daemon stopped"
    assert not (home / "daemon.pid").exists()
    assert not (home / "daemon.sock").exists()


@pytest.mark.integration
def test_start_detached_rejects_double_start(tmp_path: pathlib.Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")

    start_detached(home, ready_timeout_s=60.0)
    try:
        with pytest.raises(DaemonStartError, match="already running"):
            start_detached(home, ready_timeout_s=5.0)
    finally:
        stop_daemon(home)
