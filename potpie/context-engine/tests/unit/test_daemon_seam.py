"""host.daemon.Daemon seam: in-process stand-in vs detached lifecycle (launcher faked)."""

from __future__ import annotations

import pathlib

from domain.lifecycle import DONE, SKIPPED
from host.daemon import Daemon


def test_in_process_status_health_and_ensure_skips(tmp_path: pathlib.Path):
    d = Daemon(home=tmp_path, in_process=True)
    assert d.status()["mode"] == "in_process" and d.status()["up"] is True
    assert d.health() == {"live": True, "mode": "in_process"}
    res = d.ensure()
    assert res.state == SKIPPED and res.metadata["mode"] == "in_process"


def test_detached_ensure_starts_when_not_running(tmp_path: pathlib.Path, monkeypatch):
    started = {}

    def fake_start_detached(home, **kw):
        started["home"] = home
        return {
            "pid": 4242,
            "socket": str(home / "daemon.sock"),
            "bind": f"unix:{home}/daemon.sock",
        }

    monkeypatch.setattr(
        "adapters.outbound.daemon_process.launcher.start_detached", fake_start_detached
    )
    d = Daemon(home=tmp_path, in_process=False)
    res = d.ensure()
    assert res.state == DONE
    assert res.metadata["pid"] == 4242
    assert started["home"] == tmp_path


def test_detached_ensure_reuses_running_daemon(tmp_path: pathlib.Path, monkeypatch):
    # Pretend a live daemon is already recorded.
    (tmp_path / "daemon.pid").write_text("999999\n")
    (tmp_path / "discovery.json").write_text('{"bind": "unix:/x/daemon.sock"}')
    monkeypatch.setattr("host.daemon._pid_alive", lambda pid: True)

    def _boom(*a, **k):  # must NOT be called when already running
        raise AssertionError("start_detached should not be called when daemon is up")

    monkeypatch.setattr(
        "adapters.outbound.daemon_process.launcher.start_detached", _boom
    )
    d = Daemon(home=tmp_path, in_process=False)
    res = d.ensure()
    assert res.state == DONE and "already running" in (res.detail or "")


def test_install_is_idempotent_noop(tmp_path: pathlib.Path):
    d = Daemon(home=tmp_path, in_process=False)
    out = d.install()
    assert (
        out["installed"] is False
    )  # never raises; does not gate the installer setup step
