"""Daemon seam: in-process stand-in versus controller-backed lifecycle."""

from __future__ import annotations

# ruff: noqa: S101 - pytest assertions are intentional.

import pathlib

from potpie.runtime import ControllerStatus
from potpie_context_engine import Success
from potpie_context_engine.core.lifecycle import DONE, SKIPPED
from potpie.daemon.lifecycle import Daemon


def test_in_process_status_health_and_ensure_skips(tmp_path: pathlib.Path):
    d = Daemon(home=tmp_path, in_process=True)
    assert d.status()["mode"] == "in_process" and d.status()["up"] is True
    assert d.health() == {"live": True, "mode": "in_process"}
    res = d.ensure()
    assert res.state == SKIPPED and res.metadata["mode"] == "in_process"


def test_detached_ensure_starts_when_not_running(tmp_path: pathlib.Path, monkeypatch):
    started = {}

    class _Controller:
        pid = None

        async def start(self):
            started["home"] = tmp_path
            return Success(
                ControllerStatus(
                    running=True,
                    ready=True,
                    pid=4242,
                    instance_id=None,
                    exit_code=None,
                )
            )

    monkeypatch.setattr(Daemon, "_new_controller", lambda *_args, **_kw: _Controller())
    d = Daemon(home=tmp_path, in_process=False)
    res = d.ensure()
    assert res.state == DONE
    assert res.metadata["pid"] == 4242
    assert started["home"] == tmp_path


def test_detached_ensure_reuses_running_daemon(tmp_path: pathlib.Path, monkeypatch):
    # Pretend a live daemon is already recorded.
    (tmp_path / "daemon.pid").write_text("999999\n")
    (tmp_path / "discovery.json").write_text('{"bind": "unix:/x/daemon.sock"}')
    monkeypatch.setattr("potpie.daemon.lifecycle._pid_alive", lambda pid: True)

    d = Daemon(home=tmp_path, in_process=False)
    res = d.ensure()
    assert res.state == SKIPPED and "already running" in (res.detail or "")


def test_install_is_idempotent_noop(tmp_path: pathlib.Path):
    d = Daemon(home=tmp_path, in_process=False)
    out = d.install()
    assert (
        out["installed"] is False
    )  # never raises; does not gate the installer setup step
