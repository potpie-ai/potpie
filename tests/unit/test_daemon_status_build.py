"""``Daemon.status`` says which build answers, and whether it is this one's."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie import build_info
from potpie.daemon import lifecycle
from potpie.daemon.lifecycle import Daemon, build_is_stale


def test_build_is_stale_only_when_both_revs_are_known_and_differ(monkeypatch) -> None:
    monkeypatch.setattr(build_info, "build_stamp", lambda: {"rev": "ours"})
    assert build_is_stale({"rev": "ours"}) is False
    assert build_is_stale({"rev": "theirs"}) is True
    assert build_is_stale({"rev": None}) is None
    monkeypatch.setattr(build_info, "build_stamp", lambda: {})
    assert build_is_stale({"rev": "theirs"}) is None


def test_status_carries_the_served_build_and_the_comparison(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(build_info, "build_stamp", lambda: {"rev": "ours"})
    daemon = Daemon(home=tmp_path, in_process=False)
    (tmp_path / "daemon.pid").write_text("4242\n")
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        Daemon,
        "health",
        lambda self: {
            "live": True,
            "mode": "detached",
            "pid": 4242,
            "backend": "embedded",
            "version": "2.0.0",
            "build": {"rev": "theirs", "dirty": False, "built_at": "2026-08-19T00:00:00+00:00"},
        },
    )

    status = daemon.status()

    assert status["state"] == lifecycle.STATE_RUNNING
    assert status["version"] == "2.0.0"
    assert status["build"]["rev"] == "theirs"
    assert status["stale"] is True


def test_status_without_a_build_has_nothing_to_say_about_staleness(monkeypatch, tmp_path: Path) -> None:
    """A daemon from before the stamp: no `build`, no `stale` key at all."""
    daemon = Daemon(home=tmp_path, in_process=False)
    (tmp_path / "daemon.pid").write_text("4242\n")
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        Daemon, "health", lambda self: {"live": True, "mode": "detached", "pid": 4242, "backend": "embedded"}
    )

    status = daemon.status()

    assert "build" not in status
    assert "stale" not in status


@pytest.mark.parametrize("in_process", [True])
def test_in_process_status_is_untouched(in_process: bool, tmp_path: Path) -> None:
    status = Daemon(home=tmp_path, in_process=in_process).status()
    assert status["state"] == lifecycle.STATE_IN_PROCESS
    assert "stale" not in status
