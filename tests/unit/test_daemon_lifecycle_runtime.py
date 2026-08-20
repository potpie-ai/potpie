"""Unit coverage for detached-daemon lifecycle decisions."""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

from pathlib import Path

import pytest

from potpie.daemon.lifecycle import Daemon


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
