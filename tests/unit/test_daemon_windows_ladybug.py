"""Windows Ladybug daemon lifecycle - real start path (no refuse-only guard)."""

from __future__ import annotations

# ruff: noqa: S101 - pytest assertions are intentional in this test module.

from types import SimpleNamespace

from typer.testing import CliRunner

import pytest

from potpie.cli.commands import daemon as daemon_cmd
from potpie.cli.main import build_app
from potpie.daemon.__main__ import _warm_daemon_embedder
from potpie.daemon.lifecycle import DaemonStartError


@pytest.mark.unit
def test_daemon_start_surfaces_start_error_not_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any start failure must be daemon_start_failed, never Unexpected internal error."""

    def boom(_daemon):
        raise DaemonStartError(
            "the daemon did not complete an authenticated handshake",
            log_path=None,
        )

    monkeypatch.setattr(daemon_cmd, "_detached_daemon", lambda: object())
    monkeypatch.setattr(
        "potpie.daemon.lifecycle.Daemon.start",
        boom,
    )
    # Patch through the module-level path used by _start
    monkeypatch.setattr(
        daemon_cmd.Daemon,
        "start",
        boom,
    )

    class _BoomDaemon:
        def start(self):
            raise DaemonStartError("readiness failed", log_path=None)

    monkeypatch.setattr(daemon_cmd, "_detached_daemon", lambda: _BoomDaemon())
    runner = CliRunner()
    result = runner.invoke(build_app(), ["--json", "daemon", "start"])
    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Unexpected internal error" not in combined
    assert "daemon_start_failed" in combined or "readiness failed" in combined


@pytest.mark.unit
def test_daemon_start_unexpected_exception_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BoomDaemon:
        def start(self):
            raise AttributeError("os.WNOHANG")

    monkeypatch.setattr(daemon_cmd, "_detached_daemon", lambda: _BoomDaemon())
    runner = CliRunner()
    result = runner.invoke(build_app(), ["--json", "daemon", "start"])
    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Unexpected internal error" not in combined
    assert "daemon_start_failed" in combined
    assert "AttributeError" in combined


@pytest.mark.unit
def test_pid_alive_uses_windows_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from potpie.daemon import lifecycle as life

    called: list[int] = []

    def fake_windows(pid: int) -> bool:
        called.append(pid)
        return True

    monkeypatch.setattr(life, "_pid_alive_windows", fake_windows)
    monkeypatch.setattr(life.os, "name", "nt")
    assert life._pid_alive(4242) is True
    assert called == [4242]


@pytest.mark.unit
def test_daemon_warms_embedder_before_accepting_requests() -> None:
    class _Embedder:
        name = "sentence-transformers/all-MiniLM-L6-v2"
        calls = 0

        def prepare(self) -> dict[str, object]:
            self.calls += 1
            return {"dimensions": 384}

    class _Backend:
        def __init__(self, embedder: _Embedder) -> None:
            self.embedder = embedder

    class _Engine:
        def __init__(self, backend: _Backend) -> None:
            self.backend = backend

    embedder = _Embedder()
    composition = SimpleNamespace(engine=_Engine(_Backend(embedder)))

    _warm_daemon_embedder(composition)

    assert embedder.calls == 1


@pytest.mark.unit
def test_daemon_warmup_is_noop_for_embedders_without_prepare() -> None:
    class _Backend:
        embedder = object()

    class _Engine:
        backend = _Backend()

    _warm_daemon_embedder(SimpleNamespace(engine=_Engine()))


@pytest.mark.unit
def test_daemon_warmup_is_soft_skipped_on_windows(monkeypatch) -> None:
    from potpie.daemon import __main__ as daemon_main

    class _Embedder:
        name = "sentence-transformers/all-MiniLM-L6-v2"

        def prepare(self):
            raise AssertionError("Windows daemon must not load native model in-process")

    class _Backend:
        embedder = _Embedder()

    class _Engine:
        backend = _Backend()

    monkeypatch.setattr(daemon_main.sys, "platform", "win32")
    daemon_main._warm_daemon_embedder(SimpleNamespace(engine=_Engine()))
