"""A daemon left running by another build is seen, stopped and replaced.

Two builds are simulated with real processes: the "older" daemon runs with a
different operation catalog fingerprint, and this test process is the newer
CLI. The older daemon refuses the newer client's context operations, but its
control-scoped ticket still lets ``status`` report it and ``restart`` stop it
through authenticated typed shutdown, with no signal.
"""

# ruff: noqa: S101 - pytest integration tests use assertions intentionally.

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from potpie.cli import main as cli_main
from potpie.cli.commands import _common
from potpie.daemon.lifecycle import Daemon
from potpie.runtime import clients as runtime_clients

pytestmark = pytest.mark.integration

_OLDER_CATALOG = "0" * 64
#: The daemon entry point, run as a build whose operation catalog differs.
_OLDER_BUILD_DAEMON = (
    "import potpie.runtime.server as server\n"
    f"server.operation_catalog_fingerprint = lambda: {_OLDER_CATALOG!r}\n"
    "from potpie.daemon.__main__ import main\n"
    "main()\n"
)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runtime_home = tmp_path / "runtime"
    for name, path in (
        ("HOME", tmp_path / "home"),
        ("XDG_CONFIG_HOME", tmp_path / "xdg"),
        ("POTPIE_HARNESS_HOME", tmp_path / "harness"),
    ):
        path.mkdir(parents=True)
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(runtime_home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "daemon")
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "embedded")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    _common.set_runtime(None)
    try:
        yield runtime_home
    finally:
        _common.set_runtime(None)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_older_build_daemon(
    runtime_home: Path, monkeypatch: pytest.MonkeyPatch
) -> int:
    launch_spec = Daemon._launch_spec

    def older_build_launch_spec(self: Daemon, **kwargs):
        return replace(
            launch_spec(self, **kwargs),
            command=(sys.executable, "-c", _OLDER_BUILD_DAEMON),
        )

    # While it starts, this process is the older build's CLI too.
    with monkeypatch.context() as older_build:
        older_build.setattr(Daemon, "_launch_spec", older_build_launch_spec)
        older_build.setattr(
            runtime_clients, "operation_catalog_fingerprint", lambda: _OLDER_CATALOG
        )
        started = Daemon(home=runtime_home, in_process=False, startup_timeout_s=30)
        return int(started.start(backend="embedded")["pid"])


def test_a_daemon_from_another_build_is_reported_and_replaced_by_restart(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older_pid = _start_older_build_daemon(isolated_home, monkeypatch)
    replacement_pid: int | None = None
    try:
        daemon = Daemon(home=isolated_home, in_process=False, startup_timeout_s=30)

        seen = daemon.status()
        cli = CliRunner().invoke(cli_main.app, ["--json", "daemon", "status"])
        restarted = daemon.restart()
        replacement_pid = int(restarted["pid"])
        deadline = time.monotonic() + 5
        while _alive(older_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        current = daemon.status()

        assert seen["up"] is True
        assert seen["ready"] is False
        assert seen["compatible"] is False
        assert seen["stale"] is True
        assert seen["backend"] == "embedded"
        assert seen["pid"] == older_pid
        assert cli.exit_code == 2, cli.stdout
        payload = json.loads(cli.stdout)
        assert payload["code"] == "daemon_incompatible"
        assert payload["stale"] is True
        assert "potpie daemon restart" in payload["recommended_next_action"]
        assert replacement_pid != older_pid
        assert not _alive(older_pid)
        assert current["ready"] is True
        assert current["compatible"] is True
        assert current["stale"] is not True
        assert current["backend"] == "embedded"
    finally:
        with contextlib.suppress(Exception):
            Daemon(home=isolated_home, in_process=False).stop()
        for pid in (older_pid, replacement_pid):
            if pid is not None and _alive(pid):
                # Test cleanup of processes this test started, if a stop failed.
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)


def test_daemon_stop_reaches_a_daemon_from_another_build(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older_pid = _start_older_build_daemon(isolated_home, monkeypatch)
    try:
        stopped = CliRunner().invoke(cli_main.app, ["--json", "daemon", "stop"])
        deadline = time.monotonic() + 5
        while _alive(older_pid) and time.monotonic() < deadline:
            time.sleep(0.05)

        assert stopped.exit_code == 0, stopped.stdout
        assert json.loads(stopped.stdout)["detail"] == "daemon stopped"
        assert not _alive(older_pid)
        assert not (isolated_home / "discovery.json").exists()
    finally:
        if _alive(older_pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(older_pid, signal.SIGKILL)
