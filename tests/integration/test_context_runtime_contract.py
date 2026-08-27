"""Final behavior contract for local and canonical-daemon runtime paths."""

# ruff: noqa: S101 - pytest characterization tests use assertions intentionally.

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from potpie.cli import main as cli_main
from potpie.cli.commands import _common
from potpie.daemon.discovery import write_daemon_pid
from potpie.daemon.lifecycle import Daemon
from potpie_context_engine.core.lifecycle import DONE, SKIPPED, SetupPlan


pytestmark = pytest.mark.integration

_FACT = "final-context-runtime-contract"
_DISCOVERY_FILES = (
    "daemon.pid",
    "discovery.json",
    "daemon.credential",
    "daemon.json",
)


def _configure_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime_home: Path,
    *,
    host_mode: str,
) -> None:
    user_home = runtime_home / "home"
    xdg_home = runtime_home / "xdg"
    user_home.mkdir(parents=True, exist_ok=True)
    xdg_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(runtime_home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", host_mode)
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "embedded")
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    monkeypatch.delenv("POTPIE_DAEMON_PORT", raising=False)
    monkeypatch.delenv("POTPIE_DAEMON_TOKEN", raising=False)
    _common.set_runtime(None)


@pytest.fixture
def isolated_daemon_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide a fully isolated filesystem and environment for a real daemon."""
    runtime_home = tmp_path / "runtime"
    _configure_runtime(monkeypatch, runtime_home, host_mode="daemon")
    try:
        yield runtime_home
    finally:
        _common.set_runtime(None)


def _assert_json_result(
    result: Any,
    command: str,
    *,
    exit_code: int = 0,
) -> dict[str, Any]:
    assert result.exit_code == exit_code, result.output
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - assertion helper
        raise AssertionError(
            f"expected one JSON value from potpie {command}; "
            f"stdout was:\n{result.stdout}"
        ) from exc
    assert isinstance(payload, dict)
    return payload


def _invoke_json(
    runner: CliRunner,
    *args: str,
    exit_code: int = 0,
) -> dict[str, Any]:
    result = runner.invoke(cli_main.app, ["--json", *args])
    return _assert_json_result(result, " ".join(args), exit_code=exit_code)


def _process_alive(pid: int) -> bool:
    try:
        waited_pid, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        waited_pid = 0
    if waited_pid == pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _assert_daemon_stopped(runtime_home: Path, pid: int) -> None:
    deadline = time.monotonic() + 2.0
    while _process_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_alive(pid)
    assert all(not (runtime_home / name).exists() for name in _DISCOVERY_FILES)


def _stop_daemon(runtime_home: Path, *, pid: int | None = None) -> None:
    daemon = Daemon(home=runtime_home, in_process=False)
    daemon.stop()
    if pid is not None:
        _assert_daemon_stopped(runtime_home, pid)


def test_daemon_ensure_starts_reuses_and_cleans_up(
    isolated_daemon_runtime: Path,
) -> None:
    runtime_home = isolated_daemon_runtime
    daemon = Daemon(home=runtime_home, in_process=False, startup_timeout_s=15)
    pid: int | None = None
    try:
        first = daemon.ensure(SetupPlan(backend="embedded", embeddings="none"))
        second = daemon.ensure(SetupPlan(backend="embedded", embeddings="none"))
        pid = int(first.metadata["pid"])

        assert first.state == DONE
        assert second.state == SKIPPED
        status = daemon.status()
        assert status["up"] is True
        assert status["mode"] == "detached"
        assert status["backend"] == "embedded"
    finally:
        _stop_daemon(runtime_home, pid=pid)


def test_daemon_restart_preserves_running_backend(
    isolated_daemon_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_home = isolated_daemon_runtime
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "falkordb_lite")
    daemon = Daemon(home=runtime_home, in_process=False, startup_timeout_s=15)
    restarted_pid: int | None = None
    try:
        daemon.ensure(SetupPlan(backend="embedded", embeddings="none"))
        restarted = daemon.restart()
        restarted_pid = int(restarted["pid"])

        assert restarted["started"]["backend"] == "embedded"
        assert daemon.status()["backend"] == "embedded"
    finally:
        _stop_daemon(runtime_home, pid=restarted_pid)


def test_fresh_controller_can_status_then_typed_stop_existing_daemon(
    isolated_daemon_runtime: Path,
) -> None:
    runtime_home = isolated_daemon_runtime
    starter = Daemon(home=runtime_home, in_process=False, startup_timeout_s=15)
    pid: int | None = None
    try:
        started = starter.ensure(SetupPlan(backend="embedded", embeddings="none"))
        pid = int(started.metadata["pid"])
        attached = Daemon(home=runtime_home, in_process=False, startup_timeout_s=15)

        status = attached.status()
        stopped = attached.stop()

        assert status["ready"] is True
        assert stopped["detail"] == "daemon stopped"
        _assert_daemon_stopped(runtime_home, pid)
        pid = None
    finally:
        _stop_daemon(runtime_home, pid=pid)


def test_daemon_crash_status_recovers_stale_runtime_records(
    isolated_daemon_runtime: Path,
) -> None:
    runtime_home = isolated_daemon_runtime
    daemon = Daemon(home=runtime_home, in_process=False, startup_timeout_s=15)
    pid: int | None = None
    try:
        started = daemon.ensure(SetupPlan(backend="embedded", embeddings="none"))
        pid = int(started.metadata["pid"])

        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while _process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)

        status = daemon.status()

        assert status["up"] is False
        assert status["pid"] is None
        assert all(not (runtime_home / name).exists() for name in _DISCOVERY_FILES)
    finally:
        _stop_daemon(runtime_home, pid=pid)


def test_stale_reused_pid_never_signals_unrelated_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_home = tmp_path / "stale-runtime"
    _configure_runtime(monkeypatch, runtime_home, host_mode="daemon")
    child = subprocess.Popen(  # noqa: S603 - controlled disposable test process
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        write_daemon_pid(runtime_home, child.pid)

        result = CliRunner().invoke(cli_main.app, ["--json", "daemon", "stop"])
        payload = _assert_json_result(
            result,
            "daemon stop with stale reused pid",
            exit_code=_common.EXIT_UNAVAILABLE,
        )

        assert payload["code"] == "daemon_attached_identity_unavailable"
        assert "refusing to signal" in payload["message"]
        assert child.poll() is None
        assert (runtime_home / "daemon.pid").read_text(encoding="utf-8").strip() == str(
            child.pid
        )
    finally:
        _common.set_runtime(None)
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_canonical_daemon_uses_private_discovery_and_separate_credential(
    isolated_daemon_runtime: Path,
) -> None:
    runtime_home = isolated_daemon_runtime
    daemon = Daemon(home=runtime_home, in_process=False, startup_timeout_s=15)
    pid: int | None = None
    try:
        started = daemon.ensure(SetupPlan(backend="embedded", embeddings="none"))
        pid = int(started.metadata["pid"])
        discovery = json.loads(
            (runtime_home / "discovery.json").read_text(encoding="utf-8")
        )
        credential = (
            (runtime_home / "daemon.credential").read_text(encoding="utf-8").strip()
        )

        assert discovery["pid"] == pid
        assert discovery["transport"]["kind"] in {"uds", "tcp"}
        assert discovery["authentication"] == {
            "scheme": "bearer",
            "credential_file": str(runtime_home / "daemon.credential"),
        }
        assert credential and credential not in json.dumps(discovery)
        assert not (runtime_home / "daemon.json").exists()
        assert (runtime_home / "discovery.json").stat().st_mode & 0o077 == 0
        assert (runtime_home / "daemon.credential").stat().st_mode & 0o077 == 0
        status = daemon.status()
        assert status["ready"] is True
        ui_redirect = httpx.get(f"{status['url']}/ui", timeout=3.0)
        assert ui_redirect.status_code == 307
        assert ui_redirect.headers["location"] == f"{status['url']}/ui/"
        assert httpx.get(f"{status['url']}/ui/", timeout=3.0).status_code == 200
        assert httpx.post(f"{status['url']}/rpc", timeout=3.0).status_code == 404
        assert httpx.post(f"{status['url']}/attr", timeout=3.0).status_code == 404
    finally:
        _stop_daemon(runtime_home, pid=pid)


@pytest.mark.parametrize("host_mode", ("in_process", "daemon"))
def test_local_and_daemon_cli_paths_have_equivalent_domain_and_error_outcomes(
    host_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_home = tmp_path / "runtime"
    daemon_pid: int | None = None
    stop_result: Any | None = None
    _configure_runtime(monkeypatch, runtime_home, host_mode=host_mode)
    runner = CliRunner()
    try:
        setup = _invoke_json(
            runner,
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
            "--embeddings",
            "none",
            "--daemon" if host_mode == "daemon" else "--in-process",
        )
        if host_mode == "daemon":
            daemon_status = _invoke_json(runner, "daemon", "status")
            daemon_pid = int(daemon_status["pid"])

        record = _invoke_json(
            runner,
            "record",
            "--type",
            "feature_note",
            "--summary",
            _FACT,
            "--scope",
            "service:context-engine",
        )
        search = _invoke_json(runner, "search", _FACT, "--include", "raw_graph")
        resolve = _invoke_json(runner, "resolve", _FACT, "--include", "raw_graph")
        status = _invoke_json(runner, "status")
        invalid = _invoke_json(
            runner,
            "record",
            "--type",
            "note",
            "--summary",
            "unsupported record type",
            exit_code=_common.EXIT_VALIDATION,
        )

        observation = {
            "setup_ok": setup["ok"],
            "record_status": record["status"],
            "mutations_applied": record["mutations_applied"],
            "search_facts": [item["payload"].get("fact") for item in search["items"]],
            "resolve_facts": [item["payload"].get("fact") for item in resolve["items"]],
            "backend_ready": status["backend_ready"],
            "backend_profile": status["data_plane"]["backend_profile"],
            "claim_count": status["data_plane"]["counts"]["claims"],
            "invalid_code": invalid["code"],
        }
    finally:
        if host_mode == "daemon":
            stop_result = runner.invoke(cli_main.app, ["--json", "daemon", "stop"])
            if stop_result.exit_code != 0:
                Daemon(home=runtime_home, in_process=False).stop()
            if daemon_pid is not None:
                _assert_daemon_stopped(runtime_home, daemon_pid)
        _common.set_runtime(None)

    if host_mode == "daemon":
        assert stop_result is not None
        stop_payload = _assert_json_result(stop_result, "daemon stop")
        assert stop_payload["detail"] == "daemon stopped"
    else:
        assert all(not (runtime_home / name).exists() for name in _DISCOVERY_FILES)

    assert observation == {
        "setup_ok": True,
        "record_status": "recorded",
        "mutations_applied": 1,
        "search_facts": [_FACT],
        "resolve_facts": [_FACT],
        "backend_ready": True,
        "backend_profile": "embedded",
        "claim_count": 1,
        "invalid_code": "validation_error",
    }
