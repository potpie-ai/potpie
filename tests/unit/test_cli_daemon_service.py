from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from typer.testing import CliRunner

from potpie.cli import main as host_cli
from potpie.cli.commands import _common, bootstrap
from potpie.cli.telemetry.onboarding_events import CliSetupAnalyticsObserver
from potpie.daemon.lifecycle import Daemon
from potpie.daemon.process import launcher
from potpie_context_core.lifecycle import (
    SKIPPED,
    PlannedSetupStep,
    SetupPlan,
    SetupPreview,
    SetupReport,
    StepResult,
)

runner = CliRunner()


@dataclass
class _FakeDaemon:
    home: Path
    in_process: bool = False
    backend: str | None = None
    calls: list[str] = field(default_factory=list)

    def start(self) -> dict[str, int | str]:
        self.calls.append("start")
        return {"pid": 123, "socket": str(self.home / "daemon.sock"), "bind": "unix:x"}

    def status(self) -> dict[str, bool | str | int]:
        self.calls.append("status")
        status: dict[str, bool | str | int] = {
            "up": True,
            # Mirrors the real Daemon: "a process owns the pid" and "the process
            # answers" are separate facts, and only the second gates exit 0.
            "serving": True,
            "state": "running",
            "mode": "detached",
            "home": str(self.home),
            "pid": 123,
        }
        if self.backend is not None:
            status["backend"] = self.backend
        return status

    def ensure(self, plan: SetupPlan) -> None:
        self.calls.append(f"ensure:{plan.backend}")

    def stop(self) -> dict[str, str]:
        self.calls.append("stop")
        return {"detail": "daemon stopped"}

    def logs(self, *, tail: int | None = 200, since: object = None) -> list[str]:
        self.calls.append(f"logs:{tail}")
        return ["line one"]

    def log_path(self) -> Path:
        return self.home / "logs" / "potpied.log"


@dataclass
class _FakeHost:
    daemon: _FakeDaemon
    backend: object = field(
        default_factory=lambda: type("B", (), {"profile": "falkordb_lite"})()
    )


def test_daemon_lifecycle_commands_use_detached_daemon(tmp_path: Path) -> None:
    daemon = _FakeDaemon(home=tmp_path)
    _common.set_host(_FakeHost(daemon=daemon))

    start = runner.invoke(host_cli.app, ["--json", "daemon", "start"])
    status = runner.invoke(host_cli.app, ["--json", "daemon", "status"])
    restart = runner.invoke(host_cli.app, ["--json", "daemon", "restart"])
    stop = runner.invoke(host_cli.app, ["--json", "daemon", "stop"])

    assert start.exit_code == 0, start.stdout
    assert json.loads(start.stdout)["pid"] == 123
    assert status.exit_code == 0, status.stdout
    assert json.loads(status.stdout)["mode"] == "detached"
    assert restart.exit_code == 0, restart.stdout
    assert stop.exit_code == 0, stop.stdout
    assert daemon.calls == ["start", "status", "stop", "start", "stop"]


# --- S2-18: the exit code is the answer -------------------------------------
#
# Driven through the real ``Daemon`` over a real home directory, because the
# whole finding is that the command reported success regardless of what the
# lifecycle told it. A fake daemon returning a hand-written status dict would
# only test that this file and that file agree.


@dataclass
class _DaemonHost:
    daemon: Daemon


def _local_daemon(home: Path) -> Daemon:
    daemon = Daemon(home=home, in_process=False, health_timeout_s=1.0)
    _common.set_host(_DaemonHost(daemon=daemon))
    return daemon


def test_daemon_status_exits_nonzero_when_the_daemon_is_down(tmp_path: Path) -> None:
    """``potpie daemon status`` exited 0 for a daemon that was not there, which
    is the one thing a liveness check must never do."""
    _local_daemon(tmp_path)

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["code"] == "daemon_unavailable"
    # The status itself survives the error envelope: a wrapper that gates on the
    # exit code still gets to say what it found.
    assert payload["state"] == "not_running"
    assert payload["home"] == str(tmp_path)
    assert payload["recommended_next_action"] == "start it with 'potpie daemon start'"


def test_daemon_status_exits_nonzero_for_a_wedged_daemon(tmp_path: Path) -> None:
    """A live pid that does not answer is not a healthy daemon, and the repair
    is a restart rather than a start."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "daemon.pid").write_text(f"{child.pid}\n")
        (tmp_path / "discovery.json").write_text(
            json.dumps({"base_url": f"http://127.0.0.1:{_closed_port()}"})
        )
        _local_daemon(tmp_path)
        result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])
    finally:
        child.kill()
        child.wait(timeout=10)

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.stdout
    payload = json.loads(result.stdout)
    assert payload["state"] == "unresponsive"
    assert payload["up"] is True
    assert payload["recommended_next_action"] == (
        "restart it with 'potpie daemon restart'"
    )


def test_daemon_status_exits_zero_only_when_it_is_serving(tmp_path: Path) -> None:
    """The control: without this, "always exit 2" would pass the tests above."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (tmp_path / "daemon.pid").write_text(f"{os.getpid()}\n")
        (tmp_path / "discovery.json").write_text(
            json.dumps({"base_url": f"http://127.0.0.1:{server.server_address[1]}"})
        )
        _local_daemon(tmp_path)
        result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["state"] == "running" and payload["serving"] is True
    assert "ok" not in payload  # a success, not an error envelope


# --- S2-20 / S2-21 / S2-25: a daemon that died on its bind did not start -----


class _DiesOnBind:
    """A ``Popen`` stand-in that writes what uvicorn writes when it cannot bind.

    Driven through the real ``Daemon`` and the real launcher, because the whole
    finding is that this reported ``daemon started (pid=…)`` at exit 0 — a fake
    daemon returning a hand-written dict would only test that the CLI prints
    what it is handed.
    """

    pid = 5150
    returncode = 1

    def __call__(self, _args, *, env, **_kwargs):
        log = Path(env["CONTEXT_ENGINE_HOME"]) / "logs" / "potpied.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(
                "INFO:     Application startup complete.\n"
                "ERROR:    [Errno 48] error while attempting to bind on address "
                "('127.0.0.1', 8123): address already in use\n"
                "INFO:     Application shutdown complete.\n"
            )
        return self

    def poll(self) -> int:
        return 1


def test_daemon_start_exits_nonzero_when_the_daemon_dies_on_its_bind(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(launcher.subprocess, "Popen", _DiesOnBind())
    _local_daemon(tmp_path)

    result = runner.invoke(host_cli.app, ["--json", "daemon", "start"])

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["code"] == "daemon_start_failed"
    # The bind error itself, not "the daemon did not become ready".
    assert "address already in use" in payload["message"]
    assert payload["detail"].endswith("logs/potpied.log")


def test_daemon_restart_exits_nonzero_when_the_daemon_dies_on_its_bind(
    monkeypatch, tmp_path: Path
) -> None:
    """The same refusal on the other path into the launcher."""
    monkeypatch.setattr(launcher.subprocess, "Popen", _DiesOnBind())
    _local_daemon(tmp_path)

    result = runner.invoke(host_cli.app, ["--json", "daemon", "restart"])

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "daemon_start_failed"
    assert "address already in use" in payload["message"]


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        body = json.dumps({"ok": True, "backend": "falkordb_lite"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --- S2-11 / S2-40: the log flags ------------------------------------------


def _write_log(home: Path, lines: list[str]) -> Path:
    log_file = home / "logs" / "potpied.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_file


def test_daemon_logs_are_capped_and_tail_is_selectable(tmp_path: Path) -> None:
    _write_log(tmp_path, [f"line {n}" for n in range(1000)])
    _local_daemon(tmp_path)

    default = runner.invoke(host_cli.app, ["--json", "daemon", "logs"])
    explicit = runner.invoke(host_cli.app, ["--json", "daemon", "logs", "--tail", "5"])
    everything = runner.invoke(
        host_cli.app, ["--json", "daemon", "logs", "--tail", "0"]
    )

    assert default.exit_code == 0, default.stdout
    assert len(json.loads(default.stdout)["lines"]) == 200
    assert json.loads(explicit.stdout)["lines"] == [
        f"line {n}" for n in range(995, 1000)
    ]
    assert len(json.loads(everything.stdout)["lines"]) == 1000


def test_daemon_logs_since_rejects_an_unreadable_time(tmp_path: Path) -> None:
    _write_log(tmp_path, ["2026-08-12 09:00:00,000 INFO potpie [] hello"])
    _local_daemon(tmp_path)

    result = runner.invoke(
        host_cli.app, ["--json", "daemon", "logs", "--since", "yesterday"]
    )

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    assert json.loads(result.stdout)["code"] == "validation_error"


def test_daemon_logs_follow_reaches_the_lifecycle_and_streams_ndjson(
    monkeypatch, tmp_path: Path
) -> None:
    """``--follow`` parsed and then did nothing: the flag never left this file.

    The real stream is endless, so what is bounded here is the *wiring* — that
    the flag selects ``follow_logs`` at all, carries ``--tail``/``--since`` into
    it, and prints one JSON object per line rather than one document that could
    never be closed. ``follow_logs`` itself is tested against a real growing
    file in tests/unit/test_daemon_seam.py.
    """
    seen: dict[str, object] = {}

    def _fake_follow(self, **kwargs):
        seen.update(kwargs)
        yield from ("alpha", "beta")

    monkeypatch.setattr(Daemon, "follow_logs", _fake_follow)
    _write_log(tmp_path, ["alpha", "beta"])
    _local_daemon(tmp_path)

    result = runner.invoke(
        host_cli.app,
        ["--json", "daemon", "logs", "--follow", "--tail", "2", "--since", "5m"],
    )

    assert result.exit_code == 0, result.stdout
    assert [
        json.loads(line) for line in result.stdout.splitlines() if line.strip()
    ] == [
        {"line": "alpha"},
        {"line": "beta"},
    ]
    assert seen["tail"] == 2
    assert isinstance(seen["since"], datetime)


def test_daemon_logs_without_follow_stays_one_document(tmp_path: Path) -> None:
    """The control on the shape switch above."""
    _write_log(tmp_path, ["alpha"])
    _local_daemon(tmp_path)

    result = runner.invoke(host_cli.app, ["--json", "daemon", "logs"])

    payload = json.loads(result.stdout)
    assert payload["lines"] == ["alpha"]
    assert payload["follow"] is False
    assert payload["log_file"].endswith("logs/potpied.log")


def test_service_command_group_is_removed(tmp_path: Path) -> None:
    _common.set_host(_FakeHost(daemon=_FakeDaemon(home=tmp_path)))

    result = runner.invoke(host_cli.app, ["--json", "service", "status"])

    assert result.exit_code == 2
    assert "No such command 'service'" in result.output


def _patch_setup_host(monkeypatch, host: "_SetupHost") -> None:
    """Inject the host through the seam ``setup`` actually builds from.

    ``setup`` used to reach for ``get_host()`` under ``--json`` — which, on the
    default local origin, is the daemon RPC — and for a locally wired host
    everywhere else. It now builds the local host in both modes, so this is the
    one seam left; a fake behind ``get_host`` would silently stop being used and
    these tests would start asserting against a real setup run.
    """
    monkeypatch.setattr(
        bootstrap,
        "_build_local_setup_host",
        lambda **_kwargs: (host, host.backend.profile, host.daemon.in_process),
    )
    monkeypatch.setattr(
        "potpie.cli.ui.setup_ux.rich_enabled",
        lambda **_kwargs: False,
    )


def test_setup_daemon_dry_run_marks_daemon_host_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    host = _SetupHost(home=tmp_path)
    _patch_setup_host(monkeypatch, host)

    result = runner.invoke(host_cli.app, ["--json", "setup", "--daemon", "--dry-run"])

    assert result.exit_code == 0, result.stdout
    assert host.setup.host_mode == "daemon"
    assert json.loads(result.stdout)["plan"]["host_mode"] == "daemon"


def test_setup_daemon_uses_daemon_status_for_backend_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    host = _SetupHost(home=tmp_path)
    host.backend.profile = "falkordb_lite"
    host.daemon.backend = "embedded"
    _patch_setup_host(monkeypatch, host)

    result = runner.invoke(
        host_cli.app,
        ["--json", "setup", "--backend", "embedded", "--repo", "potpie", "--yes"],
    )

    assert result.exit_code == 0, result.stdout
    assert host.setup.host_mode == "daemon"
    # Reads the running daemon's backend; it no longer *starts* one first. The
    # pre-start existed only because the run itself was about to travel over
    # that daemon's RPC — the orchestrator's own `daemon` step starts it now,
    # after config and the backend exist.
    assert host.daemon.calls == ["status"]


def test_setup_daemon_fails_when_requested_backend_cannot_be_verified(
    monkeypatch,
    tmp_path: Path,
) -> None:
    host = _SetupHost(home=tmp_path)
    _patch_setup_host(monkeypatch, host)

    result = runner.invoke(
        host_cli.app,
        ["--json", "setup", "--backend", "embedded", "--repo", "potpie", "--yes"],
    )

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.stdout)
    assert payload["code"] == "validation_error"
    assert "backend could not be verified" in payload["message"]


class _Setup:
    host_mode: str | None = None

    def set_observer(self, observer: CliSetupAnalyticsObserver) -> None:
        return None

    def preview(self, plan: SetupPlan) -> SetupPreview:
        self.host_mode = plan.host_mode
        return SetupPreview(
            plan,
            (
                PlannedSetupStep(
                    "daemon",
                    True,
                    "host.daemon",
                    "ensure daemon",
                ),
            ),
        )

    def run(self, plan: SetupPlan) -> SetupReport:
        self.host_mode = plan.host_mode
        return SetupReport(
            plan,
            (
                StepResult(
                    "daemon",
                    SKIPPED,
                    "daemon already running",
                    metadata={"mode": plan.host_mode},
                ),
            ),
        )


class _Backend:
    profile = "embedded"


@dataclass
class _SetupHost:
    home: Path
    profile: str = "local"
    backend: _Backend = field(init=False)
    daemon: _FakeDaemon = field(init=False)
    setup: _Setup = field(init=False)

    def __post_init__(self) -> None:
        self.backend = _Backend()
        self.daemon = _FakeDaemon(home=self.home, in_process=False)
        self.setup = _Setup()
