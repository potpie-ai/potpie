from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli
from adapters.inbound.cli.commands import _common, bootstrap
from adapters.inbound.cli.telemetry.onboarding_events import CliSetupAnalyticsObserver
from domain.lifecycle import SKIPPED, PlannedSetupStep, SetupPlan, SetupPreview, SetupReport, StepResult

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

    def logs(self, *, follow: bool = False) -> list[str]:
        self.calls.append(f"logs:{follow}")
        return ["line one"]


@dataclass
class _FakeHost:
    daemon: _FakeDaemon
    backend: object = field(default_factory=lambda: type("B", (), {"profile": "falkordb_lite"})())


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


def test_service_status_fails_cleanly_when_daemon_is_down(tmp_path: Path) -> None:
    _common.set_host(_FakeHost(daemon=_FakeDaemon(home=tmp_path)))

    result = runner.invoke(host_cli.app, ["--json", "service", "status"])

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    payload = json.loads(result.stdout)
    assert payload["code"] == "daemon_not_running"


def test_service_logs_reads_service_log_without_running_daemon(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "service-graph.log").write_text("hello\n", encoding="utf-8")
    _common.set_host(_FakeHost(daemon=_FakeDaemon(home=tmp_path)))

    result = runner.invoke(host_cli.app, ["--json", "service", "logs", "graph"])

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["lines"] == ["hello"]


def test_service_logs_falkordb_lite_explains_embedded_backend(tmp_path: Path) -> None:
    _common.set_host(_FakeHost(daemon=_FakeDaemon(home=tmp_path)))

    result = CliRunner().invoke(
        host_cli.app, ["--json", "service", "logs", "falkordb_lite"]
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "embedded_backend"
    assert payload["profile"] == "falkordb_lite"
    assert payload["recommended_log_command"] == "potpie daemon logs"
    assert "database_path" in payload
    assert "no separate service log" in result.stdout


def test_service_logs_returns_helpful_message_when_daemon_ipc_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    @contextmanager
    def _failing_client(_home: Path):
        class _Client:
            def get(self, _path: str) -> None:
                raise httpx.ConnectError("stale daemon socket")

        yield _Client()

    monkeypatch.setattr(
        "adapters.inbound.cli.commands.service.client_for",
        _failing_client,
    )
    _common.set_host(_FakeHost(daemon=_FakeDaemon(home=tmp_path)))

    result = runner.invoke(host_cli.app, ["--json", "service", "logs", "graph"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_log_file"
    assert "no log file for 'graph'" in payload["message"]


def test_service_logs_follow_exits_cleanly_on_keyboard_interrupt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "service-graph.log").write_text("", encoding="utf-8")
    _common.set_host(_FakeHost(daemon=_FakeDaemon(home=tmp_path)))
    monkeypatch.setattr(
        "time.sleep",
        lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    result = runner.invoke(host_cli.app, ["service", "logs", "graph", "--follow"])

    assert result.exit_code == 0, result.stdout


def test_setup_daemon_dry_run_marks_daemon_host_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    host = _SetupHost(home=tmp_path)
    monkeypatch.setattr(bootstrap, "get_host", lambda: host)
    monkeypatch.setattr(
        "adapters.inbound.cli.ui.setup_ux.rich_enabled",
        lambda **_kwargs: False,
    )

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
    monkeypatch.setattr(bootstrap, "get_host", lambda: host)
    monkeypatch.setattr(
        "adapters.inbound.cli.ui.setup_ux.rich_enabled",
        lambda **_kwargs: False,
    )

    result = runner.invoke(
        host_cli.app,
        ["--json", "setup", "--backend", "embedded", "--repo", "potpie", "--yes"],
    )

    assert result.exit_code == 0, result.stdout
    assert host.setup.host_mode == "daemon"
    assert host.daemon.calls == ["ensure:embedded", "status"]


def test_setup_daemon_fails_when_requested_backend_cannot_be_verified(
    monkeypatch,
    tmp_path: Path,
) -> None:
    host = _SetupHost(home=tmp_path)
    monkeypatch.setattr(bootstrap, "get_host", lambda: host)
    monkeypatch.setattr(
        "adapters.inbound.cli.ui.setup_ux.rich_enabled",
        lambda **_kwargs: False,
    )

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
