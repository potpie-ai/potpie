from __future__ import annotations

# ruff: noqa: S101 - pytest assertions are intentional.

import json
from dataclasses import dataclass, field
from pathlib import Path

from typer.testing import CliRunner

from potpie.cli import main as host_cli
from potpie.cli.commands import _common, bootstrap
from potpie.cli.telemetry.onboarding_events import CliSetupAnalyticsObserver
from potpie.runtime.root_services import build_root_runtime_services
from potpie_context_engine.core.lifecycle import (
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


def test_service_command_group_is_removed(tmp_path: Path) -> None:
    _common.set_host(_FakeHost(daemon=_FakeDaemon(home=tmp_path)))

    result = runner.invoke(host_cli.app, ["--json", "service", "status"])

    assert result.exit_code == 2
    assert "No such command 'service'" in result.output


def test_setup_daemon_dry_run_marks_daemon_host_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    host = _SetupHost(home=tmp_path)
    runtime = build_root_runtime_services(host)
    monkeypatch.setattr(bootstrap, "get_root_runtime", lambda: runtime)
    monkeypatch.setattr(
        "potpie.cli.ui.setup_ux.rich_enabled",
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
    runtime = build_root_runtime_services(host)
    monkeypatch.setattr(bootstrap, "get_root_runtime", lambda: runtime)
    monkeypatch.setattr(
        "potpie.cli.ui.setup_ux.rich_enabled",
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
    runtime = build_root_runtime_services(host)
    monkeypatch.setattr(bootstrap, "get_root_runtime", lambda: runtime)
    monkeypatch.setattr(
        "potpie.cli.ui.setup_ux.rich_enabled",
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
    pots: object = field(default_factory=object)
    backend: _Backend = field(init=False)
    daemon: _FakeDaemon = field(init=False)
    setup: _Setup = field(init=False)

    def __post_init__(self) -> None:
        self.backend = _Backend()
        self.daemon = _FakeDaemon(home=self.home, in_process=False)
        self.setup = _Setup()
