from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from typer.testing import CliRunner

from adapters.inbound.cli import host_cli
from adapters.inbound.cli.commands import _common, bootstrap
from adapters.inbound.cli.telemetry.onboarding_events import CliSetupAnalyticsObserver
from domain.lifecycle import PlannedSetupStep, SetupPlan, SetupPreview


runner = CliRunner()


@dataclass
class _FakeDaemon:
    home: Path
    in_process: bool = False
    calls: list[str] = field(default_factory=list)

    def start(self) -> dict[str, int | str]:
        self.calls.append("start")
        return {"pid": 123, "socket": str(self.home / "daemon.sock"), "bind": "unix:x"}

    def status(self) -> dict[str, bool | str | int]:
        self.calls.append("status")
        return {"up": True, "mode": "detached", "home": str(self.home), "pid": 123}

    def stop(self) -> dict[str, str]:
        self.calls.append("stop")
        return {"detail": "daemon stopped"}

    def logs(self, *, follow: bool = False) -> list[str]:
        self.calls.append(f"logs:{follow}")
        return ["line one"]


@dataclass
class _FakeHost:
    daemon: _FakeDaemon


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
