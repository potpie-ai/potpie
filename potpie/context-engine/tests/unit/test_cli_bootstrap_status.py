"""Unit tests for bootstrap ``status`` host routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli as cli_main
from adapters.inbound.cli.commands import bootstrap
from adapters.inbound.cli.commands._common import EXIT_DEGRADED
from bootstrap.host_wiring import default_host_mode
from domain.lifecycle import DONE, FAILED, SetupPlan, SetupReport, StepResult
from domain.ports.agent_context import StatusReport, StatusRequest

runner = CliRunner()


@dataclass(frozen=True)
class _MetricCall:
    name: str
    attributes: dict[str, Union[str, bool]]


class _FakeSetupMetrics:
    def __init__(self) -> None:
        self.calls: list[_MetricCall] = []

    def count(
        self,
        name: str,
        *,
        attributes: dict[str, Union[str, bool]] | None = None,
    ) -> None:
        self.calls.append(_MetricCall(name, {} if attributes is None else attributes))


def test_status_host_path_emits_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = StatusReport(
        pot_id="foo-pot",
        profile="local",
        daemon_up=True,
        active_pot="foo-pot",
        backend_ready=True,
        data_plane={"counts": {"nodes": 3}},
        recommended_next_action="potpie ingest",
    )
    mock_host = MagicMock()
    mock_host.agent_context.status.return_value = report

    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        bootstrap, "resolve_pot_id", lambda _host, pot: pot or "foo-pot"
    )

    result = runner.invoke(cli_main.app, ["status", "--host"])

    assert result.exit_code == 0, result.stdout
    assert "profile=local" in result.stdout
    assert "daemon=up" in result.stdout
    assert "potpie ingest" in result.stdout
    mock_host.agent_context.status.assert_called_once()
    req = mock_host.agent_context.status.call_args[0][0]
    assert isinstance(req, StatusRequest)
    assert req.intent == "feature"
    assert req.harness == "claude"


def test_status_non_default_pot_triggers_host_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []

    def _integration_status(*, verify: bool = False) -> None:
        called.append(True)

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.auth_commands.integration_status",
        _integration_status,
    )
    report = StatusReport(
        pot_id="custom-pot",
        profile="local",
        daemon_up=False,
        active_pot="custom-pot",
        backend_ready=False,
    )
    mock_host = MagicMock()
    mock_host.agent_context.status.return_value = report
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        bootstrap,
        "resolve_pot_id",
        lambda _host, pot: pot or "custom-pot",
    )

    result = runner.invoke(cli_main.app, ["status", "--pot", "custom-pot"])

    assert result.exit_code == 0, result.stdout
    assert called == []
    assert "daemon=down" in result.stdout


def test_status_host_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    report = StatusReport(
        pot_id="foo-pot",
        profile="managed",
        daemon_up=True,
        active_pot="foo-pot",
        backend_ready=True,
    )
    mock_host = MagicMock()
    mock_host.agent_context.status.return_value = report
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(bootstrap, "resolve_pot_id", lambda _host, pot: "foo-pot")

    result = runner.invoke(cli_main.app, ["--json", "status", "--host"])

    assert result.exit_code == 0, result.stdout
    assert '"profile": "managed"' in result.stdout
    assert '"daemon_up": true' in result.stdout


def test_default_host_mode_rejects_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "deamon")

    with pytest.raises(ValueError, match="CONTEXT_ENGINE_HOST_MODE"):
        default_host_mode()


def test_setup_dry_run_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = _FakeSetupMetrics()
    preview = MagicMock()
    preview.to_dict.return_value = {"steps": [{"name": "config", "status": "pending"}]}

    mock_host = MagicMock()
    mock_host.profile = "local"
    mock_host.backend.profile = "falkordb"
    mock_host.daemon.in_process = True
    mock_host.setup.preview.return_value = preview

    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        "adapters.inbound.cli.ui.setup_ux.rich_enabled",
        lambda **_k: False,
    )
    monkeypatch.setattr(bootstrap, "sentry_metrics_runtime", metrics, raising=False)

    result = runner.invoke(cli_main.app, ["setup", "--dry-run"])

    assert result.exit_code == 0, result.stdout
    mock_host.setup.preview.assert_called_once()
    mock_host.setup.run.assert_not_called()
    assert "config" in result.stdout or "steps" in result.stdout
    assert metrics.calls == [
        _MetricCall(
            "ce.setup.runs_total",
            {
                "result": "dry_run",
                "backend": "falkordb",
                "host_mode": "in_process",
                "scan": False,
                "dry_run": True,
            },
        ),
    ]


def test_setup_success_emits_run_and_step_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _FakeSetupMetrics()
    plan = SetupPlan(
        mode="local",
        host_mode="daemon",
        backend="falkordb",
        repo="/private/project",
        pot="customer-pot",
        agent="gpt-9",
        scan=True,
    )
    report = SetupReport(
        plan=plan,
        steps=(
            StepResult("config", DONE, hard=True),
            StepResult("source", DONE, hard=False),
        ),
    )
    mock_host = MagicMock()
    mock_host.profile = "local"
    mock_host.backend.profile = "falkordb"
    mock_host.daemon.in_process = False
    mock_host.setup.run.return_value = report

    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        "adapters.inbound.cli.ui.setup_ux.rich_enabled",
        lambda **_k: False,
    )
    monkeypatch.setattr(bootstrap, "sentry_metrics_runtime", metrics, raising=False)

    result = runner.invoke(
        cli_main.app,
        [
            "setup",
            "--repo",
            "/private/project",
            "--pot",
            "customer-pot",
            "--agent",
            "gpt-9",
            "--scan",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert metrics.calls == [
        _MetricCall(
            "ce.setup.runs_total",
            {
                "result": "ok",
                "backend": "falkordb",
                "host_mode": "daemon",
                "scan": True,
                "dry_run": False,
            },
        ),
        _MetricCall(
            "ce.setup.step_total",
            {"step": "config", "state": "done", "hard": True},
        ),
        _MetricCall(
            "ce.setup.step_total",
            {"step": "source", "state": "done", "hard": False},
        ),
    ]
    for call in metrics.calls:
        assert "repo" not in call.attributes
        assert "pot" not in call.attributes
        assert "agent" not in call.attributes
        assert "/private/project" not in call.attributes.values()
        assert "customer-pot" not in call.attributes.values()
        assert "gpt-9" not in call.attributes.values()


def test_setup_degraded_report_preserves_exit_code_and_emits_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _FakeSetupMetrics()
    plan = SetupPlan(mode="local", host_mode="daemon", backend="falkordb")
    report = SetupReport(
        plan=plan,
        steps=(StepResult("backend.provision", FAILED, hard=True),),
    )
    assert not report.ok
    mock_host = MagicMock()
    mock_host.profile = "local"
    mock_host.backend.profile = "falkordb"
    mock_host.daemon.in_process = False
    mock_host.setup.run.return_value = report

    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        "adapters.inbound.cli.ui.setup_ux.rich_enabled",
        lambda **_k: False,
    )
    monkeypatch.setattr(bootstrap, "sentry_metrics_runtime", metrics, raising=False)

    result = runner.invoke(cli_main.app, ["setup", "--yes"])

    assert result.exit_code == EXIT_DEGRADED, result.stdout
    assert metrics.calls == [
        _MetricCall(
            "ce.setup.runs_total",
            {
                "result": "degraded",
                "backend": "falkordb",
                "host_mode": "daemon",
                "scan": False,
                "dry_run": False,
            },
        ),
        _MetricCall(
            "ce.setup.step_total",
            {"step": "backend.provision", "state": "failed", "hard": True},
        ),
    ]


def test_doctor_emits_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_caps = MagicMock()
    mock_caps.implemented.return_value = ["graph.read", "graph.write"]
    mock_host = MagicMock()
    mock_host.backend.profile = "falkordb"
    mock_host.backend.capabilities.return_value = mock_caps
    mock_host.daemon.status.return_value = {"mode": "in_process", "up": True}
    mock_host.ledger.status.return_value = MagicMock(available=True, binding="local")

    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)

    result = runner.invoke(cli_main.app, ["doctor"])

    assert result.exit_code == 0, result.stdout
    assert "falkordb" in result.stdout
    assert "graph.read" in result.stdout
