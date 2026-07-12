"""Unit tests for bootstrap ``status`` host routing."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Union
from unittest.mock import MagicMock

import pytest
from potpie.cli import host_cli as cli_main
from potpie.cli.commands import bootstrap
from potpie.cli.commands._common import EXIT_DEGRADED
from typer.testing import CliRunner

from potpie.setup import (
    DONE,
    FAILED,
    PlannedSetupStep,
    SetupPlan,
    SetupPreview,
    SetupReport,
    StepResult,
    ProductStatusResult,
)

runner = CliRunner()


def _status_result(**overrides: object) -> ProductStatusResult:
    values = {
        "schema_version": "1",
        "ready": True,
        "runtime_mode": "in-process",
        "daemon_state": "up",
        "pot_id": "foo-pot",
        "pot_name": "foo",
        "backend": "embedded",
        "backend_ready": True,
        "storage_ready": True,
        "ingestion_ready": True,
        "source_count": 3,
        "skills_state": "ready",
        "setup_state": "configured",
        "issues": (),
        "recommended_next_action": None,
    }
    values.update(overrides)
    return ProductStatusResult(**values)


def _patch_status_runtime(
    monkeypatch: pytest.MonkeyPatch,
    report: ProductStatusResult,
    *,
    doctor: dict[str, object] | None = None,
) -> MagicMock:
    status = MagicMock()
    status.get.return_value = report
    status.doctor.return_value = doctor or {
        **report.to_dict(),
        "cli_install": {"on_path": True, "package_name": "potpie"},
        "daemon": {"up": report.daemon_state == "up"},
    }
    monkeypatch.setattr(
        bootstrap,
        "get_cli_runtime",
        lambda: SimpleNamespace(status=status),
    )
    return status


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


def _patch_local_setup_host(
    monkeypatch: pytest.MonkeyPatch,
    mock_host: MagicMock,
) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_EMBEDDER", raising=False)
    monkeypatch.delenv("CONTEXT_ENGINE_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(
        bootstrap,
        "get_cli_runtime",
        lambda: SimpleNamespace(
            settings=SimpleNamespace(
                data_dir=Path("/tmp/potpie-test"),
                backend=mock_host.backend.profile,
                runtime_mode=(
                    "in-process" if mock_host.daemon.in_process else "daemon"
                ),
            ),
            setup=mock_host.setup,
        ),
    )


def test_root_version_option_exits_with_cli_and_python_details() -> None:
    result = runner.invoke(cli_main.app, ["--version"])

    assert result.exit_code == 0, result.stdout
    assert "potpie " in result.stdout
    assert "potpie-context-engine " in result.stdout
    assert f"python {platform.python_version()}" in result.stdout
    assert sys.executable in result.stdout


def test_status_default_emits_host_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _status_result(
        recommended_next_action={
            "command": "potpie source add",
            "reason": "No sources are registered.",
        }
    )
    status = _patch_status_runtime(monkeypatch, report)

    result = runner.invoke(cli_main.app, ["status"])

    assert result.exit_code == 0, result.stdout
    assert "runtime=in-process" in result.stdout
    assert "daemon=up" in result.stdout
    assert "potpie source add" in result.stdout
    status.get.assert_called_once_with(pot_id=None, harness="claude")


def test_status_host_flag_remains_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    status = _patch_status_runtime(monkeypatch, _status_result())

    result = runner.invoke(cli_main.app, ["status", "--host"])

    assert result.exit_code == 0, result.stdout
    assert "runtime=in-process" in result.stdout
    status.get.assert_called_once()


def test_status_non_default_pot_triggers_host_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []

    def _integration_status(*, verify: bool = False) -> None:
        called.append(True)

    monkeypatch.setattr(
        "potpie.auth.auth_commands.integration_status",
        _integration_status,
    )
    report = _status_result(
        ready=False,
        pot_id="custom-pot",
        daemon_state="unavailable",
        backend_ready=False,
    )
    status = _patch_status_runtime(monkeypatch, report)

    result = runner.invoke(cli_main.app, ["status", "--pot", "custom-pot"])

    assert result.exit_code == 0, result.stdout
    assert called == []
    assert "daemon=unavailable" in result.stdout
    status.get.assert_called_once_with(pot_id="custom-pot", harness="claude")


def test_status_host_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_status_runtime(monkeypatch, _status_result(runtime_mode="daemon"))

    result = runner.invoke(cli_main.app, ["--json", "status"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)["data"]
    assert payload["runtime_mode"] == "daemon"
    assert payload["daemon_state"] == "up"
    assert "profile" not in payload


def test_status_verify_points_to_auth_status() -> None:
    result = runner.invoke(cli_main.app, ["--json", "status", "--verify"])

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)["error"]
    assert payload["code"] == "validation_error"
    assert (
        "potpie integration status --verify"
        in payload["recommended_next_action"]["command"]
    )


def test_doctor_json_includes_backend_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _status_result(
        ready=False,
        backend_ready=False,
        issues=("mutation store is unavailable",),
        recommended_next_action={
            "command": "potpie graph backend doctor",
            "reason": "The active engine backend is degraded.",
        },
    )
    _patch_status_runtime(monkeypatch, report)

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)["data"]
    assert payload["backend_ready"] is False
    assert payload["issues"] == ["mutation store is unavailable"]
    assert payload["recommended_next_action"]["command"] == (
        "potpie graph backend doctor"
    )


def test_setup_dry_run_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = _FakeSetupMetrics()

    mock_host = MagicMock()
    mock_host.profile = "local"
    mock_host.backend.profile = "falkordb"
    mock_host.daemon.in_process = True
    preview = SetupPreview(
        plan=SetupPlan(mode="local", host_mode="in_process", backend="falkordb"),
        steps=(
            PlannedSetupStep(
                "config",
                hard=True,
                owner="config",
                action="write config",
            ),
        ),
    )
    mock_host.setup.preview.return_value = preview

    _patch_local_setup_host(monkeypatch, mock_host)
    monkeypatch.setattr(
        "potpie.cli.ui.setup_ux.rich_enabled",
        lambda **_k: False,
    )
    monkeypatch.setattr(bootstrap, "sentry_metrics", metrics, raising=False)

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
                "host_mode": "in-process",
                "scan": False,
                "dry_run": True,
            },
        ),
    ]


def test_setup_success_emits_run_and_step_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _FakeSetupMetrics()
    steps = (
        StepResult("config", DONE, hard=True),
        StepResult("source", DONE, hard=False),
    )
    mock_host = MagicMock()
    mock_host.profile = "local"
    mock_host.backend.profile = "falkordb"
    mock_host.daemon.in_process = False

    _patch_local_setup_host(monkeypatch, mock_host)
    monkeypatch.setattr(
        bootstrap.setup_ux,
        "run_setup_plain",
        lambda _setup, plan, **_kwargs: SetupReport(plan=plan, steps=steps),
    )
    monkeypatch.setattr(
        "potpie.cli.ui.setup_ux.rich_enabled",
        lambda **_k: False,
    )
    monkeypatch.setattr(bootstrap, "sentry_metrics", metrics, raising=False)

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
    steps = (StepResult("backend.provision", FAILED, hard=True),)
    report = SetupReport(plan=SetupPlan(), steps=steps)
    assert not report.ok
    mock_host = MagicMock()
    mock_host.profile = "local"
    mock_host.backend.profile = "falkordb"
    mock_host.daemon.in_process = False

    _patch_local_setup_host(monkeypatch, mock_host)
    monkeypatch.setattr(
        bootstrap.setup_ux,
        "run_setup_plain",
        lambda _setup, plan, **_kwargs: SetupReport(plan=plan, steps=steps),
    )
    monkeypatch.setattr(
        "potpie.cli.ui.setup_ux.rich_enabled",
        lambda **_k: False,
    )
    monkeypatch.setattr(bootstrap, "sentry_metrics", metrics, raising=False)

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
    report = _status_result(backend="falkordb")
    _patch_status_runtime(monkeypatch, report)

    result = runner.invoke(cli_main.app, ["doctor"])

    assert result.exit_code == 0, result.stdout
    assert "falkordb" in result.stdout
    assert "ready=True" in result.stdout


# ---------------------------------------------------------------------------
# doctor — audit-26: effective repo pot fields
# ---------------------------------------------------------------------------


def test_doctor_reuses_flat_public_status_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _status_result(pot_id="pot-active", pot_name="project")
    _patch_status_runtime(monkeypatch, report)

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)["data"]
    for key, value in report.to_dict().items():
        assert payload[key] == (list(value) if isinstance(value, tuple) else value)
    assert "active_pot" not in payload
    assert "repo_default_pot" not in payload


def test_doctor_human_output_includes_flat_product_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_status_runtime(monkeypatch, _status_result(backend="embedded"))

    result = runner.invoke(cli_main.app, ["doctor"])

    assert result.exit_code == 0, result.stdout
    assert "runtime=in-process" in result.stdout
    assert "backend=embedded" in result.stdout
