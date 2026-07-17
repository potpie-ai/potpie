"""Unit tests for bootstrap ``status`` host routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import platform
import sys
from typing import Union
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from potpie.cli import main as cli_main
from potpie.cli.commands import _common, bootstrap
from potpie.cli.commands._common import EXIT_DEGRADED
from potpie.services.host_wiring import default_host_mode
from potpie_context_core.lifecycle import (
    DONE,
    FAILED,
    PlannedSetupStep,
    SetupPlan,
    SetupPreview,
    SetupReport,
    StepResult,
)
from potpie_context_core.ports.agent_context import StatusReport, StatusRequest
from potpie_context_core.ports.graph.backend import BackendCapabilities
from potpie_context_core.ports.graph.mutation import BackendReadiness

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


def _patch_local_setup_host(
    monkeypatch: pytest.MonkeyPatch,
    mock_host: MagicMock,
) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_EMBEDDER", raising=False)
    monkeypatch.delenv("CONTEXT_ENGINE_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(
        bootstrap,
        "_build_local_setup_host",
        lambda **_kwargs: (
            mock_host,
            mock_host.backend.profile,
            mock_host.daemon.in_process,
        ),
    )
    monkeypatch.setattr(bootstrap, "configured_embedder_choice", lambda: "local")
    monkeypatch.setattr(bootstrap, "configured_embedding_model", lambda: "test-model")


def test_root_version_option_exits_with_cli_and_python_details() -> None:
    result = runner.invoke(cli_main.app, ["--version"])

    assert result.exit_code == 0, result.stdout
    assert "potpie-context-engine " in result.stdout
    assert f"python {platform.python_version()}" in result.stdout
    assert sys.executable in result.stdout


def test_status_default_emits_host_report(monkeypatch: pytest.MonkeyPatch) -> None:
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

    result = runner.invoke(cli_main.app, ["status"])

    assert result.exit_code == 0, result.stdout
    assert "profile=local" in result.stdout
    assert "daemon=up" in result.stdout
    assert "potpie ingest" in result.stdout
    mock_host.agent_context.status.assert_called_once()
    req = mock_host.agent_context.status.call_args[0][0]
    assert isinstance(req, StatusRequest)
    assert req.intent == "feature"
    assert req.harness == "claude"


def test_status_host_flag_remains_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    report = StatusReport(
        pot_id="foo-pot",
        profile="local",
        daemon_up=True,
        active_pot="foo-pot",
        backend_ready=True,
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
    mock_host.agent_context.status.assert_called_once()


def test_status_non_default_pot_triggers_host_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []

    def _integration_status(*, verify: bool = False) -> None:
        called.append(True)

    monkeypatch.setattr(
        "potpie.cli.auth.auth_commands.integration_status",
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

    result = runner.invoke(cli_main.app, ["--json", "status"])

    assert result.exit_code == 0, result.stdout
    assert '"profile": "managed"' in result.stdout
    assert '"daemon_up": true' in result.stdout


def test_status_verify_points_to_auth_status() -> None:
    result = runner.invoke(cli_main.app, ["--json", "status", "--verify"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "validation_error"
    assert "potpie auth status --verify" in payload["recommended_next_action"]


def test_doctor_json_includes_backend_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pot:
        pot_id = "foo-pot"

    class _LedgerStatus:
        available = True
        binding = "none"

    mock_host = MagicMock()
    mock_host.daemon.status.return_value = {"mode": "in_process"}
    mock_host.backend.profile = "memory"
    mock_host.backend.capabilities.return_value = BackendCapabilities(
        profile="memory",
        mutation=True,
        claim_query=True,
    )
    mock_host.backend.mutation.readiness.return_value = BackendReadiness(
        profile="memory",
        ready=False,
        detail="mutation store is unavailable",
        capability_ready={"mutation": False},
    )
    mock_host.pots.active_pot.return_value = _Pot()
    mock_host.ledger.status.return_value = _LedgerStatus()
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["backend_ready"] is False
    assert payload["backend_readiness"]["detail"] == "mutation store is unavailable"
    assert payload["backend_readiness"]["capability_ready"] == {"mutation": False}
    assert payload["active_pot"] == "foo-pot"
    assert "graph status" in payload["recommended_next_action"]


def test_default_host_mode_rejects_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "deamon")

    with pytest.raises(ValueError, match="CONTEXT_ENGINE_HOST_MODE"):
        default_host_mode()


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
    mock_host.backend.mutation.readiness.return_value = BackendReadiness(
        profile="falkordb",
        ready=True,
        capability_ready={"mutation": True},
    )
    mock_host.pots.active_pot.return_value = None
    mock_host.daemon.status.return_value = {"mode": "in_process", "up": True}
    mock_host.ledger.status.return_value = MagicMock(available=True, binding="local")

    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)

    result = runner.invoke(cli_main.app, ["doctor"])

    assert result.exit_code == 0, result.stdout
    assert "falkordb" in result.stdout
    assert "graph.read" in result.stdout


# ---------------------------------------------------------------------------
# doctor — audit-26: effective repo pot fields
# ---------------------------------------------------------------------------


def _make_doctor_host(
    *,
    active_pot_id: str | None,
    repo_default: str | None,
) -> MagicMock:
    class _Pot:
        def __init__(self, pid: str) -> None:
            self.pot_id = pid

    class _LedgerStatus:
        available = True
        binding = "none"

    mock_host = MagicMock()
    mock_host.backend.profile = "memory"
    mock_host.backend.capabilities.return_value = BackendCapabilities(
        profile="memory", mutation=True, claim_query=True
    )
    mock_host.backend.mutation.readiness.return_value = BackendReadiness(
        profile="memory", ready=True, capability_ready={"mutation": True}
    )
    mock_host.daemon.status.return_value = {"mode": "in_process", "up": True}
    mock_host.ledger.status.return_value = _LedgerStatus()
    mock_host.pots.active_pot.return_value = (
        _Pot(active_pot_id) if active_pot_id else None
    )
    mock_host.pots.repo_default.return_value = repo_default
    known_pot_ids = {pid for pid in (active_pot_id, repo_default) if pid}
    mock_host.pots.list_pots.return_value = [_Pot(pid) for pid in sorted(known_pot_ids)]
    return mock_host


def test_doctor_json_includes_effective_and_default_repo_pot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """doctor JSON includes effective_current_repo_pot and repo_default_pot."""
    mock_host = _make_doctor_host(
        active_pot_id="pot-active", repo_default="pot-default"
    )
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        bootstrap, "current_repo_identity_for_cli", lambda: "github.com/acme/shop"
    )

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["active_pot"] == "pot-active"
    assert payload["repo_default_pot"] == "pot-default"
    # effective uses repo default when set
    assert payload["effective_current_repo_pot"] == "pot-default"


def test_doctor_json_effective_falls_back_to_active_when_no_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no repo default is set, effective_current_repo_pot equals active_pot."""
    mock_host = _make_doctor_host(active_pot_id="pot-active", repo_default=None)
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        bootstrap, "current_repo_identity_for_cli", lambda: "github.com/acme/shop"
    )

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["repo_default_pot"] is None
    assert payload["effective_current_repo_pot"] == "pot-active"


def test_doctor_json_effective_prefers_single_linked_repo_pot_over_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When exactly one linked repo pot exists, doctor matches CLI resolution."""

    class _NamedPot:
        def __init__(self, pid: str, name: str) -> None:
            self.pot_id = pid
            self.name = name

    class _RepoSource:
        kind = "repo"
        name = "github.com/acme/shop"
        location = "github.com/acme/shop"

    mock_host = _make_doctor_host(
        active_pot_id="pot-active",
        repo_default=None,
    )
    mock_host.graph.data_plane_status.side_effect = AssertionError(
        "doctor effective-pot lookup should not fetch graph counts"
    )
    mock_host.pots.list_pots.return_value = [
        _NamedPot("pot-active", "active"),
        _NamedPot("pot-linked", "linked"),
    ]
    mock_host.pots.list_sources.side_effect = lambda *, pot_id: (
        [_RepoSource()] if pot_id == "pot-linked" else []
    )
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        bootstrap, "current_repo_identity_for_cli", lambda: "github.com/acme/shop"
    )
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["repo_default_pot"] is None
    assert payload["active_pot"] == "pot-active"
    assert payload["effective_current_repo_pot"] == "pot-linked"


def test_doctor_json_no_repo_identity_leaves_effective_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside a git repo, effective_current_repo_pot and repo_default_pot are None."""
    mock_host = _make_doctor_host(active_pot_id="pot-active", repo_default=None)
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(bootstrap, "current_repo_identity_for_cli", lambda: None)

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["repo_default_pot"] is None
    assert payload["effective_current_repo_pot"] is None


def test_doctor_human_output_includes_repo_line_when_in_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain-text doctor output includes a repo → effective-pot line."""
    mock_host = _make_doctor_host(
        active_pot_id="pot-active", repo_default="pot-default"
    )
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        bootstrap, "current_repo_identity_for_cli", lambda: "github.com/acme/shop"
    )

    result = runner.invoke(cli_main.app, ["doctor"])

    assert result.exit_code == 0, result.stdout
    assert "github.com/acme/shop" in result.stdout
    assert "pot-default" in result.stdout
