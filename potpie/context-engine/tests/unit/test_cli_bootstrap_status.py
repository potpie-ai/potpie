"""Unit tests for bootstrap ``status`` host routing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli as cli_main
from adapters.inbound.cli.commands import bootstrap
from domain.ports.agent_context import StatusReport, StatusRequest

runner = CliRunner()


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
    monkeypatch.setattr(bootstrap, "resolve_pot_id", lambda _host, pot: pot or "foo-pot")

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


def test_setup_dry_run_preview(monkeypatch: pytest.MonkeyPatch) -> None:
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

    result = runner.invoke(cli_main.app, ["setup", "--dry-run"])

    assert result.exit_code == 0, result.stdout
    mock_host.setup.preview.assert_called_once()
    assert "config" in result.stdout or "steps" in result.stdout


def test_doctor_emits_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_caps = MagicMock()
    mock_caps.implemented.return_value = ["graph.read", "graph.write"]
    mock_host = MagicMock()
    mock_host.backend.profile = "falkordb"
    mock_host.backend.capabilities.return_value = mock_caps
    mock_host.daemon.status.return_value = {"mode": "in_process", "up": True}
    mock_host.ledger.status.return_value = MagicMock(
        available=True, binding="local"
    )

    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)

    result = runner.invoke(cli_main.app, ["doctor"])

    assert result.exit_code == 0, result.stdout
    assert "falkordb" in result.stdout
    assert "graph.read" in result.stdout
