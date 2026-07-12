"""Exact workflow-first command tree and versioned JSON contract."""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from potpie.auth import auth_commands
from potpie.cli import host_cli
from potpie.cli.commands import _common
from scripts.generate_skill_command_manifest import collect_manifest

pytestmark = pytest.mark.unit

EXPECTED_COMMANDS = {
    "config get",
    "config list",
    "config set",
    "daemon logs",
    "daemon restart",
    "daemon service down",
    "daemon service logs",
    "daemon service status",
    "daemon service up",
    "daemon start",
    "daemon status",
    "daemon stop",
    "doctor",
    "graph backend doctor",
    "graph backend list",
    "graph backend status",
    "graph backend use",
    "graph bulk apply",
    "graph catalog",
    "graph commit",
    "graph export",
    "graph history",
    "graph import",
    "graph inbox add",
    "graph inbox claim",
    "graph inbox close",
    "graph inbox list",
    "graph inbox mark-applied",
    "graph inbox mark-rejected",
    "graph inbox show",
    "graph inspect",
    "graph mutation-template",
    "graph nudge",
    "graph propose",
    "graph quality conflicting-claims",
    "graph quality duplicate-candidates",
    "graph quality low-confidence",
    "graph quality orphan-entities",
    "graph quality projection-drift",
    "graph quality stale-facts",
    "graph quality summary",
    "graph read",
    "graph repair",
    "graph search-entities",
    "graph status",
    "integration confluence list",
    "integration confluence login",
    "integration confluence logout",
    "integration confluence select",
    "integration github list",
    "integration github login",
    "integration github logout",
    "integration jira list",
    "integration jira login",
    "integration jira logout",
    "integration jira select",
    "integration linear list",
    "integration linear login",
    "integration linear logout",
    "integration linear select",
    "integration list",
    "integration status",
    "ledger disconnect",
    "ledger pull",
    "ledger query",
    "ledger sources",
    "ledger status",
    "ledger use",
    "login",
    "logout",
    "pot archive",
    "pot create",
    "pot default",
    "pot info",
    "pot linked",
    "pot list",
    "pot rename",
    "pot reset",
    "pot use",
    "record",
    "resolve",
    "search",
    "setup",
    "skills add",
    "skills install",
    "skills list",
    "skills remove",
    "skills status",
    "skills update",
    "source add",
    "source list",
    "source remove",
    "source status",
    "status",
    "telemetry disable",
    "telemetry enable",
    "telemetry status",
    "timeline recent",
    "ui",
    "whoami",
}


def test_workflow_first_command_tree_is_exact() -> None:
    assert set(collect_manifest()["commands"]) == EXPECTED_COMMANDS


@pytest.mark.parametrize(
    "args",
    [
        ["use", "default"],
        ["cloud", "status"],
        ["backend", "status"],
        ["service", "status"],
        ["auth", "status"],
        ["github", "list"],
        ["linear", "list"],
        ["jira", "list"],
        ["confluence", "list"],
        ["graph", "mutate"],
        ["graph", "describe"],
        ["graph", "neighborhood"],
        ["pot", "default", "show"],
        ["ledger", "sources", "list"],
    ],
)
def test_removed_commands_are_unknown(
    args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        host_cli.run_cli(["--json", *args])

    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["ok"] is False
    assert payload["error"]["code"] == "usage_error"
    assert payload["meta"]["schema_version"] == "1"


def test_json_success_and_list_envelopes_are_uniform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auth_commands, "ensure_runtime_environment_loaded", lambda: None
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {"provider": provider, "authenticated": False},
    )
    result = CliRunner().invoke(host_cli.app, ["--json", "integration", "list"])

    assert result.exit_code == 0, result.stdout
    assert result.stderr == ""
    assert len([line for line in result.stdout.splitlines() if line.strip()]) == 1
    payload = json.loads(result.stdout)
    assert set(payload) == {"ok", "data", "meta"}
    assert payload["ok"] is True
    assert payload["meta"] == {
        "schema_version": "1",
        "command": "integration.list",
        "runtime_mode": "in-process",
        "request_id": None,
    }
    assert set(payload["data"]) == {"items", "count", "next_cursor"}
    assert payload["data"]["count"] == 4
    assert payload["data"]["next_cursor"] is None


def test_cli_exit_code_contract() -> None:
    assert (
        _common.EXIT_OK,
        _common.EXIT_OPERATION,
        _common.EXIT_VALIDATION,
        _common.EXIT_UNAVAILABLE,
        _common.EXIT_AUTH,
        _common.EXIT_DEGRADED,
        _common.EXIT_INTERNAL,
        _common.EXIT_INTERRUPTED,
    ) == (0, 1, 2, 3, 4, 5, 70, 130)
