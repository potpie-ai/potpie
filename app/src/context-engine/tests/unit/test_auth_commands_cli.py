"""CLI command coverage for Jira/Confluence auth subcommands (mocked)."""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from adapters.inbound.cli import auth_commands
from adapters.inbound.cli.atlassian_read import AtlassianReadError

runner = CliRunner()


def _mock_cli(monkeypatch: pytest.MonkeyPatch, *, json_mode: bool = False) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (json_mode, False))


def test_print_wiki_row_pages_and_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(
        auth_commands,
        "_print_remote_line",
        lambda message: lines.append(message),
    )

    auth_commands._print_wiki_row(
        {
            "title": "Page [bold]",
            "space": "DOCS",
            "status": "current",
            "created": "2024-01-01",
            "created_by": "Ada",
            "updated": "2024-02-01",
            "updated_by": "Bob",
            "excerpt": "Hello",
            "url": "https://x.com/p",
        },
        pages=True,
    )
    assert any("Page" in line for line in lines)
    assert any("DOCS" in line for line in lines)


def test_run_product_use_result_human(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cli(monkeypatch, json_mode=False)
    lines: list[str] = []
    monkeypatch.setattr(auth_commands, "print_plain_line", lambda m, **k: lines.append(m))
    monkeypatch.setattr(auth_commands, "_print_remote_line", lambda m: lines.append(m))

    auth_commands._run_product_use_result(
        {
            "product": "confluence",
            "workspace_key": "DOCS",
            "workspace_name": "Documentation",
            "items": [{"title": "Intro", "space": "DOCS", "url": "https://x.com"}],
        },
        product_label="Confluence",
    )
    assert any("Documentation" in line for line in lines)
    assert any("Intro" in line for line in lines)


def test_auth_status_verify_token_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cli(monkeypatch, json_mode=True)
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {"provider": provider, "authenticated": True, "auth_type": "api_token"},
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_tokens",
        lambda _p: (_ for _ in ()).throw(RuntimeError("token load broke")),
    )
    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])
    assert result.exit_code == 0
    assert '"verified": false' in result.stdout
    assert "token load broke" in result.stdout


def test_jira_issues_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cli(monkeypatch, json_mode=False)
    monkeypatch.setattr(
        auth_commands,
        "fetch_jira_issues_sample",
        lambda **_kwargs: (_ for _ in ()).throw(AtlassianReadError("not connected")),
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )
    with pytest.raises(typer.Exit):
        auth_commands.jira_issues(limit=5)
    assert captured
    assert "Jira read failed" in captured[0][0]
