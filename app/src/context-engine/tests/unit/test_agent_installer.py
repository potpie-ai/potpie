"""Tests for AGENTS.md / CLAUDE.md installer helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli.agent_installer import (
    install_agent_bundle,
    iter_template_files,
    resolve_install_root,
)
from adapters.inbound.cli.main import app


def test_resolve_install_root_prefers_git_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()

    assert resolve_install_root(nested) == repo


def test_install_agent_bundle_creates_expected_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    result = install_agent_bundle(repo)

    expected = {rel.as_posix() for rel, _ in iter_template_files()}
    created = set(result.created)
    assert created == expected
    assert not result.updated
    assert not result.skipped
    assert (repo / "AGENTS.md").exists()
    assert (repo / ".agents" / "skills" / "potpie-cli" / "SKILL.md").exists()


def test_install_agent_bundle_skips_conflicting_files_without_force(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "AGENTS.md"
    target.write_text("local edits\n", encoding="utf-8")

    result = install_agent_bundle(repo)

    assert "AGENTS.md" in result.skipped
    assert target.read_text(encoding="utf-8") == "local edits\n"


def test_install_agent_bundle_overwrites_with_force(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "AGENTS.md"
    target.write_text("local edits\n", encoding="utf-8")

    result = install_agent_bundle(repo, force=True)

    assert "AGENTS.md" in result.updated
    assert "# Context Engine" in target.read_text(encoding="utf-8")


def test_init_agent_cli_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("adapters.inbound.cli.main.load_cli_env", lambda: None)
    runner = CliRunner()

    result = runner.invoke(app, ["--json", "init-agent", str(tmp_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["root"] == str(tmp_path.resolve())
    assert "AGENTS.md" in payload["created"]


# --- Claude bundle tests ---


def test_install_agent_bundle_claude_creates_claude_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    result = install_agent_bundle(repo, agent="claude")

    assert "CLAUDE.md" in result.created
    content = (repo / "CLAUDE.md").read_text(encoding="utf-8")
    assert "<!-- potpie-start -->" in content
    assert "context_resolve" in content
    assert (repo / ".claude" / "commands" / "potpie-feature.md").exists()
    assert (repo / ".claude" / "commands" / "potpie-record.md").exists()


def test_install_agent_bundle_claude_merges_into_existing_claude_md(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CLAUDE.md").write_text("# My Project\n\nExisting content.\n", encoding="utf-8")

    result = install_agent_bundle(repo, agent="claude")

    assert "CLAUDE.md" in result.created
    content = (repo / "CLAUDE.md").read_text(encoding="utf-8")
    assert "# My Project" in content
    assert "Existing content." in content
    assert "<!-- potpie-start -->" in content
    assert "context_resolve" in content


def test_install_agent_bundle_claude_unchanged_on_second_run(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    install_agent_bundle(repo, agent="claude")

    result = install_agent_bundle(repo, agent="claude")

    assert "CLAUDE.md" in result.unchanged


def test_install_agent_bundle_claude_updates_section_with_force(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CLAUDE.md").write_text(
        "# Project\n\n<!-- potpie-start -->\nOLD SECTION\n<!-- potpie-end -->\n",
        encoding="utf-8",
    )

    result = install_agent_bundle(repo, agent="claude", force=True)

    assert "CLAUDE.md" in result.updated
    content = (repo / "CLAUDE.md").read_text(encoding="utf-8")
    assert "OLD SECTION" not in content
    assert "context_resolve" in content
    assert "# Project" in content


def test_install_agent_bundle_claude_skips_changed_section_without_force(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CLAUDE.md").write_text(
        "<!-- potpie-start -->\nCUSTOM\n<!-- potpie-end -->\n",
        encoding="utf-8",
    )

    result = install_agent_bundle(repo, agent="claude")

    assert "CLAUDE.md" in result.skipped


def test_install_agent_bundle_invalid_agent_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown agent type"):
        install_agent_bundle(tmp_path, agent="unknown")


def test_init_agent_cli_claude_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("adapters.inbound.cli.main.load_cli_env", lambda: None)
    runner = CliRunner()

    result = runner.invoke(app, ["--json", "init-agent", "claude", str(tmp_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert "CLAUDE.md" in payload["created"]


def test_init_agent_cli_path_without_agent_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare path as first arg (not an agent type) should default to codex/agent bundle."""
    monkeypatch.setattr("adapters.inbound.cli.main.load_cli_env", lambda: None)
    runner = CliRunner()

    result = runner.invoke(app, ["--json", "init-agent", str(tmp_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "AGENTS.md" in payload["created"]
