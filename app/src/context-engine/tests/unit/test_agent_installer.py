"""Tests for AGENTS.md / .agents installer helpers."""

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
    assert (repo / ".agents" / "skills" / "context-engine-cli" / "SKILL.md").exists()


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
    assert "Potpie Agent Guide" in target.read_text(encoding="utf-8")


def test_init_agent_cli_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("adapters.inbound.cli.main.load_cli_env", lambda: None)
    runner = CliRunner()

    result = runner.invoke(app, ["--json", "init-agent", str(tmp_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["root"] == str(tmp_path.resolve())
    assert "AGENTS.md" in payload["created"]
