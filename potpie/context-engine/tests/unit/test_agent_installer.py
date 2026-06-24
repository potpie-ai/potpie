"""Tests for AGENTS.md / CLAUDE.md installer helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.outbound.skills.agent_installer import (
    install_global_agent_instructions,
    install_agent_bundle,
    install_skill_bundle,
    iter_template_files,
    resolve_install_root,
)
from adapters.outbound.skills.claude_target import FileBackedAgentTarget
from application.services.skill_manager import DefaultSkillManager
from adapters.outbound.skills.bundle_catalog import catalog_by_id


def test_resolve_install_root_prefers_git_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()

    assert resolve_install_root(nested) == repo


def test_install_agent_bundle_creates_expected_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    result = install_agent_bundle(repo)

    expected = {rel.as_posix() for rel, _ in iter_template_files()}
    created = set(result.created)
    assert created == expected
    assert not result.updated
    assert not result.skipped
    assert (repo / "AGENTS.md").exists()
    assert (repo / ".agents" / "skills" / "potpie-cli" / "SKILL.md").exists()


def test_packaged_skill_names_match_directories_and_catalog() -> None:
    catalog = catalog_by_id()
    for rel_path, _content in iter_template_files():
        if rel_path.name != "SKILL.md":
            continue
        skill_id = rel_path.parent.name
        assert skill_id in catalog
        assert catalog[skill_id].id == skill_id
        assert catalog[skill_id].version
        assert catalog[skill_id].description


def test_install_skill_bundle_writes_to_global_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"

    result = install_skill_bundle(root, skill_ids=("potpie-cli",))

    assert "potpie-cli/SKILL.md" in result.created
    assert (root / "potpie-cli" / "SKILL.md").exists()
    assert {path.name for path in root.iterdir()} == {"potpie-cli"}


def test_install_global_agent_instructions_merges_compact_agents_md(
    tmp_path: Path,
) -> None:
    root = tmp_path / "codex"
    root.mkdir()
    target = root / "AGENTS.md"
    target.write_text("# Personal defaults\n", encoding="utf-8")

    result = install_global_agent_instructions(root, agent="codex")

    text = target.read_text(encoding="utf-8")
    managed = text.split("<!-- potpie-start -->", 1)[1].split(
        "<!-- potpie-end -->", 1
    )[0]
    assert result.created == ["AGENTS.md"]
    assert "# Personal defaults" in text
    assert "Potpie is durable project memory" in text
    assert "potpie --json source list" in text
    assert len([line for line in managed.splitlines() if line.strip()]) <= 6

    rerun = install_global_agent_instructions(root, agent="codex")

    assert rerun.unchanged == ["AGENTS.md"]


def test_install_global_agent_instructions_updates_managed_claude_section(
    tmp_path: Path,
) -> None:
    root = tmp_path / "claude"
    root.mkdir()
    target = root / "CLAUDE.md"
    target.write_text(
        "# Personal defaults\n\n<!-- potpie-start -->\nold\n<!-- potpie-end -->\n",
        encoding="utf-8",
    )

    result = install_global_agent_instructions(root, agent="claude")

    text = target.read_text(encoding="utf-8")
    assert result.updated == ["CLAUDE.md"]
    assert "# Personal defaults" in text
    assert "old" not in text
    assert "Potpie is durable project memory" in text


def test_file_backed_target_installs_global_support_files(tmp_path: Path) -> None:
    target = FileBackedAgentTarget(
        agent="codex",
        skills_root=tmp_path / ".agents" / "skills",
        instructions_root=tmp_path / ".codex",
        instructions_agent="codex",
        home=tmp_path / "potpie",
    )

    target.install_support_files()

    text = (tmp_path / ".codex" / "AGENTS.md").read_text(encoding="utf-8")
    assert "Potpie is durable project memory" in text


def test_skill_manager_repairs_support_files_when_skill_is_current(
    tmp_path: Path,
) -> None:
    catalog = catalog_by_id()
    version = catalog["potpie-cli"].version
    calls: list[str | None] = []

    class _Target:
        agent = "codex"
        skills_root = tmp_path / ".agents" / "skills"

        def installed(self) -> dict[str, str]:
            return {"potpie-cli": version}

        def install(
            self, *, skill_id: str, version: str, path: str | None = None
        ) -> None:
            raise AssertionError("current skill should not be reinstalled")

        def install_support_files(self, *, path: str | None = None) -> None:
            calls.append(path)

        def remove(self, *, skill_id: str) -> None:
            raise AssertionError("remove should not be called")

    manager = DefaultSkillManager(targets={"codex": _Target()})

    result = manager.install(agent="codex", skill_id="potpie-cli")

    assert result.changed == ()
    assert calls == [None]


def test_install_agent_bundle_skips_conflicting_files_without_force(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "AGENTS.md"
    target.write_text("local edits\n", encoding="utf-8")

    result = install_agent_bundle(repo)

    assert "AGENTS.md" in result.skipped
    assert target.read_text(encoding="utf-8") == "local edits\n"


def test_install_agent_bundle_overwrites_with_force(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "AGENTS.md"
    target.write_text("local edits\n", encoding="utf-8")

    result = install_agent_bundle(repo, force=True)

    assert "AGENTS.md" in result.updated
    assert "# Context Engine" in target.read_text(encoding="utf-8")


# --- Claude bundle tests ---


def test_install_agent_bundle_claude_creates_claude_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    result = install_agent_bundle(repo, agent="claude")

    assert "CLAUDE.md" in result.created
    content = (repo / "CLAUDE.md").read_text(encoding="utf-8")
    assert "<!-- potpie-start -->" in content
    assert "context_resolve" in content
    assert (repo / ".claude" / "commands" / "potpie-feature.md").exists()
    assert (repo / ".claude" / "commands" / "potpie-record.md").exists()
    assert (repo / ".claude" / "skills" / "potpie-cli" / "SKILL.md").exists()


def test_install_agent_bundle_claude_merges_into_existing_claude_md(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "CLAUDE.md").write_text(
        "# My Project\n\nExisting content.\n", encoding="utf-8"
    )

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
    (repo / ".git").mkdir()
    install_agent_bundle(repo, agent="claude")

    result = install_agent_bundle(repo, agent="claude")

    assert "CLAUDE.md" in result.unchanged


def test_install_agent_bundle_claude_updates_section_with_force(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
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


def test_install_agent_bundle_claude_skips_changed_section_without_force(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "CLAUDE.md").write_text(
        "<!-- potpie-start -->\nCUSTOM\n<!-- potpie-end -->\n",
        encoding="utf-8",
    )

    result = install_agent_bundle(repo, agent="claude")

    assert "CLAUDE.md" in result.skipped


def test_install_agent_bundle_claude_plugin_lays_out_plugin_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    result = install_agent_bundle(repo, agent="claude-plugin")

    base = repo / ".claude" / "potpie-plugin"
    assert (base / ".claude-plugin" / "plugin.json").exists()
    assert (base / "hooks" / "hooks.json").exists()
    assert (base / "hooks" / "potpie_nudge.py").exists()
    assert (base / "skills" / "potpie-graph" / "SKILL.md").exists()
    # Everything is created on a fresh repo; nothing skipped.
    assert ".claude/potpie-plugin/.claude-plugin/plugin.json" in result.created
    assert not result.skipped


def test_install_agent_bundle_cursor_writes_cursor_skills(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    try:
        result = install_agent_bundle(repo, agent="cursor")
    except PermissionError as exc:
        if ".cursor" in str(exc):
            pytest.skip("sandbox blocks writing .cursor directories")
        raise

    assert "AGENTS.md" in result.created
    skill = repo / ".cursor" / "skills" / "potpie-cli" / "SKILL.md"
    assert skill.exists()
    assert "potpie" in skill.read_text(encoding="utf-8").lower()


def test_install_agent_bundle_opencode_writes_opencode_skills(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    result = install_agent_bundle(repo, agent="opencode")

    assert "AGENTS.md" not in result.created
    skill = repo / ".opencode" / "skills" / "potpie-graph" / "SKILL.md"
    assert skill.exists()


def test_install_agent_bundle_invalid_agent_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown agent type"):
        install_agent_bundle(tmp_path, agent="unknown")
