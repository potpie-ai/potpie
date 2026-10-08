"""Tests for AGENTS.md / CLAUDE.md installer helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import potpie.skills.installer as agent_installer
from potpie.skills.bundle import bundle_files, routing_block
from potpie.skills.catalog import catalog_by_id
from potpie.skills.errors import InvalidSkillsInstallPathError, UnknownAgentTargetError
from potpie.skills.harnesses import AGENT_TYPES, HARNESS_LAYOUTS, harness_layout
from potpie.skills.installer import (
    install_agent_bundle,
    install_bundle,
    iter_template_files,
    resolve_install_root,
)
from potpie.skills.manager import DefaultSkillManager
from potpie.skills.targets import AgentTarget


def test_resolve_install_root_prefers_git_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()

    assert resolve_install_root(nested) == repo


def test_resolve_install_root_uses_a_typed_error_for_file_paths(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.touch()

    with pytest.raises(InvalidSkillsInstallPathError):
        resolve_install_root(file_path)


def test_skill_manager_uses_a_typed_error_for_unknown_agent_targets() -> None:
    with pytest.raises(UnknownAgentTargetError):
        DefaultSkillManager().install(agent="missing")


def test_install_agent_bundle_creates_expected_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    result = install_agent_bundle(repo)

    expected = {rel.as_posix() for rel, _ in iter_template_files()} | {"AGENTS.md"}
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


def test_install_bundle_writes_selected_skills_to_a_skills_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"

    result = install_bundle(root, skill_ids=("potpie-cli",))

    assert "potpie-cli/SKILL.md" in result.created
    assert (root / "potpie-cli" / "SKILL.md").exists()
    assert {path.name for path in root.iterdir()} == {"potpie-cli"}


def test_install_bundle_with_no_skills_writes_only_the_routing_block(
    tmp_path: Path,
) -> None:
    result = install_bundle(tmp_path, skill_ids=(), instructions="AGENTS.md")

    assert result.created == ["AGENTS.md"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AGENTS.md"]


def test_routing_block_merges_into_compact_global_agents_md(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    root.mkdir()
    target = root / "AGENTS.md"
    target.write_text("# Personal defaults\n", encoding="utf-8")

    result = install_bundle(root, skill_ids=(), instructions="AGENTS.md")

    text = target.read_text(encoding="utf-8")
    managed = text.split("<!-- potpie-start -->", 1)[1].split("<!-- potpie-end -->", 1)[
        0
    ]
    assert result.updated == ["AGENTS.md"]
    assert "# Personal defaults" in text
    assert "Potpie is durable project memory" in text
    # The health check the block prescribes is the one-call `potpie status`.
    assert "potpie status" in text
    # A budget, not a line count. This block is prepended to the global
    # instructions of every agent on the machine, so what it costs is context
    # every single turn pays for -- and the person paying it never chose it.
    # Measured in characters because line count is a wrapping artifact: reflowing
    # the same prose to a wider column would have "fixed" the old `<= 6` bound
    # without removing a word, and adding a paragraph the block genuinely wanted
    # broke it. Raise this deliberately, having decided the words earn their keep
    # in a file the reader did not write.
    assert len(managed.strip()) <= 950

    rerun = install_bundle(root, skill_ids=(), instructions="AGENTS.md")

    assert rerun.unchanged == ["AGENTS.md"]


def test_global_instructions_have_one_canonical_source() -> None:
    bundle = dict(bundle_files("routing"))
    assert set(path.as_posix() for path in bundle) == {"POTPIE.md"}


def test_routing_block_replaces_the_managed_claude_section(tmp_path: Path) -> None:
    root = tmp_path / "claude"
    root.mkdir()
    target = root / "CLAUDE.md"
    target.write_text(
        "# Personal defaults\n\n<!-- potpie-start -->\nold\n<!-- potpie-end -->\n",
        encoding="utf-8",
    )

    result = install_bundle(root, skill_ids=(), instructions="CLAUDE.md")

    text = target.read_text(encoding="utf-8")
    assert result.updated == ["CLAUDE.md"]
    assert "# Personal defaults" in text
    assert "old" not in text
    assert "Potpie is durable project memory" in text


def test_global_target_installs_the_routing_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(tmp_path))
    target = AgentTarget(agent="codex", home=tmp_path / "potpie")

    result = target.install_instructions()

    assert result is not None and result.created == ["AGENTS.md"]
    text = (tmp_path / ".codex" / "AGENTS.md").read_text(encoding="utf-8")
    assert "Potpie is durable project memory" in text


# Every harness's paths at both scopes, as the table has always placed them.
_GLOBAL_PATHS = {
    "claude": (".claude/skills", ".claude/CLAUDE.md"),
    "codex": (".agents/skills", ".codex/AGENTS.md"),
    "cursor": (".cursor/skills", None),
    "opencode": (".config/opencode/skills", None),
}
_PROJECT_PATHS = {
    "claude": (".claude/skills", "CLAUDE.md"),
    "codex": (".agents/skills", "AGENTS.md"),
    "cursor": (".cursor/skills", "AGENTS.md"),
    "opencode": (".opencode/skills", None),
}


def test_the_layout_table_covers_every_harness_and_the_default_alias() -> None:
    assert set(HARNESS_LAYOUTS) == set(_GLOBAL_PATHS) == set(_PROJECT_PATHS)
    assert AGENT_TYPES == ("default", "codex", "claude", "cursor", "opencode")
    assert harness_layout("default") is HARNESS_LAYOUTS["codex"]
    assert harness_layout(" Claude ") is HARNESS_LAYOUTS["claude"]


@pytest.mark.parametrize("agent", sorted(_GLOBAL_PATHS))
def test_global_target_writes_where_the_harness_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent: str
) -> None:
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(tmp_path / "h"))
    skills, instructions = _GLOBAL_PATHS[agent]
    target = AgentTarget(agent=agent, home=tmp_path / "potpie")

    target.install(skill_id="potpie-cli", version="1")
    written = target.install_instructions()

    assert target.skills_root == tmp_path / "h" / skills
    assert target.target_root == target.skills_root
    assert (tmp_path / "h" / skills / "potpie-cli" / "SKILL.md").is_file()
    if instructions is None:
        assert written is None
    else:
        assert written is not None and written.created == [Path(instructions).name]
        assert (tmp_path / "h" / instructions).is_file()


@pytest.mark.parametrize("agent", sorted(_PROJECT_PATHS))
def test_project_target_writes_where_the_harness_reads(
    tmp_path: Path, agent: str
) -> None:
    repo = tmp_path / "repo"
    nested = repo / "pkg"
    (repo / ".git").mkdir(parents=True)
    nested.mkdir()
    skills, instructions = _PROJECT_PATHS[agent]
    target = AgentTarget(
        agent=agent, scope="project", path=nested, home=tmp_path / "potpie"
    )

    target.install(skill_id="potpie-cli", version="1")
    written = target.install_instructions()

    assert target.target_root == repo
    assert target.skills_root == repo / skills
    assert (repo / skills / "potpie-cli" / "SKILL.md").is_file()
    if instructions is None:
        assert written is None
        assert sorted(p.name for p in repo.iterdir()) == [".git", ".opencode", "pkg"]
    else:
        assert written is not None and written.created == [instructions]
        assert (repo / instructions).is_file()


def test_unknown_harness_has_no_target() -> None:
    with pytest.raises(ValueError, match="Unknown agent type"):
        AgentTarget(agent="clawd", scope="project")
    with pytest.raises(ValueError, match="scope must be"):
        AgentTarget(agent="claude", scope="workspace")


def test_skill_manager_repairs_support_files_when_skill_is_current(
    tmp_path: Path,
) -> None:
    catalog = catalog_by_id()
    current = {sid: info.version for sid, info in catalog.items()}
    calls: list[str | None] = []

    class _Target:
        agent = "codex"
        skills_root = tmp_path / ".agents" / "skills"

        def installed(self) -> dict[str, str]:
            return dict(current)

        def install(
            self, *, skill_id: str, version: str, path: str | None = None
        ) -> None:
            raise AssertionError("current skill should not be reinstalled")

        def install_instructions(self, *, path: str | None = None) -> None:
            calls.append(path)

        def remove(self, *, skill_id: str) -> None:
            raise AssertionError("remove should not be called")

    manager = DefaultSkillManager(targets={"codex": _Target()})

    # The sweep still repairs them — that is the command that owns the harness's
    # own files. Naming one skill no longer does; see
    # ``test_installing_one_named_skill_does_not_touch_the_instruction_file``.
    result = manager.install(agent="codex")

    assert result.changed == ()
    assert calls == [None]


def test_install_agent_bundle_merges_existing_agents_md_without_force(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "AGENTS.md"
    target.write_text("local edits\n", encoding="utf-8")

    result = install_agent_bundle(repo)

    text = target.read_text(encoding="utf-8")
    assert "AGENTS.md" in result.updated
    assert "local edits" in text
    assert "<!-- potpie-start -->" in text
    assert "Potpie is durable project memory" in text


def test_install_agent_bundle_does_not_overwrite_agents_md_with_force(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "AGENTS.md"
    target.write_text("local edits\n", encoding="utf-8")

    result = install_agent_bundle(repo, force=True)

    text = target.read_text(encoding="utf-8")
    assert "AGENTS.md" in result.updated
    assert "local edits" in text
    assert "<!-- potpie-start -->" in text
    assert "Potpie is durable project memory" in text


def test_install_agent_bundle_wraps_old_unmarked_agents_md(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "AGENTS.md"
    marked_template = routing_block()
    old_unmarked = (
        marked_template.split("\n", 1)[1].rsplit("\n<!-- potpie-end -->", 1)[0].strip()
        + "\n"
    )
    target.write_text(old_unmarked, encoding="utf-8")

    result = install_agent_bundle(repo, force=True)

    text = target.read_text(encoding="utf-8")
    assert "AGENTS.md" in result.updated
    assert text.count("Potpie is durable project memory") == 1
    assert "<!-- potpie-start -->" in text


def test_install_agent_bundle_replaces_embedded_unmarked_agents_md(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "AGENTS.md"
    marked_template = routing_block()
    old_unmarked = (
        marked_template.split("\n", 1)[1].rsplit("\n<!-- potpie-end -->", 1)[0].strip()
    )
    target.write_text(
        "# My notes\n\nSome custom project instructions.\n\n"
        f"{old_unmarked}\n\n"
        "## Team notes\n\nKeep these too.\n",
        encoding="utf-8",
    )

    result = install_agent_bundle(repo)

    text = target.read_text(encoding="utf-8")
    assert "AGENTS.md" in result.updated
    assert "# My notes" in text
    assert "Some custom project instructions." in text
    assert "## Team notes" in text
    assert "Keep these too." in text
    assert text.count("Potpie is durable project memory") == 1
    assert text.count("<!-- potpie-start -->") == 1
    assert text.count("<!-- potpie-end -->") == 1


def test_install_agent_bundle_updates_marked_agents_md_without_force(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "AGENTS.md"
    target.write_text(
        "# Local Setup\n\n<!-- potpie-start -->\nstale\n<!-- potpie-end -->\n\nKeep me.\n",
        encoding="utf-8",
    )

    result = install_agent_bundle(repo)

    text = target.read_text(encoding="utf-8")
    assert "AGENTS.md" in result.updated
    assert "# Local Setup" in text
    assert "Keep me." in text
    assert "stale" not in text
    assert "Potpie is durable project memory" in text
    assert text.count("<!-- potpie-start -->") == 1


# --- Claude bundle tests ---


def test_install_agent_bundle_claude_creates_claude_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    result = install_agent_bundle(repo, agent="claude")

    assert "CLAUDE.md" in result.created
    content = (repo / "CLAUDE.md").read_text(encoding="utf-8")
    assert "<!-- potpie-start -->" in content
    assert "potpie graph read" in content
    assert "context_resolve" not in content
    # The skills and the routing block are the whole Claude install: no slash
    # commands, no plugin directory.
    assert not (repo / ".claude" / "commands").exists()
    assert sorted(p.name for p in (repo / ".claude").iterdir()) == ["skills"]
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

    assert "CLAUDE.md" in result.updated
    content = (repo / "CLAUDE.md").read_text(encoding="utf-8")
    assert "# My Project" in content
    assert "Existing content." in content
    assert "<!-- potpie-start -->" in content
    assert "potpie graph read" in content
    assert "context_resolve" not in content


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
    assert "potpie graph read" in content
    assert "context_resolve" not in content
    assert "# Project" in content


def test_install_agent_bundle_claude_updates_changed_section_without_force(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "CLAUDE.md").write_text(
        "# Project\n\n<!-- potpie-start -->\nCUSTOM\n<!-- potpie-end -->\n\nKeep me.\n",
        encoding="utf-8",
    )

    result = install_agent_bundle(repo, agent="claude")

    content = (repo / "CLAUDE.md").read_text(encoding="utf-8")
    assert "CLAUDE.md" in result.updated
    assert "# Project" in content
    assert "Keep me." in content
    assert "CUSTOM" not in content
    assert "potpie graph read" in content
    assert "context_resolve" not in content


# --- files earlier releases installed for Claude ---


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    return repo


def _claude_project(repo: Path) -> AgentTarget:
    return AgentTarget(
        agent="claude", scope="project", path=repo, home=repo.parent / "potpie"
    )


def _ship(monkeypatch: pytest.MonkeyPatch, name: str, content: str) -> None:
    """Pretend ``content`` is the one version of a retired command Potpie shipped."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    monkeypatch.setitem(
        agent_installer._RETIRED_CLAUDE_COMMANDS, name, frozenset({digest})
    )


def test_retired_command_hashes_cover_both_commands() -> None:
    retired = agent_installer._RETIRED_CLAUDE_COMMANDS
    assert set(retired) == {"potpie-feature.md", "potpie-record.md"}
    for digests in retired.values():
        assert digests
        assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in digests)


def test_claude_sweep_deletes_retired_commands_still_as_shipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _git_repo(tmp_path)
    _ship(monkeypatch, "potpie-feature.md", "Load context first.\n")
    _ship(monkeypatch, "potpie-record.md", "Record what you learned.\n")
    commands = repo / ".claude" / "commands"
    commands.mkdir(parents=True)
    (commands / "potpie-feature.md").write_bytes(b"Load context first.\n")
    # A text-mode write on Windows stored CRLF; it is still the shipped file.
    (commands / "potpie-record.md").write_bytes(b"Record what you learned.\r\n")

    result = install_agent_bundle(repo, agent="claude")

    assert sorted(result.removed) == [
        ".claude/commands/potpie-feature.md",
        ".claude/commands/potpie-record.md",
    ]
    assert result.leftovers == []
    assert not commands.exists()


def test_claude_sweep_leaves_an_edited_retired_command_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _git_repo(tmp_path)
    _ship(monkeypatch, "potpie-feature.md", "Load context first.\n")
    commands = repo / ".claude" / "commands"
    commands.mkdir(parents=True)
    edited = commands / "potpie-feature.md"
    edited.write_text("Load context first.\nAnd my own step.\n", encoding="utf-8")
    own = commands / "deploy.md"
    own.write_text("The user's own command.\n", encoding="utf-8")

    result = install_agent_bundle(repo, agent="claude")

    assert result.removed == []
    assert [item["path"] for item in result.leftovers] == [
        ".claude/commands/potpie-feature.md"
    ]
    assert "delete it" in result.leftovers[0]["recommended_next_action"]
    assert edited.exists()
    assert own.exists()


def test_claude_sweep_reports_the_old_plugin_directory_without_deleting_it(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    manifest = repo / ".claude" / "potpie-plugin" / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "potpie"}), encoding="utf-8")

    installed = install_agent_bundle(repo, agent="claude")
    removed = _claude_project(repo).remove_instructions()

    for result in (installed, removed):
        assert [item["path"] for item in result.leftovers] == [".claude/potpie-plugin"]
        assert (
            "/plugin marketplace remove potpie"
            in (result.leftovers[0]["recommended_next_action"])
        )
    assert manifest.exists()
    # Only Potpie's own manifest identifies the directory.
    manifest.write_text(json.dumps({"name": "another-plugin"}), encoding="utf-8")
    assert install_agent_bundle(repo, agent="claude").leftovers == []


def test_only_a_claude_support_sweep_touches_retired_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _git_repo(tmp_path)
    _ship(monkeypatch, "potpie-feature.md", "Load context first.\n")
    shipped = repo / ".claude" / "commands" / "potpie-feature.md"
    shipped.parent.mkdir(parents=True)
    shipped.write_text("Load context first.\n", encoding="utf-8")

    install_agent_bundle(
        repo, agent="claude", skill_ids=("potpie-cli",), instructions=False
    )
    install_agent_bundle(
        repo, agent="claude", skill_ids=("potpie-cli",), support_files=False
    )
    install_agent_bundle(repo, agent="codex")
    install_agent_bundle(repo, agent="claude", dry_run=True)
    _claude_project(repo).install(skill_id="potpie-cli", version="1")
    AgentTarget(agent="claude", home=tmp_path / "potpie").install_instructions()
    assert shipped.exists()

    removed = _claude_project(repo).remove_instructions()

    assert removed is not None
    assert ".claude/commands/potpie-feature.md" in removed.removed
    assert not shipped.exists()


def test_support_files_is_the_old_name_of_instructions(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)

    skills_only = install_agent_bundle(
        repo, agent="claude", skill_ids=("potpie-cli",), support_files=False
    )
    with_block = install_agent_bundle(
        repo, agent="claude", skill_ids=(), instructions=False, support_files=True
    )

    assert "CLAUDE.md" not in skills_only.created
    assert with_block.created == ["CLAUDE.md"]


def test_uninstall_bundle_strips_the_block_and_removes_whole_skill_directories(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    (repo / "CLAUDE.md").write_text("# Mine\n", encoding="utf-8")
    install_agent_bundle(repo, agent="claude")
    extra = repo / ".claude" / "skills" / "potpie-cli" / "notes.md"
    extra.write_text("mine too", encoding="utf-8")

    dry = agent_installer.uninstall_bundle(
        repo, skills_dir=".claude/skills", skill_ids=None, dry_run=True
    )
    assert extra.exists() and dry.removed
    result = agent_installer.uninstall_bundle(
        repo, skills_dir=".claude/skills", skill_ids=None, instructions="CLAUDE.md"
    )

    assert result.removed[0] == "CLAUDE.md"
    assert ".claude/skills/potpie-cli" in result.removed
    assert (repo / "CLAUDE.md").read_text(encoding="utf-8") == "# Mine\n"
    assert not (repo / ".claude").exists()


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
