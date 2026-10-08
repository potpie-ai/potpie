"""Skill manager global and project target behavior."""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.errors import CapabilityNotImplemented

import potpie.skills.installer as agent_installer
from potpie.runtime.composition import build_local_runtime
from potpie.skills.catalog import RECOMMENDED_SKILL_IDS

EXPECTED_CATALOG = {
    "potpie-change-timeline": "2",
    "potpie-cli": "3",
    "potpie-debug-memory": "2",
    "potpie-graph": "6",
    "potpie-infra-architecture": "2",
    "potpie-project-preferences": "2",
    "potpie-repo-baseline": "2",
    "potpie-source-ingestion": "2",
}


def _root_runtime():
    return build_local_runtime(backend=InMemoryGraphBackend()).root


class _OutdatedTarget:
    agent = "codex"

    def __init__(self) -> None:
        self.versions = {"potpie-cli": "1"}

    def installed(self) -> dict[str, str]:
        return dict(self.versions)

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        del path
        self.versions[skill_id] = version

    def remove(self, *, skill_id: str) -> None:
        self.versions.pop(skill_id, None)


def test_runtime_skill_catalog_identifiers_and_versions(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()

    assert {
        skill.id: skill.version for skill in host.skills.list(agent="codex")
    } == EXPECTED_CATALOG
    assert set(RECOMMENDED_SKILL_IDS) == set(EXPECTED_CATALOG)


def test_skill_manager_installs_global_harness_targets(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    potpie_home = tmp_path / "potpie"
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(potpie_home))
    host = _root_runtime()

    expected = {
        "claude": home / ".claude" / "skills" / "potpie-cli" / "SKILL.md",
        "opencode": home
        / ".config"
        / "opencode"
        / "skills"
        / "potpie-cli"
        / "SKILL.md",
        "codex": home / ".agents" / "skills" / "potpie-cli" / "SKILL.md",
        "cursor": home / ".cursor" / "skills" / "potpie-cli" / "SKILL.md",
    }
    expected_support = {
        "claude": home / ".claude" / "CLAUDE.md",
        "codex": home / ".codex" / "AGENTS.md",
    }

    for agent, skill_file in expected.items():
        try:
            result = host.skills.install(agent=agent, skill_id="potpie-cli")
        except PermissionError as exc:
            if agent == "cursor" and ".cursor" in str(exc):
                pytest.skip("sandbox blocks writing .cursor directories")
            raise

        assert result.metadata["scope"] == "global"
        assert skill_file.exists()
        support_file = expected_support.get(agent)
        if support_file is not None:
            # Naming one skill installs that skill. The harness's own
            # instruction file belongs to the bundle sweep below.
            assert not support_file.exists()
        status = host.skills.status(agent=agent)
        assert [s.id for s in status.installed] == ["potpie-cli"]

    for agent in expected:
        host.skills.install(agent=agent)
        support_file = expected_support.get(agent)
        if support_file is not None:
            assert support_file.exists()
            assert "Potpie is durable project memory" in support_file.read_text(
                encoding="utf-8"
            )


def test_global_harness_target_paths(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()

    assert host.skills.targets["cursor"].skills_root == home / ".cursor" / "skills"
    assert host.skills.targets["claude"].skills_root == home / ".claude" / "skills"
    assert (
        host.skills.targets["opencode"].skills_root
        == home / ".config" / "opencode" / "skills"
    )
    assert host.skills.targets["codex"].skills_root == home / ".agents" / "skills"


def test_skill_manager_installs_project_scope(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    result = host.skills.install(
        agent="opencode",
        skill_id="potpie-cli",
        path=str(repo),
        scope="project",
    )

    assert result.metadata["scope"] == "project"
    assert (repo / ".opencode" / "skills" / "potpie-cli" / "SKILL.md").exists()
    status = host.skills.status(agent="opencode", path=str(repo), scope="project")
    assert [s.id for s in status.installed] == ["potpie-cli"]

    rerun = host.skills.install(
        agent="opencode",
        skill_id="potpie-cli",
        path=str(repo),
        scope="project",
    )
    assert rerun.changed == ()


def test_project_scope_bundle_install_preserves_existing_agents_md(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    agents_md = repo / "AGENTS.md"
    agents_md.write_text("# Existing Setup\n\nKeep this.\n", encoding="utf-8")

    result = host.skills.install(agent="codex", path=str(repo), scope="project")

    text = agents_md.read_text(encoding="utf-8")
    assert result.metadata["scope"] == "project"
    assert "# Existing Setup" in text
    assert "Keep this." in text
    assert "<!-- potpie-start -->" in text
    assert "Potpie is durable project memory" in text
    assert (repo / ".agents" / "skills" / "potpie-cli" / "SKILL.md").exists()
    assert result.metadata["support_files"] == ["AGENTS.md"]


def test_installing_one_named_skill_does_not_touch_the_instruction_file(
    monkeypatch, tmp_path: Path
) -> None:
    """Naming a skill asks for that skill, not for an edit to AGENTS.md.

    ``skills install potpie-cli`` used to also rewrite the harness instruction
    file the user wrote, and — for Claude — a second skill, while reporting
    only the one id it was given.
    """
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    agents_md = repo / "AGENTS.md"
    agents_md.write_text("# Existing Setup\n\nKeep this.\n", encoding="utf-8")

    result = host.skills.install(
        agent="codex", skill_id="potpie-cli", path=str(repo), scope="project"
    )

    assert result.changed == ("potpie-cli",)
    assert "support_files" not in result.metadata
    assert agents_md.read_text(encoding="utf-8") == "# Existing Setup\n\nKeep this.\n"
    assert (repo / ".agents" / "skills" / "potpie-cli" / "SKILL.md").exists()
    assert not (repo / ".agents" / "skills" / "potpie-graph").exists()


def test_skill_manager_removes_all_global_harness_skills(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()

    install_result = host.skills.install(agent="codex")
    assert set(install_result.changed) == set(RECOMMENDED_SKILL_IDS)

    skills_root = home / ".agents" / "skills"
    assert all(
        (skills_root / skill_id / "SKILL.md").exists()
        for skill_id in RECOMMENDED_SKILL_IDS
    )

    remove_result = host.skills.remove(agent="codex", all_=True)

    assert remove_result.metadata["scope"] == "global"
    assert set(remove_result.changed) == set(RECOMMENDED_SKILL_IDS)
    assert all(
        not (skills_root / skill_id / "SKILL.md").exists()
        for skill_id in RECOMMENDED_SKILL_IDS
    )
    assert host.skills.status(agent="codex").installed == ()


def test_skill_manager_update_reports_and_repairs_outdated_target(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    target = _OutdatedTarget()
    manager = type(host.skills)(targets={"codex": target})

    result = manager.update(agent="codex", skill_id="potpie-cli")

    assert result.operation == "update"
    assert result.changed == ("potpie-cli",)
    assert result.metadata == {"scope": "global"}
    assert target.versions == {"potpie-cli": EXPECTED_CATALOG["potpie-cli"]}


def test_remove_uninstalled_skill_reports_no_change(
    monkeypatch, tmp_path: Path
) -> None:
    # Regression: remove() previously appended the requested id to ``changed`` even
    # when it was never installed, reporting false removals in CLI/API output.
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()

    result = host.skills.remove(agent="codex", skill_id="potpie-cli")
    assert result.changed == ()


def test_skill_manager_rejects_ambiguous_remove(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()

    with pytest.raises(ValueError, match="either a skill id or --all"):
        host.skills.remove(agent="codex", skill_id="potpie-cli", all_=True)

    with pytest.raises(ValueError, match="pass a skill id or --all"):
        host.skills.remove(agent="codex")


def _repo(root: Path, name: str = "repo") -> Path:
    repo = root / name
    (repo / ".git").mkdir(parents=True)
    return repo


# --- removal owns what installation wrote ------------------------------------


def test_remove_all_takes_the_support_files_with_it(
    monkeypatch, tmp_path: Path
) -> None:
    """The mirror of the sweep that wrote them.

    ``skills remove --all`` deleted every skill directory and left ``CLAUDE.md``
    in place, so the harness went on routing to skills that were gone — and
    ``skills status`` reported a clean uninstall.
    """
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = _repo(tmp_path)
    installed = host.skills.install(agent="claude", path=str(repo), scope="project")
    assert installed.metadata["support_files"] == ["CLAUDE.md"]

    result = host.skills.remove(
        agent="claude", all_=True, path=str(repo), scope="project"
    )

    assert not (repo / "CLAUDE.md").exists()
    assert not (repo / ".claude").exists()
    # Named, exactly as the install names it: a file the command touched that
    # no id in ``removed`` accounts for.
    assert result.metadata["support_files"] == ["CLAUDE.md"]


def test_remove_all_keeps_the_parts_of_claude_md_the_user_wrote(
    monkeypatch, tmp_path: Path
) -> None:
    """Install merges into a user-authored file, so removal has to un-merge.

    Deleting a hand-written ``CLAUDE.md`` because Potpie once appended to it
    would be a far larger removal than the one the caller asked for.
    """
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = _repo(tmp_path)
    claude_md = repo / "CLAUDE.md"
    claude_md.write_text("# House rules\n\nKeep this.\n", encoding="utf-8")
    host.skills.install(agent="claude", path=str(repo), scope="project")
    assert "<!-- potpie-start -->" in claude_md.read_text(encoding="utf-8")

    host.skills.remove(agent="claude", all_=True, path=str(repo), scope="project")

    text = claude_md.read_text(encoding="utf-8")
    assert "# House rules" in text
    assert "Keep this." in text
    assert "potpie-start" not in text


def test_claude_sweeps_report_what_earlier_releases_left_behind(
    monkeypatch, tmp_path: Path
) -> None:
    """Retired files are cleared only when provably untouched, else reported."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = _repo(tmp_path)
    shipped = "Load context first.\n"
    monkeypatch.setitem(
        agent_installer._RETIRED_CLAUDE_COMMANDS,
        "potpie-feature.md",
        frozenset({hashlib.sha256(shipped.encode("utf-8")).hexdigest()}),
    )
    command = repo / ".claude" / "commands" / "potpie-feature.md"
    command.parent.mkdir(parents=True)
    command.write_text(shipped, encoding="utf-8")
    manifest = repo / ".claude" / "potpie-plugin" / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "potpie"}), encoding="utf-8")

    installed = host.skills.install(agent="claude", path=str(repo), scope="project")

    assert installed.metadata["retired_files_removed"] == [
        ".claude/commands/potpie-feature.md"
    ]
    assert [item["path"] for item in installed.metadata["leftovers"]] == [
        ".claude/potpie-plugin"
    ]
    assert not command.exists()
    assert manifest.exists()

    removed = host.skills.remove(
        agent="claude", all_=True, path=str(repo), scope="project"
    )

    assert [item["path"] for item in removed.metadata["leftovers"]] == [
        ".claude/potpie-plugin"
    ]
    assert manifest.exists()


def test_removing_one_named_skill_leaves_the_support_files_alone(
    monkeypatch, tmp_path: Path
) -> None:
    """Symmetric with install: naming an id is a request about that id."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = _repo(tmp_path)
    host.skills.install(agent="claude", path=str(repo), scope="project")

    result = host.skills.remove(
        agent="claude", skill_id="potpie-cli", path=str(repo), scope="project"
    )

    assert result.changed == ("potpie-cli",)
    assert "support_files" not in result.metadata
    assert (repo / "CLAUDE.md").exists()


def test_remove_says_which_ids_were_never_installed(
    monkeypatch, tmp_path: Path
) -> None:
    """``removed: []`` is also what a successful removal reports next time round.

    So a caller who named a skill that was never installed for this harness read
    "already removed" and moved on, when the truth was "you are pointed at the
    wrong harness (or the wrong project)".
    """
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = _repo(tmp_path)
    host.skills.install(
        agent="codex", skill_id="potpie-cli", path=str(repo), scope="project"
    )

    result = host.skills.remove(
        agent="codex", skill_id="potpie-graph", path=str(repo), scope="project"
    )

    assert result.changed == ()
    assert result.metadata["not_installed"] == ["potpie-graph"]
    # An id that was there is still reported the same way, with nothing extra.
    removed = host.skills.remove(
        agent="codex", skill_id="potpie-cli", path=str(repo), scope="project"
    )
    assert removed.changed == ("potpie-cli",)
    assert "not_installed" not in removed.metadata


# --- refusals that used to be plausible answers -------------------------------


def test_an_unknown_agent_is_refused_at_project_scope_too(
    monkeypatch, tmp_path: Path
) -> None:
    """Global scope refused a typo; project scope answered it.

    The project target was built for any string and fell through to the
    default ``.agents/skills`` layout, so a mistyped harness produced a complete
    and entirely plausible listing — every catalog skill, all ``installed:
    false`` — for a harness that does not exist.
    """
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = _repo(tmp_path)

    for call in (
        lambda: host.skills.list(agent="clawd", path=str(repo), scope="project"),
        lambda: host.skills.status(agent="clawd", path=str(repo), scope="project"),
        lambda: host.skills.install(agent="clawd", path=str(repo), scope="project"),
        lambda: host.skills.remove(
            agent="clawd", all_=True, path=str(repo), scope="project"
        ),
    ):
        with pytest.raises(ValueError) as exc:
            call()
        assert "clawd" in str(exc.value)
        assert "claude" in str(exc.value)  # the "Known:" listing
    # And the registered harnesses still work at project scope.
    assert host.skills.list(agent="claude", path=str(repo), scope="project")


def test_a_read_only_target_names_the_directory_to_fix(
    monkeypatch, tmp_path: Path
) -> None:
    """A bare ``PermissionError`` reaches the CLI as an unexpected internal error.

    The actual repair is one ``chmod`` on a directory that message never named,
    so the refusal is a validation error that carries the directory to fix.
    """
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    repo = _repo(tmp_path)
    repo.chmod(0o500)
    try:
        with pytest.raises(ValueError) as exc:
            host.skills.install(agent="codex", path=str(repo), scope="project")
    finally:
        repo.chmod(0o700)

    assert "not writable" in str(exc.value)
    # The nearest directory that exists — not the leaf the failed mkdir never
    # created, which is what a `chmod` on the suggested path would have missed.
    assert getattr(exc.value, "recommended_next_action", "").startswith(
        f"make '{repo}' writable"
    )


def test_a_manifest_of_the_wrong_shape_does_not_crash_the_read(
    monkeypatch, tmp_path: Path
) -> None:
    """The manifest is a cache; the files on disk are the truth.

    Only ``JSONDecodeError`` was caught, so valid JSON of the wrong shape — a
    list, whatever a stray write leaves behind — raised ``AttributeError: 'list'
    object has no attribute 'items'`` out of ``skills list``.
    """
    potpie_home = tmp_path / "potpie"
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(potpie_home))
    host = _root_runtime()
    repo = _repo(tmp_path)
    host.skills.install(
        agent="codex", skill_id="potpie-cli", path=str(repo), scope="project"
    )
    manifest = next(potpie_home.glob("skill_manifest_codex_project_*.json"))
    manifest.write_text('["not", "a", "dict"]', encoding="utf-8")

    rows = {
        s.id: s
        for s in host.skills.list(agent="codex", path=str(repo), scope="project")
    }

    # Present on disk, version unrecoverable — so it is reported outdated and
    # the reinstall that repair names rewrites the manifest.
    assert rows["potpie-cli"].installed is True
    assert rows["potpie-cli"].installed_version == "unknown"
    status = host.skills.status(agent="codex", path=str(repo), scope="project")
    assert [s.id for s in status.outdated] == ["potpie-cli"]
    assert host.skills.install(
        agent="codex", skill_id="potpie-cli", path=str(repo), scope="project"
    ).changed == ("potpie-cli",)


# --- catalog add --------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", "Pass a skill source"),
        ("   ", "Pass a skill source"),
        ("/no/such/place", "No such skill source"),
        ("ftp://example.com/skill.zip", "Cannot fetch a skill over 'ftp'"),
    ],
)
def test_add_refuses_a_source_it_could_never_resolve(
    monkeypatch, tmp_path: Path, source: str, expected: str
) -> None:
    """``skills add`` took anything and answered exit 0.

    A path with a typo, a bare hostname, an ``ftp://`` URL — every one came back
    looking like it had registered a skill, and the only way to find out
    otherwise was that it never appeared in ``skills list``.
    """
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()

    with pytest.raises(ValueError, match=expected):
        host.skills.add(source=source)


def test_add_of_a_usable_source_reports_the_capability_gap(
    monkeypatch, tmp_path: Path
) -> None:
    """Validating the source is not the same as having implemented the command."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    source = tmp_path / "my-skill"
    source.mkdir()
    (source / "SKILL.md").write_text("# mine\n", encoding="utf-8")

    with pytest.raises(CapabilityNotImplemented) as exc:
        host.skills.add(source=str(source))

    assert "skills.catalog.add" in str(exc.value)
    assert "potpie skills install" in (exc.value.recommended_next_action or "")


def test_add_refuses_a_directory_that_is_not_a_skill(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _root_runtime()
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="no SKILL.md"):
        host.skills.add(source=str(empty))
