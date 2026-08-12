"""Skills: per-project manifests, content drift, and harness-home isolation.

Four defects that compounded into "the skills subsystem cannot tell you what is
installed":

- the project-scope version manifest was keyed by agent and scope with no
  project path, so every repository on the machine shared one record — and
  ``skills remove --all`` in any one of them marked every other project fully
  outdated;
- ``install``/``update`` compared only a version integer, so a truncated or
  hand-edited ``SKILL.md`` was undetectable *and* unfixable: both exited 0 with
  ``changed: []`` and the harness went on loading the broken file;
- ``skills list`` printed the *catalog* version beside every installed skill, so
  it structurally could not show drift and gave a clean bill of health to the
  same skills ``skills status`` was calling outdated;
- harness roots came off ``Path.home()`` unconditionally, so a sandboxed run
  installed into the developer's real ``~/.claude`` and friends.

These drive the real targets against real files. The manifest key and the
content comparison both live in the target, which the manager-level tests fake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.skills.claude_target import (
    ClaudeAgentTarget,
    ProjectAgentTarget,
)
from potpie_context_engine.adapters.outbound.skills.harness_home import harness_home
from potpie_context_engine.bootstrap.host_wiring import build_host_shell

SKILL = "potpie-cli"


def _repo(root: Path, name: str) -> Path:
    repo = root / name
    (repo / ".git").mkdir(parents=True)
    return repo


# --- harness home ------------------------------------------------------------


def test_harness_home_defaults_to_the_real_home(monkeypatch, tmp_path: Path) -> None:
    """``CONTEXT_ENGINE_HOME`` must *not* move harness files.

    Someone who relocates Potpie's state still runs their harness out of
    ``~/.claude``; installing under the state home instead would write skills
    somewhere nothing reads and report success.
    """
    monkeypatch.delenv("POTPIE_HARNESS_HOME", raising=False)
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "relocated"))

    assert harness_home() == Path.home()


def test_harness_home_follows_its_own_variable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(tmp_path / "sandbox"))

    assert (
        ClaudeAgentTarget().skills_root == tmp_path / "sandbox" / ".claude" / "skills"
    )


# --- one manifest per project ------------------------------------------------


def test_two_projects_do_not_share_one_version_record(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = build_host_shell(backend=InMemoryGraphBackend())
    first = _repo(tmp_path, "first")
    second = _repo(tmp_path, "second")

    host.skills.install(agent="codex", skill_id=SKILL, path=str(first), scope="project")

    # The second project has never had a skill installed, so nothing about it
    # may be "already installed" — the shared manifest used to say it was.
    status = host.skills.status(agent="codex", path=str(second), scope="project")
    assert [s.id for s in status.installed] == []
    assert SKILL in [s.id for s in status.missing]

    installed = host.skills.install(
        agent="codex", skill_id=SKILL, path=str(second), scope="project"
    )
    assert installed.changed == (SKILL,)


def test_removing_in_one_project_leaves_the_other_alone(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = build_host_shell(backend=InMemoryGraphBackend())
    first = _repo(tmp_path, "first")
    second = _repo(tmp_path, "second")
    for repo in (first, second):
        host.skills.install(
            agent="codex", skill_id=SKILL, path=str(repo), scope="project"
        )

    host.skills.remove(agent="codex", all_=True, path=str(first), scope="project")

    survivor = host.skills.status(agent="codex", path=str(second), scope="project")
    assert [s.id for s in survivor.installed] == [SKILL]


def test_target_root_is_the_repo_root_not_the_path_passed_in(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "repo")
    nested = repo / "packages" / "api"
    nested.mkdir(parents=True)

    target = ProjectAgentTarget(agent="codex", path=nested, home=tmp_path / "potpie")

    # Installs have always resolved to the nearest git root, so reporting the
    # raw --path named a directory nothing was written to.
    assert target.target_root == repo


# --- content drift -----------------------------------------------------------


def test_a_damaged_skill_is_detected_and_repaired(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = build_host_shell(backend=InMemoryGraphBackend())
    repo = _repo(tmp_path, "repo")
    host.skills.install(agent="codex", skill_id=SKILL, path=str(repo), scope="project")
    skill_md = repo / ".agents" / "skills" / SKILL / "SKILL.md"
    intact = skill_md.read_text(encoding="utf-8")

    skill_md.write_text("truncated", encoding="utf-8")

    status = host.skills.status(agent="codex", path=str(repo), scope="project")
    drifted = [s for s in status.outdated if s.id == SKILL]
    assert drifted and drifted[0].drifted is True
    # It is still at the recorded version — which is exactly why a version
    # comparison reported it healthy.
    assert drifted[0].installed_version == drifted[0].version

    repaired = host.skills.install(
        agent="codex", skill_id=SKILL, path=str(repo), scope="project"
    )

    assert repaired.changed == (SKILL,)
    assert skill_md.read_text(encoding="utf-8") == intact
    after = host.skills.status(agent="codex", path=str(repo), scope="project")
    assert [s.id for s in after.installed] == [SKILL]


def test_update_all_repairs_damage_too(monkeypatch, tmp_path: Path) -> None:
    """``update --all`` exited 0 with ``changed: []`` on a corrupted file."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = build_host_shell(backend=InMemoryGraphBackend())
    repo = _repo(tmp_path, "repo")
    host.skills.install(agent="codex", skill_id=SKILL, path=str(repo), scope="project")
    skill_md = repo / ".agents" / "skills" / SKILL / "SKILL.md"
    skill_md.write_text("truncated", encoding="utf-8")

    result = host.skills.update(
        agent="codex", all_=True, path=str(repo), scope="project"
    )

    assert result.changed == (SKILL,)
    assert "truncated" not in skill_md.read_text(encoding="utf-8")


def test_list_reports_the_installed_version_and_drift(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = build_host_shell(backend=InMemoryGraphBackend())
    repo = _repo(tmp_path, "repo")
    host.skills.install(agent="codex", skill_id=SKILL, path=str(repo), scope="project")
    (repo / ".agents" / "skills" / SKILL / "SKILL.md").write_text("x", encoding="utf-8")

    rows = {
        s.id: s
        for s in host.skills.list(agent="codex", path=str(repo), scope="project")
    }

    assert rows[SKILL].installed is True
    assert rows[SKILL].drifted is True
    # A skill that was never installed reports no installed version at all,
    # rather than borrowing the catalog's.
    never = next(s for s in rows.values() if not s.installed)
    assert never.installed_version is None
    assert never.drifted is False


# --- the Claude Code plugin --------------------------------------------------


def test_the_plugin_installs_at_project_scope(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = build_host_shell(backend=InMemoryGraphBackend())
    repo = _repo(tmp_path, "repo")

    result = host.skills.install(agent="claude-plugin", path=str(repo), scope="project")

    plugin = repo / ".claude" / "potpie-plugin"
    assert (plugin / ".claude-plugin" / "plugin.json").exists()
    assert (plugin / "skills" / "potpie-graph" / "SKILL.md").exists()
    # The bundle carries ten of the eleven catalog skills. The missing one is
    # reported, not silently skipped — and it must never appear in `changed`,
    # which is what made every rerun claim to install it again.
    assert SKILL not in result.changed
    assert result.metadata["unavailable"] == [SKILL]
    assert (
        host.skills.install(
            agent="claude-plugin", path=str(repo), scope="project"
        ).changed
        == ()
    )


def test_the_plugin_refuses_global_scope_with_a_repair(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = build_host_shell(backend=InMemoryGraphBackend())

    with pytest.raises(ValueError) as exc:
        host.skills.install(agent="claude-plugin")

    # Not "no install target registered … Known: claude, codex, …", which reads
    # as "this harness is unsupported" for a bundle that ships in the wheel.
    assert "--scope project" in str(exc.value)


def test_naming_a_skill_the_plugin_cannot_carry_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = build_host_shell(backend=InMemoryGraphBackend())
    repo = _repo(tmp_path, "repo")

    with pytest.raises(ValueError) as exc:
        host.skills.install(
            agent="claude-plugin", skill_id=SKILL, path=str(repo), scope="project"
        )

    assert SKILL in str(exc.value)
    assert "--agent claude" in str(exc.value)
