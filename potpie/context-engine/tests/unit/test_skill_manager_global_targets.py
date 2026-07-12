"""Skill manager global and project target behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.skills.bundle_catalog import (
    recommended_skill_ids,
)
from potpie_context_engine.adapters.outbound.skills.template_resources import (
    PackageTemplateResources,
)
from potpie_context_engine.bootstrap.host_wiring import build_host_shell

TEMPLATE_RESOURCES = PackageTemplateResources("potpie.cli")


def _build_host_shell():
    return build_host_shell(
        backend=InMemoryGraphBackend(),
        template_resources=TEMPLATE_RESOURCES,
    )


def _recommended_skill_ids():
    return recommended_skill_ids(template_resources=TEMPLATE_RESOURCES)


def test_skill_manager_installs_global_harness_targets(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    potpie_home = tmp_path / "potpie"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(potpie_home))
    host = _build_host_shell()

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
            assert support_file.exists()
            assert "Potpie is durable project memory" in support_file.read_text(
                encoding="utf-8"
            )
        status = host.skills.status(agent=agent)
        assert [s.id for s in status.installed] == ["potpie-cli"]


def test_global_harness_target_paths(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _build_host_shell()

    assert host.skills.targets["cursor"].skills_root == home / ".cursor" / "skills"
    assert host.skills.targets["claude"].skills_root == home / ".claude" / "skills"
    assert (
        host.skills.targets["opencode"].skills_root
        == home / ".config" / "opencode" / "skills"
    )
    assert host.skills.targets["codex"].skills_root == home / ".agents" / "skills"


def test_skill_manager_installs_project_scope(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _build_host_shell()
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


def test_project_scope_install_preserves_existing_agents_md(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _build_host_shell()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    agents_md = repo / "AGENTS.md"
    agents_md.write_text("# Existing Setup\n\nKeep this.\n", encoding="utf-8")

    result = host.skills.install(
        agent="codex",
        skill_id="potpie-cli",
        path=str(repo),
        scope="project",
    )

    text = agents_md.read_text(encoding="utf-8")
    assert result.metadata["scope"] == "project"
    assert "# Existing Setup" in text
    assert "Keep this." in text
    assert "<!-- potpie-start -->" in text
    assert "# Context Engine" in text
    assert (repo / ".agents" / "skills" / "potpie-cli" / "SKILL.md").exists()


def test_skill_manager_removes_all_global_harness_skills(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _build_host_shell()

    expected_skill_ids = _recommended_skill_ids()
    install_result = host.skills.install(agent="codex")
    assert set(install_result.changed) == set(expected_skill_ids)

    skills_root = home / ".agents" / "skills"
    assert all(
        (skills_root / skill_id / "SKILL.md").exists()
        for skill_id in expected_skill_ids
    )

    remove_result = host.skills.remove(agent="codex", all_=True)

    assert remove_result.metadata["scope"] == "global"
    assert set(remove_result.changed) == set(expected_skill_ids)
    assert all(
        not (skills_root / skill_id / "SKILL.md").exists()
        for skill_id in expected_skill_ids
    )
    assert host.skills.status(agent="codex").installed == ()


def test_remove_uninstalled_skill_reports_no_change(
    monkeypatch, tmp_path: Path
) -> None:
    # Regression: remove() previously appended the requested id to ``changed`` even
    # when it was never installed, reporting false removals in CLI/API output.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _build_host_shell()

    result = host.skills.remove(agent="codex", skill_id="potpie-cli")
    assert result.changed == ()


def test_skill_manager_rejects_ambiguous_remove(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    host = _build_host_shell()

    with pytest.raises(ValueError, match="either a skill id or --all"):
        host.skills.remove(agent="codex", skill_id="potpie-cli", all_=True)

    with pytest.raises(ValueError, match="pass a skill id or --all"):
        host.skills.remove(agent="codex")
