"""CLI coverage for skill management commands."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, skills
from potpie.skills.contracts import (
    SkillOperationResult,
)


@pytest.fixture(autouse=True)
def _reset_cli_output_mode():
    """Keep global CLI JSON mode from leaking across tests (order-independence)."""
    _common.set_json(False)
    yield
    _common.set_json(False)


@dataclass
class _Skills:
    calls: list[dict[str, object]] = field(default_factory=list)

    def remove(
        self,
        *,
        agent: str,
        skill_id: str | None = None,
        all_: bool = False,
        path: str | None = None,
        scope: str = "global",
    ) -> SkillOperationResult:
        self.calls.append(
            {
                "agent": agent,
                "skill_id": skill_id,
                "all_": all_,
                "path": path,
                "scope": scope,
            }
        )
        return SkillOperationResult(
            agent=agent,
            operation="remove",
            changed=("potpie-graph", "potpie-cli"),
            metadata={"scope": scope},
        )


@dataclass
class _Host:
    skills: _Skills


def test_skills_remove_all_defaults_to_global_scope() -> None:
    fake_skills = _Skills()
    _common.set_runtime(_Host(skills=fake_skills))

    result = CliRunner().invoke(
        skills.skills_app,
        ["remove", "--all", "--agent", "codex"],
    )

    assert result.exit_code == 0, result.output
    assert fake_skills.calls == [
        {
            "agent": "codex",
            "skill_id": None,
            "all_": True,
            "path": None,
            "scope": "global",
        }
    ]
    assert "removed Potpie skills for codex" in result.output


def test_skills_remove_all_json_output() -> None:
    fake_skills = _Skills()
    _common.set_runtime(_Host(skills=fake_skills))
    _common.set_json(True)

    result = CliRunner().invoke(
        skills.skills_app,
        ["remove", "--all", "--agent", "codex"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    assert emitted["agent"] == "codex"
    assert emitted["scope"] == "global"
    assert emitted["removed"] == ["potpie-graph", "potpie-cli"]


# --- driven through the real manager -----------------------------------------
#
# ``removed: []`` is the manager's honest answer in two very different
# situations, so what the *command* prints is the whole defect: these run the
# real ``DefaultSkillManager`` against real files and read its output.


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "potpie"))
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "in_memory")
    checkout = tmp_path / "repo"
    (checkout / ".git").mkdir(parents=True)
    return checkout


def _skills_cli(*args: str):
    return CliRunner().invoke(skills.skills_app, list(args))


def test_status_json_reports_drifted_and_disabled(repo) -> None:
    """The keys installers parse: ``drifted`` and ``disabled`` beside the rest."""
    _skills_cli("install", "--agent", "codex", "--path", str(repo))
    _skills_cli("remove", "potpie-graph", "--agent", "codex", "--path", str(repo))
    skill_md = repo / ".agents" / "skills" / "potpie-cli" / "SKILL.md"
    skill_md.write_text("truncated", encoding="utf-8")
    _common.set_json(True)

    result = _skills_cli("status", "--agent", "codex", "--path", str(repo))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {
        "agent",
        "scope",
        "installed",
        "missing",
        "outdated",
        "drifted",
        "disabled",
    }
    assert payload["drifted"] == ["potpie-cli"]
    assert "potpie-cli" in payload["outdated"]
    assert payload["disabled"] == ["potpie-graph"]
    assert "potpie-graph" not in payload["missing"]


def test_install_accepts_a_hidden_no_daemon_flag(repo) -> None:
    result = _skills_cli(
        "install", "--agent", "codex", "--path", str(repo), "--no-daemon"
    )

    assert result.exit_code == 0, result.output
    assert (repo / ".agents" / "skills" / "potpie-cli" / "SKILL.md").exists()
    assert "--no-daemon" not in _skills_cli("install", "--help").output


def test_remove_names_an_id_that_was_never_installed(repo) -> None:
    """Otherwise a typo'd id and a completed removal print the same line."""
    _skills_cli("install", "potpie-cli", "--agent", "claude", "--path", str(repo))

    result = _skills_cli(
        "remove", "potpie-graph", "--agent", "claude", "--path", str(repo)
    )

    assert result.exit_code == 0, result.output
    assert "potpie-graph" in result.output
    assert "nothing to remove" in result.output
    assert "already removed" not in result.output


def test_remove_all_reports_the_support_files_it_took_back(repo) -> None:
    """The mirror of ``install``, which names the support files it wrote."""
    _skills_cli("install", "--agent", "claude", "--path", str(repo))
    assert (repo / "CLAUDE.md").exists()

    result = _skills_cli("remove", "--all", "--agent", "claude", "--path", str(repo))

    assert result.exit_code == 0, result.output
    assert "removed support files" in result.output
    assert "CLAUDE.md" in result.output
    assert not (repo / "CLAUDE.md").exists()
    assert not (repo / ".claude" / "commands").exists()


def test_add_refuses_a_source_that_is_not_there(repo) -> None:
    _common.set_json(True)

    result = _skills_cli("add", str(repo / "no-such-skill"))

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "No such skill source" in payload["message"]


def test_add_resolves_a_relative_source_in_the_callers_cwd(
    repo, monkeypatch, tmp_path
) -> None:
    """Only this process knows where "." is, so the source is absolutised here."""
    _common.set_json(True)
    source = tmp_path / "my-skill"
    source.mkdir()
    (source / "SKILL.md").write_text("# mine\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = _skills_cli("add", "./my-skill")

    # Reached the capability gap, which means the source resolved and validated.
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["code"] == "not_implemented"


def test_add_of_a_real_source_reports_the_capability_gap(repo, tmp_path) -> None:
    """Not exit 0 with a sentence about not being implemented."""
    _common.set_json(True)
    source = tmp_path / "my-skill"
    source.mkdir()
    (source / "SKILL.md").write_text("# mine\n", encoding="utf-8")

    result = _skills_cli("add", str(source))

    assert result.exit_code == 2, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "not_implemented"
    assert "potpie skills install" in payload["recommended_next_action"]
