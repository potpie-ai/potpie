"""CLI coverage for skill management commands."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from typer.testing import CliRunner

from context_engine.adapters.inbound.cli.commands import _common, skills
from context_engine.domain.ports.services.skill_manager import SkillOperationResult


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
    _common.set_host(_Host(skills=fake_skills))

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
    _common.set_host(_Host(skills=fake_skills))
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
