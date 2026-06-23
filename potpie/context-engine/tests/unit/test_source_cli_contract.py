"""CLI contract coverage for source registration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from typer.testing import CliRunner

from context_engine.adapters.inbound.cli.commands import _common, pots
from context_engine.domain.ports.services.pot_management import PotInfo, SourceInfo

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_cli_state():
    yield
    _common.set_json(False)
    _common.set_host(None)


@dataclass
class _Pots:
    calls: list[dict[str, str | None]] = field(default_factory=list)
    repo_defaults: dict[str, str] = field(default_factory=dict)

    def list_pots(self) -> list[PotInfo]:
        return [PotInfo(pot_id="pot-1", name="default", active=True)]

    def active_pot(self) -> PotInfo:
        return PotInfo(pot_id="pot-1", name="default", active=True)

    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo:
        self.calls.append(
            {"pot_id": pot_id, "kind": kind, "location": location, "name": name}
        )
        return SourceInfo(source_id="src-1", kind=kind, name=name or location)

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        self.repo_defaults[repo] = pot_id


@dataclass
class _Host:
    pots: _Pots


def test_source_add_plain_output_is_registration_only() -> None:
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))

    result = CliRunner().invoke(
        pots.source_app, ["add", "repo", "owner/repo", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.output
    assert fake_pots.calls == [
        {
            "pot_id": "pot-1",
            "kind": "repo",
            "location": "owner/repo",
            "name": None,
        }
    ]
    assert "registered source repo:owner/repo (src-1)" in result.output
    assert "no ingestion or scan started" in result.output
    assert fake_pots.repo_defaults == {"owner/repo": "pot-1"}


def test_source_add_json_marks_registration_only() -> None:
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app,
        [
            "add",
            "repo",
            "owner/repo",
            "--name",
            "platform",
            "--pot",
            "pot-1",
            "--no-default",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    assert emitted == {
        "source_id": "src-1",
        "kind": "repo",
        "name": "platform",
        "location": "owner/repo",
        "pot_id": "pot-1",
        "registration_only": True,
        "repo_default_set": False,
    }
    assert fake_pots.repo_defaults == {}


def test_source_add_repo_default_reports_unavailable_host() -> None:
    fake_pots = _Pots()
    fake_pots.set_repo_default = None  # type: ignore[method-assign]
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app,
        ["add", "repo", "owner/repo", "--pot", "pot-1"],
    )

    assert result.exit_code != 0
    emitted = json.loads(result.output)
    assert emitted["code"] == "repo_default_unavailable"
    assert fake_pots.calls == []
