"""``pot create --repo`` must match ``source add repo`` normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from typer.testing import CliRunner

from potpie.cli import repo_location
from potpie.cli.commands import _common, pots
from potpie_context_engine.domain.ports.services.pot_management import PotInfo, SourceInfo

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_cli_state() -> None:
    yield
    _common.set_json(False)
    _common.set_host(None)


@dataclass
class _Pots:
    created: list[dict[str, object]] = field(default_factory=list)
    sources: list[dict[str, str | None]] = field(default_factory=list)
    repo_defaults: dict[str, str] = field(default_factory=dict)
    active_id: str | None = None
    _pots_by_name: dict[str, str] = field(default_factory=dict)

    def create_pot(
        self, *, name: str, repo: str | None = None, use: bool = False
    ) -> PotInfo:
        self.created.append({"name": name, "use": use})
        pot_id = self._pots_by_name.get(name, "pot-new")
        if name not in self._pots_by_name:
            self._pots_by_name[name] = pot_id
        if use:
            self.active_id = pot_id
        return PotInfo(pot_id=pot_id, name=name, active=use)

    def list_sources(self, *, pot_id: str) -> list[SourceInfo]:
        return [
            SourceInfo(
                source_id=row["source_id"],
                kind=row["kind"],
                name=row["name"] or row["location"],
                location=row["location"],
            )
            for row in self.sources
            if row["pot_id"] == pot_id
        ]

    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo:
        source_id = f"src-{len(self.sources) + 1}"
        self.sources.append(
            {
                "source_id": source_id,
                "pot_id": pot_id,
                "kind": kind,
                "location": location,
                "name": name,
            }
        )
        return SourceInfo(
            source_id=source_id,
            kind=kind,
            name=name or location,
            location=location,
        )

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        self.repo_defaults[repo] = pot_id


@dataclass
class _Host:
    pots: _Pots


def test_pot_create_repo_dot_uses_source_add_normalization(monkeypatch) -> None:
    monkeypatch.setattr(
        repo_location, "current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.pot_app,
        ["create", "flow-test", "--use", "--repo", "."],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"] == "pot-new"
    assert payload["repo_default_set"] is True
    assert payload["repo_key"] == "github.com/acme/shop"
    assert payload["source"]["repo_key"] == "github.com/acme/shop"
    assert payload["source"]["location"] == "github.com/acme/shop"
    assert fake_pots.created == [{"name": "flow-test", "use": True}]
    assert fake_pots.sources == [
        {
            "source_id": "src-1",
            "pot_id": "pot-new",
            "kind": "repo",
            "location": "github.com/acme/shop",
            "name": None,
        }
    ]
    assert fake_pots.repo_defaults == {"github.com/acme/shop": "pot-new"}


def test_pot_create_repo_is_idempotent_when_pot_and_source_exist(monkeypatch) -> None:
    monkeypatch.setattr(
        repo_location, "current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)
    runner = CliRunner()

    first = runner.invoke(
        pots.pot_app,
        ["create", "flow-test", "--use", "--repo", "."],
    )
    second = runner.invoke(
        pots.pot_app,
        ["create", "flow-test", "--use", "--repo", "."],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert len(fake_pots.sources) == 1
    assert fake_pots.created == [
        {"name": "flow-test", "use": True},
        {"name": "flow-test", "use": True},
    ]
    assert fake_pots.repo_defaults == {"github.com/acme/shop": "pot-new"}


def test_register_repo_source_skips_duplicate_matching_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        repo_location, "current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    fake_pots = _Pots()
    host = _Host(pots=fake_pots)

    first = pots.register_repo_source(
        host, pot_id="pot-new", location="github.com/acme/shop"
    )
    second = pots.register_repo_source(
        host, pot_id="pot-new", location=".", make_default=False
    )

    assert first["source_id"] == second["source_id"]
    assert len(fake_pots.sources) == 1
    assert fake_pots.repo_defaults == {"github.com/acme/shop": "pot-new"}


def test_pot_create_repo_no_default_skips_repo_default(monkeypatch) -> None:
    monkeypatch.setattr(
        repo_location, "current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.pot_app,
        ["create", "flow-test", "--repo", ".", "--no-default"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["repo_default_set"] is False
    assert payload["repo_key"] == "github.com/acme/shop"
    assert payload["source"]["repo_key"] == "github.com/acme/shop"
    assert fake_pots.repo_defaults == {}
