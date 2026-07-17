"""CLI contract coverage for source registration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, pots
from potpie_context_engine.domain.ports.services.pot_management import PotInfo, SourceInfo

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
        "repo_key": "owner/repo",
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


# ---------------------------------------------------------------------------
# source status — audit-10: no-ID per-pot summary and enriched single-source
# ---------------------------------------------------------------------------


@dataclass
class _StatusPots:
    """Fake pots service with list_sources, source_status, and repo_default."""

    _sources: list[SourceInfo] = field(default_factory=list)
    repo_defaults: dict[str, str] = field(default_factory=dict)

    def list_pots(self) -> list[PotInfo]:
        return [PotInfo(pot_id="pot-1", name="default", active=True)]

    def active_pot(self) -> PotInfo:
        return PotInfo(pot_id="pot-1", name="default", active=True)

    def list_sources(self, *, pot_id: str) -> list[SourceInfo]:
        return self._sources

    def source_status(self, *, pot_id: str, source_id: str) -> SourceInfo:
        for s in self._sources:
            if s.source_id == source_id:
                return s
        raise ValueError(f"no source {source_id}")

    def repo_default(self, *, repo: str) -> str | None:
        return self.repo_defaults.get(repo)

    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo:
        src = SourceInfo(
            source_id="src-new",
            kind=kind,
            name=name or location,
            location=location,
        )
        self._sources.append(src)
        return src

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        self.repo_defaults[repo] = pot_id


@dataclass
class _StatusHost:
    pots: _StatusPots
    graph: object = None


def test_source_status_no_id_returns_pot_summary() -> None:
    """No-ID invocation returns per-pot summary with all sources and pot info."""
    src = SourceInfo(
        source_id="src-1",
        kind="repo",
        name="acme/shop",
        location="github.com/acme/shop",
        status="ok",
    )
    fake_pots = _StatusPots(_sources=[src])
    _common.set_host(_StatusHost(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["status", "--pot", "pot-1"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pot_id"] == "pot-1"
    assert payload["source_count"] == 1
    assert len(payload["sources"]) == 1
    row = payload["sources"][0]
    assert row["id"] == "src-1"
    assert row["kind"] == "repo"
    assert row["location"] == "github.com/acme/shop"
    assert row["registration_only"] is True
    assert row["ingestion_status"] == "not_started"
    assert "claim_count" in payload


def test_source_status_no_id_marks_repo_default() -> None:
    """Source whose location is the pot's repo default is marked repo_default=True."""
    src = SourceInfo(
        source_id="src-1",
        kind="repo",
        name="acme/shop",
        location="github.com/acme/shop",
        status="ok",
    )
    fake_pots = _StatusPots(_sources=[src])
    fake_pots.repo_defaults["github.com/acme/shop"] = "pot-1"
    _common.set_host(_StatusHost(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["status", "--pot", "pot-1"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sources"][0]["repo_default"] is True


def test_source_status_no_id_no_sources_recommends_add() -> None:
    """Per-pot summary with no sources includes a recommended_next_action hint."""
    fake_pots = _StatusPots(_sources=[])
    _common.set_host(_StatusHost(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["status", "--pot", "pot-1"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source_count"] == 0
    assert payload["recommended_next_action"] is not None
    assert "source add repo" in payload["recommended_next_action"]


def test_source_status_with_id_returns_enriched_row() -> None:
    """Providing a source-id returns a single enriched row, not the old 3-field shape."""
    src = SourceInfo(
        source_id="src-abc",
        kind="repo",
        name=".",
        location="/home/user/project",
        status="ok",
    )
    fake_pots = _StatusPots(_sources=[src])
    _common.set_host(_StatusHost(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app, ["status", "src-abc", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"] == "src-abc"
    assert payload["kind"] == "repo"
    assert payload["location"] == "/home/user/project"
    assert payload["registration_only"] is True
    assert payload["ingestion_status"] == "not_started"
    assert "status" in payload


def test_source_status_no_id_human_output_contains_kind_and_location() -> None:
    """Plain-text no-ID output shows kind, location, and registration-only hint."""
    src = SourceInfo(
        source_id="src-1",
        kind="repo",
        name="shop",
        location="github.com/acme/shop",
        status="ok",
    )
    fake_pots = _StatusPots(_sources=[src])
    _common.set_host(_StatusHost(pots=fake_pots))

    result = CliRunner().invoke(pots.source_app, ["status", "--pot", "pot-1"])

    assert result.exit_code == 0, result.output
    assert "repo" in result.output
    assert "github.com/acme/shop" in result.output
    assert "registration-only" in result.output
