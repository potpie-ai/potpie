"""Tests for the compact ``pot default`` command."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, pots
from tests.runtime_fakes import runtime_from_services

pytestmark = pytest.mark.unit


# ---------- helpers reused from test_cli_ergonomics -------------------------


class _Pot:
    def __init__(self, pot_id: str, name: str, active: bool = False) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = active


class _Source:
    def __init__(self, kind: str, name: str, location: str | None = None) -> None:
        self.source_id = f"src_{name}"
        self.kind = kind
        self.name = name
        self.location = location or name


class _Pots:
    def __init__(self, pots, sources_by_pot, active=None) -> None:
        self._pots = pots
        self._sources = sources_by_pot
        self._active = active
        self.repo_defaults: dict[str, str] = {}

    def list_pots(self):
        return self._pots

    def active_pot(self):
        return self._active

    def list_sources(self, *, pot_id):
        return self._sources.get(pot_id, [])

    def repo_default(self, *, repo):
        return self.repo_defaults.get(repo)

    def set_repo_default(self, *, repo, pot_id):
        self.repo_defaults[repo] = pot_id

    def clear_repo_default(self, *, repo):
        return self.repo_defaults.pop(repo, None) is not None

    def list_repo_defaults(self):
        return dict(self.repo_defaults)


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_json(False)
    _common.set_cli_runtime(None)


# ============================================================================
# Audit 8 — pot default show compact by default
# ============================================================================


def _setup_default_show(monkeypatch, *, default_pot_id: str | None = None):
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    match = _Source("repo", "github.com/acme/shop")
    pots_service = _Pots(
        [_Pot("p1", "shop"), _Pot("p2", "shop-fork")],
        {"p1": [match], "p2": [match]},
        active=None,
    )
    if default_pot_id:
        pots_service.repo_defaults["github.com/acme/shop"] = default_pot_id
    _common.set_cli_runtime(runtime_from_services(pots=pots_service))
    _common.set_json(True)
    return pots_service


class TestPotDefaultShowCompact:
    def test_default_show_omits_candidates_by_default(self, monkeypatch) -> None:
        _setup_default_show(monkeypatch, default_pot_id="p1")
        result = CliRunner().invoke(pots.pot_app, ["default"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)["data"]
        assert "candidates" not in payload
        assert payload["default_pot_id"] == "p1"
        assert payload["default_pot"]["id"] == "p1"
        assert "hint" in payload

    def test_default_show_includes_candidates_with_flag(self, monkeypatch) -> None:
        _setup_default_show(monkeypatch, default_pot_id="p1")
        result = CliRunner().invoke(pots.pot_app, ["default", "--with-candidates"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)["data"]
        assert "candidates" in payload
        assert len(payload["candidates"]) >= 1
        assert "hint" not in payload

    def test_default_show_unset_omits_candidates(self, monkeypatch) -> None:
        _setup_default_show(monkeypatch, default_pot_id=None)
        result = CliRunner().invoke(pots.pot_app, ["default"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)["data"]
        assert payload["default_pot_id"] is None
        assert "candidates" not in payload
        assert "hint" in payload

    def test_default_show_unset_with_candidates(self, monkeypatch) -> None:
        _setup_default_show(monkeypatch, default_pot_id=None)
        result = CliRunner().invoke(pots.pot_app, ["default", "--with-candidates"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)["data"]
        assert "candidates" in payload
        assert "hint" not in payload

    def test_linked_still_returns_full_candidates(self, monkeypatch) -> None:
        _setup_default_show(monkeypatch, default_pot_id="p1")
        result = CliRunner().invoke(pots.pot_app, ["linked"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)["data"]
        assert "candidates" in payload
        assert len(payload["candidates"]) >= 1
        assert payload["default_pot_id"] == "p1"
        assert payload["repo"] == "github.com/acme/shop"
