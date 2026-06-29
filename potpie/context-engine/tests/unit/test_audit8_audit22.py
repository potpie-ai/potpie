"""Tests for CLI audit 8 (compact pot default show) and audit 22 (cloud group JSON)."""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from adapters.inbound.cli.commands import _common, cloud, pots

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


class _Host:
    def __init__(self, pots_service) -> None:
        self.pots = pots_service


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_json(False)
    _common.set_host(None)


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
    _common.set_host(_Host(pots_service))
    _common.set_json(True)
    return pots_service


class TestPotDefaultShowCompact:
    def test_default_show_omits_candidates_by_default(self, monkeypatch) -> None:
        _setup_default_show(monkeypatch, default_pot_id="p1")
        result = CliRunner().invoke(pots.pot_app, ["default", "show"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "candidates" not in payload
        assert payload["default_pot_id"] == "p1"
        assert payload["default_pot"]["id"] == "p1"
        assert "hint" in payload

    def test_default_show_includes_candidates_with_flag(self, monkeypatch) -> None:
        _setup_default_show(monkeypatch, default_pot_id="p1")
        result = CliRunner().invoke(
            pots.pot_app, ["default", "show", "--with-candidates"]
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "candidates" in payload
        assert len(payload["candidates"]) >= 1
        assert "hint" not in payload

    def test_default_show_unset_omits_candidates(self, monkeypatch) -> None:
        _setup_default_show(monkeypatch, default_pot_id=None)
        result = CliRunner().invoke(pots.pot_app, ["default", "show"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["default_pot_id"] is None
        assert "candidates" not in payload
        assert "hint" in payload

    def test_default_show_unset_with_candidates(self, monkeypatch) -> None:
        _setup_default_show(monkeypatch, default_pot_id=None)
        result = CliRunner().invoke(
            pots.pot_app, ["default", "show", "--with-candidates"]
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "candidates" in payload
        assert "hint" not in payload


# ============================================================================
# Audit 22 — cloud group invocations return structured JSON
# ============================================================================


def _make_cloud_app() -> typer.Typer:
    """Wrap cloud_app inside a minimal root app that sets --json."""
    app = typer.Typer()

    @app.callback()
    def _root(
        json_: bool = typer.Option(False, "--json"),
    ) -> None:
        _common.set_json(json_)

    app.add_typer(cloud.cloud_app, name="cloud")
    return app


class TestCloudGroupJson:
    def test_cloud_group_json_returns_structured_error(self) -> None:
        app = _make_cloud_app()
        result = CliRunner().invoke(app, ["--json", "cloud"])

        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert payload["code"] == "not_implemented"
        assert "cloud" in payload["message"].lower()

    def test_cloud_skills_group_json_returns_structured_error(self) -> None:
        app = _make_cloud_app()
        result = CliRunner().invoke(app, ["--json", "cloud", "skills"])

        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert payload["code"] == "not_implemented"
        assert "skills" in payload["message"].lower()

    def test_cloud_leaf_command_still_works(self) -> None:
        app = _make_cloud_app()
        result = CliRunner().invoke(app, ["--json", "cloud", "status"])

        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert payload["code"] == "not_implemented"
        assert "login" in payload["message"].lower() or "status" in payload["message"].lower() or "cloud" in payload["message"].lower()

    def test_cloud_group_human_does_not_crash(self) -> None:
        app = _make_cloud_app()
        result = CliRunner().invoke(app, ["cloud"])
        assert result.exit_code != 0
        assert "not implemented" in result.output.lower() or "not_implemented" in result.output.lower()

    def test_cloud_skills_sync_still_works(self) -> None:
        app = _make_cloud_app()
        result = CliRunner().invoke(app, ["--json", "cloud", "skills", "sync"])

        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert payload["code"] == "not_implemented"
