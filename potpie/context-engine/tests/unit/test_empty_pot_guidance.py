"""Early empty-pot guidance on pot/source commands."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli.commands import _common, pots

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    _common.set_json(False)
    _common.set_host(None)
    yield
    _common.set_json(False)
    _common.set_host(None)


class _Pot:
    def __init__(self, pot_id: str, name: str, active: bool = False) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = active


class _Source:
    kind = "repo"
    name = "github.com/acme/shop"
    location = "github.com/acme/shop"

    def __init__(self, source_id: str = "src_1") -> None:
        self.source_id = source_id


class _Pots:
    def __init__(self) -> None:
        self.p1 = _Pot("p1", "empty", True)
        self.p2 = _Pot("p2", "populated")

    def active_pot(self):
        return self.p1

    def list_pots(self):
        return [self.p1, self.p2]

    def list_sources(self, *, pot_id: str):
        return [_Source(f"src_{pot_id}")]

    def create_pot(self, *, name: str, repo=None, use: bool = False):
        pot = _Pot("p-new", name, active=use)
        if use:
            self.p1.active = False
            pot.active = True
        return pot

    def use_pot(self, *, ref: str):
        return self.p1


class _Graph:
    def __init__(self, counts_by_pot: dict[str, dict[str, int]]) -> None:
        self._counts = counts_by_pot

    def data_plane_status(self, pot_id: str):
        counts = self._counts.get(pot_id, {"claims": 0, "entities": 0})
        return SimpleNamespace(counts=counts)


class _Host:
    def __init__(self) -> None:
        self.pots = _Pots()
        self.graph = _Graph({"p1": {"claims": 0}, "p2": {"claims": 82}})


def test_empty_pot_guidance_suggests_populated_sibling_pot(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    host = _Host()
    warnings = _common.empty_pot_guidance(host, "p1")
    assert any("p2" in warning and "82 claims" in warning for warning in warnings)
    assert any("harness-led ingestion" in warning for warning in warnings)


def test_source_list_emits_empty_pot_guidance_json(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    _common.set_json(True)
    _common.set_host(_Host())

    result = CliRunner().invoke(pots.source_app, ["list"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["warnings"]
    assert payload["recommended_next_action"]
    assert "harness-led ingestion" in payload["warnings"][-1]


def test_pot_create_emits_empty_pot_guidance(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    _common.set_json(True)
    _common.set_host(_Host())

    result = CliRunner().invoke(pots.pot_app, ["create", "fresh", "--use"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["warnings"]
    assert "0 claims" in payload["recommended_next_action"]
