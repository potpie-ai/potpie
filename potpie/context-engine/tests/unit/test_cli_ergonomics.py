"""Stage 5 CLI ergonomics: source shorthand, pot-resolution errors, templates.

Pins the harness-facing conveniences of the harness-led repo ingestion plan:
``source add repo .`` resolves before storing, ambiguous pot inference fails
with a structured error instead of guessing, ``source list`` exposes the
registered location, and ``graph mutation-template`` emits schema-only
skeletons that validate against the real semantic-mutation contract.
"""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import repo_location
from adapters.inbound.cli.commands import _common, graph, pots
from application.services.semantic_mutation_validator import validate_semantic_request
from domain.semantic_mutations import SemanticMutationRequest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_json(False)
    _common.set_host(None)


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
        self.added = []

    def list_pots(self):
        return self._pots

    def active_pot(self):
        return self._active

    def list_sources(self, *, pot_id):
        return self._sources.get(pot_id, [])

    def add_source(self, *, pot_id, kind, location, name=None):
        self.added.append({"pot_id": pot_id, "kind": kind, "location": location})
        return _Source(kind, name or location, location)


class _Host:
    def __init__(self, pots_service) -> None:
        self.pots = pots_service


# --- source add repo . / current --------------------------------------------


def test_source_add_repo_dot_resolves_before_storing(monkeypatch) -> None:
    monkeypatch.setattr(
        repo_location, "current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    pots_service = _Pots([_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True))
    _common.set_host(_Host(pots_service))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["add", "repo", "."])
    assert result.exit_code == 0, result.output
    assert pots_service.added[0]["location"] == "github.com/acme/shop"
    payload = json.loads(result.output)
    assert payload["location"] == "github.com/acme/shop"
    assert payload["registration_only"] is True


def test_source_add_repo_current_falls_back_to_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(repo_location, "current_git_remote", lambda cwd: None)
    monkeypatch.chdir(tmp_path)
    pots_service = _Pots([_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True))
    _common.set_host(_Host(pots_service))

    result = CliRunner().invoke(pots.source_app, ["add", "repo", "current"])
    assert result.exit_code == 0, result.output
    assert pots_service.added[0]["location"] == str(tmp_path.resolve())


def test_source_add_non_repo_kind_keeps_location_verbatim() -> None:
    pots_service = _Pots([_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True))
    _common.set_host(_Host(pots_service))

    result = CliRunner().invoke(pots.source_app, ["add", "linear", "ENG"])
    assert result.exit_code == 0, result.output
    assert pots_service.added[0]["location"] == "ENG"


def test_source_list_includes_location() -> None:
    src = _Source("repo", "shop", "github.com/acme/shop")
    pots_service = _Pots(
        [_Pot("p1", "shop", True)], {"p1": [src]}, active=_Pot("p1", "shop", True)
    )
    _common.set_host(_Host(pots_service))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["list"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sources"][0]["location"] == "github.com/acme/shop"


def test_source_add_targets_active_pot_even_when_repo_matches_other_pots(
    monkeypatch,
) -> None:
    """Registration establishes the mapping: other pots tracking the same repo
    must not divert (or block) `source add repo .` from the active pot."""
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    match = _Source("repo", "github.com/acme/shop")
    active = _Pot("p3", "fresh", True)
    pots_service = _Pots(
        [_Pot("p1", "shop"), _Pot("p2", "shop-fork"), active],
        {"p1": [match], "p2": [match]},
        active=active,
    )
    _common.set_host(_Host(pots_service))

    result = CliRunner().invoke(pots.source_app, ["add", "repo", "."])
    assert result.exit_code == 0, result.output
    assert pots_service.added[0]["pot_id"] == "p3"


# --- pot resolution errors ----------------------------------------------------


def test_resolve_pot_fails_structured_when_repo_matches_multiple_pots(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    match = _Source("repo", "github.com/acme/shop")
    pots_service = _Pots(
        [_Pot("p1", "shop"), _Pot("p2", "shop-fork")],
        {"p1": [match], "p2": [match]},
        active=None,
    )
    _common.set_host(_Host(pots_service))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["list"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["code"] == "ambiguous_pot"
    assert "shop" in payload["message"] and "shop-fork" in payload["message"]
    assert "--pot" in payload["recommended_next_action"]


def test_resolve_pot_prefers_active_when_among_multiple_matches(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    match = _Source("repo", "github.com/acme/shop")
    active = _Pot("p2", "shop-fork", True)
    pots_service = _Pots(
        [_Pot("p1", "shop"), active],
        {"p1": [match], "p2": [match]},
        active=active,
    )
    host = _Host(pots_service)
    _common.set_host(host)
    assert _common.resolve_pot_id(host) == "p2"


def test_resolve_pot_error_mentions_source_add_when_nothing_resolves(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_common, "_current_git_remote", lambda cwd: None)
    pots_service = _Pots([_Pot("p1", "shop")], {}, active=None)
    _common.set_host(_Host(pots_service))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["list"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["code"] == "no_active_pot"
    assert "source add repo ." in payload["recommended_next_action"]


# --- mutation templates --------------------------------------------------------


def _fill_placeholders(value):
    """Replace <placeholder> text with schema-conformant dummy values."""
    if isinstance(value, dict):
        return {k: _fill_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_placeholders(v) for v in value]
    if isinstance(value, str):
        if value == "<ISO-8601 timestamp>":
            return "2026-06-10T00:00:00+00:00"
        if value.startswith("<verb"):
            return "merged_pr"
        if value == "<pot-id>":
            return "pot-test"
        return re.sub(r"<[^>]+>", "x", value)
    return value


@pytest.mark.parametrize("kind", sorted(graph._MUTATION_TEMPLATES))
def test_mutation_template_validates_against_contract(kind: str) -> None:
    template = _fill_placeholders(graph._MUTATION_TEMPLATES[kind])
    request = SemanticMutationRequest.parse(template, pot_id="pot-test")
    plan = validate_semantic_request(request)
    assert plan.ok, f"{kind} template fails validation: {[i.message for i in plan.errors]}"
    # user_decision claims are medium-risk by contract, so the decision
    # template honestly lands in review_required; everything else auto-applies.
    expected = "review_required" if kind == "decision" else "apply"
    assert plan.decision == expected, f"{kind}: {plan.decision} != {expected}"


def test_mutation_template_command_emits_json() -> None:
    _common.set_json(True)
    result = CliRunner().invoke(graph.graph_app, ["mutation-template", "--kind", "repo-baseline"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result"]["kind"] == "repo-baseline"
    ops = payload["result"]["template"]["operations"]
    assert any(op.get("predicate") == "PROVIDES" for op in ops)


def test_mutation_template_unknown_kind_fails_with_next_action() -> None:
    _common.set_json(True)
    result = CliRunner().invoke(graph.graph_app, ["mutation-template", "--kind", "nope"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "unknown_template_kind"
    assert "repo-baseline" in payload["recommended_next_action"]
