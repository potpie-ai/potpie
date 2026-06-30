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
import typer
from typer.testing import CliRunner

from adapters.inbound.cli import repo_location
from adapters.inbound.cli.commands import _common, graph, pots, ui
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
        self.repo_defaults = {}

    def list_pots(self):
        return self._pots

    def active_pot(self):
        return self._active

    def list_sources(self, *, pot_id):
        return self._sources.get(pot_id, [])

    def add_source(self, *, pot_id, kind, location, name=None):
        self.added.append({"pot_id": pot_id, "kind": kind, "location": location})
        return _Source(kind, name or location, location)

    def repo_default(self, *, repo):
        return self.repo_defaults.get(repo)

    def set_repo_default(self, *, repo, pot_id):
        self.repo_defaults[repo] = pot_id

    def clear_repo_default(self, *, repo):
        return self.repo_defaults.pop(repo, None) is not None

    def list_repo_defaults(self):
        return dict(self.repo_defaults)


class _Host:
    def __init__(self, pots_service, daemon=None) -> None:
        self.pots = pots_service
        self.daemon = daemon
        self.graph = None


class _Daemon:
    def ensure(self):
        return None

    def discovery(self):
        return {"base_url": "http://127.0.0.1:8765"}


class _GraphStatus:
    def __init__(self, counts) -> None:
        self.counts = counts


class _Graph:
    def __init__(self, counts_by_pot) -> None:
        self._counts_by_pot = counts_by_pot

    def data_plane_status(self, pot_id):
        return _GraphStatus(self._counts_by_pot.get(pot_id, {}))


# --- source add repo . / current --------------------------------------------


def test_source_add_repo_dot_resolves_before_storing(monkeypatch) -> None:
    monkeypatch.setattr(
        repo_location, "current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    pots_service = _Pots(
        [_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True)
    )
    _common.set_host(_Host(pots_service))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["add", "repo", "."])
    assert result.exit_code == 0, result.output
    assert pots_service.added[0]["location"] == "github.com/acme/shop"
    payload = json.loads(result.output)
    assert payload["location"] == "github.com/acme/shop"
    assert payload["repo_default_set"] is True
    assert payload["registration_only"] is True
    assert pots_service.repo_defaults == {"github.com/acme/shop": "p1"}


def test_source_add_repo_current_falls_back_to_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(repo_location, "current_git_remote", lambda cwd: None)
    monkeypatch.chdir(tmp_path)
    pots_service = _Pots(
        [_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True)
    )
    _common.set_host(_Host(pots_service))

    result = CliRunner().invoke(pots.source_app, ["add", "repo", "current"])
    assert result.exit_code == 0, result.output
    assert pots_service.added[0]["location"] == str(tmp_path.resolve())
    assert pots_service.repo_defaults == {str(tmp_path.resolve()): "p1"}


def test_source_add_non_repo_kind_keeps_location_verbatim() -> None:
    pots_service = _Pots(
        [_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True)
    )
    _common.set_host(_Host(pots_service))

    result = CliRunner().invoke(pots.source_app, ["add", "linear", "ENG"])
    assert result.exit_code == 0, result.output
    assert pots_service.added[0]["location"] == "ENG"
    assert pots_service.repo_defaults == {}


def test_source_add_repo_no_default_skips_repo_default(monkeypatch) -> None:
    monkeypatch.setattr(
        repo_location, "current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    pots_service = _Pots(
        [_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True)
    )
    _common.set_host(_Host(pots_service))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["add", "repo", ".", "--no-default"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["repo_default_set"] is False
    assert pots_service.repo_defaults == {}


def test_source_add_repo_plain_output_shows_next_action_for_empty_pot(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        repo_location, "current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    pots_service = _Pots(
        [_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True)
    )
    host = _Host(pots_service)
    host.graph = _Graph({"p1": {"claims": 0, "entities": 0}})
    _common.set_host(host)

    result = CliRunner().invoke(pots.source_app, ["add", "repo", "."])

    assert result.exit_code == 0, result.output
    assert "no ingestion or scan started" in result.output
    assert "next actions:" in result.output
    assert "check linked pots:" in result.output
    assert "switch if one is already populated:" in result.output
    assert "potpie graph mutation-template --kind repo-baseline" in result.output


def test_source_add_repo_json_omits_next_action_when_pot_has_claims(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        repo_location, "current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    pots_service = _Pots(
        [_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True)
    )
    host = _Host(pots_service)
    host.graph = _Graph({"p1": {"claims": 12, "entities": 4}})
    _common.set_host(host)
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["add", "repo", "."])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["registration_only"] is True
    assert "recommended_next_action" not in payload


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


def test_source_list_plain_output_includes_location() -> None:
    src = _Source("repo", "shop", "github.com/acme/shop")
    pots_service = _Pots(
        [_Pot("p1", "shop", True)], {"p1": [src]}, active=_Pot("p1", "shop", True)
    )
    _common.set_host(_Host(pots_service))

    result = CliRunner().invoke(pots.source_app, ["list"])

    assert result.exit_code == 0, result.output
    assert "repo: github.com/acme/shop (src_shop)" in result.output


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


def test_resolve_pot_uses_repo_default_before_ambiguous_matches(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    match = _Source("repo", "github.com/acme/shop")
    pots_service = _Pots(
        [_Pot("p1", "shop"), _Pot("p2", "shop-fork")],
        {"p1": [match], "p2": [match]},
        active=None,
    )
    pots_service.repo_defaults["github.com/acme/shop"] = "p2"
    host = _Host(pots_service)

    assert _common.resolve_pot_id(host) == "p2"


def test_pot_default_set_and_clear_current_repo(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    pots_service = _Pots([_Pot("p1", "shop")], {}, active=None)
    _common.set_host(_Host(pots_service))
    _common.set_json(True)

    result = CliRunner().invoke(pots.pot_app, ["default", "set", "p1"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["repo"] == "github.com/acme/shop"
    assert payload["default_pot"]["id"] == "p1"
    assert pots_service.repo_defaults == {"github.com/acme/shop": "p1"}

    result = CliRunner().invoke(pots.pot_app, ["default", "clear"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cleared"] is True
    assert pots_service.repo_defaults == {}


def test_pot_linked_lists_candidates_and_default(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    match = _Source("repo", "github.com/acme/shop")
    pots_service = _Pots(
        [_Pot("p1", "shop"), _Pot("p2", "shop-fork")],
        {"p1": [match], "p2": [match]},
        active=None,
    )
    pots_service.repo_defaults["github.com/acme/shop"] = "p2"
    _common.set_host(_Host(pots_service))
    _common.set_json(True)

    result = CliRunner().invoke(pots.pot_app, ["linked"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["repo"] == "github.com/acme/shop"
    assert payload["default_pot_id"] == "p2"
    assert [row["pot_id"] for row in payload["candidates"]] == ["p1", "p2"]
    assert payload["candidates"][1]["default"] is True


def test_ui_pot_option_opens_selected_pot_url(monkeypatch) -> None:
    monkeypatch.setattr(ui, "_probe_ui", lambda base: None)
    pots_service = _Pots(
        [_Pot("p1", "shop", True)], {}, active=_Pot("p1", "shop", True)
    )
    _common.set_host(_Host(pots_service, daemon=_Daemon()))
    _common.set_json(True)

    app = typer.Typer()
    ui.register(app)
    result = CliRunner().invoke(app, ["--pot", "p1", "--no-open"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pot_id"] == "p1"
    assert payload["url"] == "http://127.0.0.1:8765/ui?pot=p1"


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
    assert plan.ok, (
        f"{kind} template fails validation: {[i.message for i in plan.errors]}"
    )
    # user_decision claims are medium-risk by contract, so the decision
    # template honestly lands in review_required; everything else auto-applies.
    expected = "review_required" if kind == "decision" else "apply"
    assert plan.decision == expected, f"{kind}: {plan.decision} != {expected}"


def test_mutation_template_command_emits_json() -> None:
    _common.set_json(True)
    result = CliRunner().invoke(
        graph.graph_app, ["mutation-template", "--kind", "repo-baseline"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result"]["kind"] == "repo-baseline"
    ops = payload["result"]["template"]["operations"]
    assert any(op.get("predicate") == "PROVIDES" for op in ops)


def test_mutation_template_unknown_kind_fails_with_next_action() -> None:
    _common.set_json(True)
    result = CliRunner().invoke(
        graph.graph_app, ["mutation-template", "--kind", "nope"]
    )
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "unknown_template_kind"
    assert "repo-baseline" in payload["recommended_next_action"]
