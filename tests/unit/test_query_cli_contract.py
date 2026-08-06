"""CLI contract for the narrow-lookup door: ``potpie search``.

Pins the two affordances P1-1 found missing. Bare ``search`` never returned a
document — it resolves intent ``unknown``, which excluded ``docs`` — and the
response offered nothing to recover with: ``--include`` had no help text at
all, so an agent reaching for it guessed subgraph names, and there was no
``--intent`` flag to narrow the lookup with instead.
"""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from potpie.cli.commands import _common, query
from potpie_context_core.agent_context_port import (
    DEFAULT_INTENT_INCLUDES,
    READER_BACKED_INCLUDES,
)
from potpie_context_core.agent_envelope import AgentEnvelope, CoverageReport

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_json(False)
    _common.set_host(None)


class _Pot:
    pot_id = "p"
    name = "default"
    active = True


class _Pots:
    def active_pot(self):
        return _Pot()

    def list_pots(self):
        return [_Pot()]


class _AgentContext:
    """Records the request the CLI built, and answers a minimal envelope."""

    def __init__(self) -> None:
        self.requests: list[object] = []

    def search(self, request):
        self.requests.append(request)
        return AgentEnvelope(
            pot_id=request.pot_id,
            intent=request.intent or "unknown",
            items=(),
            coverage=(CoverageReport(include="docs", status="empty"),),
        )


class _Host:
    def __init__(self) -> None:
        self.pots = _Pots()
        self.agent_context = _AgentContext()


def _app() -> typer.Typer:
    app = typer.Typer()
    query.register(app)
    return app


def _host() -> _Host:
    host = _Host()
    _common.set_host(host)
    return host


def test_search_help_names_the_include_families():
    """An agent with no list in front of it guessed ``--include knowledge``
    from the subgraph name; the values were nowhere in the CLI."""
    result = CliRunner().invoke(_app(), ["search", "--help"])

    assert result.exit_code == 0
    text = " ".join(result.stdout.split())
    for family in READER_BACKED_INCLUDES - {"raw_graph"}:
        assert family in text


def test_search_help_names_the_intents():
    result = CliRunner().invoke(_app(), ["search", "--help"])

    text = " ".join(result.stdout.split())
    for intent in DEFAULT_INTENT_INCLUDES:
        assert intent in text


def test_search_passes_an_explicit_intent_through():
    host = _host()
    _common.set_json(True)

    result = CliRunner().invoke(
        _app(), ["search", "pool exhausted", "--intent", "debugging"]
    )

    assert result.exit_code == 0, result.stdout
    assert host.agent_context.requests[0].intent == "debugging"
    assert json.loads(result.stdout)["intent"] == "debugging"


def test_search_leaves_the_intent_unset_by_default():
    """Unset, not the string 'unknown': normalization is the service's job, and
    the CLI has no business duplicating that vocabulary."""
    host = _host()

    result = CliRunner().invoke(_app(), ["search", "liability cap"])

    assert result.exit_code == 0, result.stdout
    assert host.agent_context.requests[0].intent is None


def test_bare_search_asks_for_documents():
    """The one-line half of P1-1, pinned where it is observable: the default
    include set behind a bare search has to contain ``docs``."""
    assert "docs" in DEFAULT_INTENT_INCLUDES["unknown"]
