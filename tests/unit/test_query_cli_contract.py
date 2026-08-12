"""CLI contract for the three agent doors: ``resolve`` / ``search`` / ``record``.

Pins the two affordances P1-1 found missing. Bare ``search`` never returned a
document — it resolves intent ``unknown``, which excluded ``docs`` — and the
response offered nothing to recover with: ``--include`` had no help text at
all, so an agent reaching for it guessed subgraph names, and there was no
``--intent`` flag to narrow the lookup with instead.

It also pins the four defects the 2026-08-11 audit found on the same three
commands, all of them the same shape — the CLI answering for something it did
not check or did not carry:

- ``--type decision`` / ``--type preference``, the two uses this command's own
  ``--type`` help advertises, were impossible to execute: both validate a
  structured field and there was no flag that could supply one (S10-17);
- a rejected write printed a receipt at exit 0, because ``accepted`` and
  ``detail`` never left the CLI (S10-14);
- the read envelope was hand-serialised and lost six fields, ``candidate_key``
  — the one an agent dedupes on — among them (S10-23);
- ``--mode`` and ``--intent`` were normalised rather than checked, so a typo
  silently changed the depth of the read or which reader families answered
  (S10-20/21).

The write tests run against a real ``HostShell`` over a real in-memory graph
backend: the record→semantic bridge and its validation are the thing under
test, so stubbing the service would assert nothing. Only the storage adapter is
faked, and only in the one test that needs the store to refuse.
"""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from potpie.cli.commands import _common, query
from potpie_context_core.agent_context_port import (
    CONTEXT_INTENTS,
    DEFAULT_INTENT_INCLUDES,
    READER_BACKED_INCLUDES,
)
from potpie_context_core.agent_envelope import (
    AgentEnvelope,
    CoverageReport,
    EvidenceItem,
)
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.reconciliation import MutationResult, MutationSummary
from potpie_context_core.source_references import RESOLVE_MODES
from potpie_context_engine.adapters.outbound.graph.backends import in_memory_backend
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.bootstrap.host_wiring import build_host_shell

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


class _Receipt:
    """A ``RecordReceipt`` shape — including the two fields the CLI dropped."""

    accepted = True
    status = "recorded"
    record_id = "rec_1"
    mutations_applied = 1
    detail = None


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

    def resolve(self, request):
        return self.search(request)

    def record(self, request):
        self.requests.append(request)
        return _Receipt()


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


# --- a request that says nothing is refused, not answered --------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["resolve", ""],
        ["resolve", "   "],
        ["search", ""],
        ["record", "--type", "fix", "--summary", ""],
        ["record", "--type", "", "--summary", "the retry needs jitter"],
    ],
    ids=lambda a: "-".join(part or "<empty>" for part in a),
)
def test_an_empty_argument_is_refused_rather_than_answered(argv):
    """An empty argument is an absent request, not a broader one.

    ``potpie search ''`` returned a ranked envelope with a confidence score
    attached to a query nobody made, and ``record --summary ''`` wrote a durable
    row nothing can ever retrieve — both at exit 0, both looking like results.
    """
    host = _host()
    _common.set_json(True)

    result = CliRunner().invoke(_app(), argv)

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "validation_error"
    assert "cannot be empty" in payload["message"]
    assert host.agent_context.requests == []


def test_a_malformed_scope_is_refused_instead_of_dropped():
    """A scope the CLI cannot read is a refusal, never a smaller filter.

    ``record --scope service`` (no colon) dropped the pair on the floor and
    wrote an *unscoped* claim at exit 0 — the caller's narrowing silently became
    no narrowing at all, and the wrong data is now in the graph.
    """
    host = _host()
    _common.set_json(True)

    result = CliRunner().invoke(
        _app(),
        [
            "record",
            "--type",
            "fix",
            "--summary",
            "retry needs jitter",
            "--scope",
            "service",
        ],
    )

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "validation_error"
    assert "--scope" in payload["message"]
    assert host.agent_context.requests == []


def test_a_well_formed_scope_still_reaches_the_request():
    host = _host()

    result = CliRunner().invoke(
        _app(),
        [
            "record",
            "--type",
            "fix",
            "--summary",
            "retry needs jitter",
            "--scope",
            "service:inventory-svc, repo:acme/shop",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert host.agent_context.requests[0].scope == {
        "service": "inventory-svc",
        "repo": "acme/shop",
    }


# --- S10-20/21: a typo is refused, never quietly normalised ------------------


@pytest.mark.parametrize(
    "argv,argument",
    [
        (["resolve", "add rate limiting", "--mode", "blanced"], "--mode"),
        (["resolve", "add rate limiting", "--intent", "debuging"], "--intent"),
        (["search", "pool exhausted", "--intent", "debuging"], "--intent"),
    ],
    ids=["resolve-mode", "resolve-intent", "search-intent"],
)
def test_a_typo_in_a_closed_vocabulary_is_refused(argv, argument):
    """Both normalisations are silent-wrong-answer bugs at a keyboard.

    ``--mode blanced`` fell back to ``fast`` — a shallower read than the one
    that was asked for, reported as though it were that read — and
    ``--intent debuging`` fell back to ``unknown``, which queries a different
    set of reader families entirely. Neither said so anywhere in the answer.
    """
    host = _host()
    _common.set_json(True)

    result = CliRunner().invoke(_app(), argv)

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "validation_error"
    assert argument in payload["message"]
    # The refusal has to carry the vocabulary; a closed set nobody can see is
    # the reason the typo was reachable in the first place.
    allowed = RESOLVE_MODES if argument == "--mode" else CONTEXT_INTENTS
    assert set(payload["detail"]["allowed"]) == set(allowed)
    # And nothing was read against the wrong depth/families on the way out.
    assert host.agent_context.requests == []


def test_a_valid_mode_and_intent_still_reach_the_request():
    host = _host()

    result = CliRunner().invoke(
        _app(),
        ["resolve", "cap the retry budget", "--mode", "deep", "--intent", "review"],
    )

    assert result.exit_code == 0, result.stdout
    request = host.agent_context.requests[0]
    assert (request.mode, request.intent) == ("deep", "review")


# --- S10-23: the envelope serialises itself ---------------------------------


def test_the_read_envelope_is_emitted_by_its_own_serializer():
    """Six fields were dropped by a hand-rolled payload, one of them load-bearing.

    ``candidate_key`` is what an agent dedupes on across calls; without it two
    reads of the same evidence are indistinguishable from two findings. The
    assertion is parity with ``to_dict()`` rather than a field list, so the next
    field added to the envelope cannot go missing here again.
    """
    host = _host()
    envelope = AgentEnvelope(
        pot_id="p",
        intent="feature",
        items=(
            EvidenceItem(
                include="coding_preferences",
                candidate_key="claim:preference:jitter-retries",
                score=0.82,
                payload={"fact": "retries need a jittered backoff"},
                coverage_status="complete",
                breakdown={"semantic": 0.6, "recency": 0.22},
            ),
        ),
        coverage=(
            CoverageReport(
                include="coding_preferences",
                status="complete",
                candidate_pool=7,
                graph_view="decisions.preferences",
            ),
        ),
        overall_confidence="high",
        metadata={"mode": "fast"},
    )
    host.agent_context.resolve = lambda request: envelope
    _common.set_json(True)

    result = CliRunner().invoke(_app(), ["resolve", "add rate limiting"])

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout) == envelope.to_dict()


# --- S10-17 / S10-14: writes, against a real graph service -------------------


@pytest.fixture()
def real_host(tmp_path, monkeypatch):
    """A real ``HostShell`` on a real (temp) home over a real graph backend.

    The record→semantic bridge and its per-record-type validation are exactly
    what these cases are about, so nothing between the CLI and the claim store
    is stubbed.
    """
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    host = build_host_shell(backend=InMemoryGraphBackend())
    host.pots.create_pot(name="p", use=True)
    _common.set_host(host)
    return host


def _claims(host) -> list:
    """Every claim in the active pot, read straight off the store."""
    pot = host.pots.active_pot()
    return list(
        host.backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot.pot_id))
    )


def test_a_decision_is_impossible_to_record_without_the_field_it_requires(real_host):
    """The headline use, refused with no way to comply.

    ``decision`` validates a non-empty ``rationale``; ``preference`` validates
    ``policy_kind``. Neither had a flag, so both of the uses ``--type``'s own
    help advertises came back as a validation error naming a field the CLI
    could not accept — a dead end, not a correction.
    """
    _common.set_json(True)

    result = CliRunner().invoke(
        _app(),
        ["record", "--type", "decision", "--summary", "use redis for rate limits"],
    )

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    payload = json.loads(result.stdout)
    assert "rationale" in payload["message"]
    # The refusal names the flag that closes it, on both transports: the daemon
    # RPC re-raises a remote validation failure as a plain ValueError and
    # re-attaches this string.
    assert "--detail" in (payload["recommended_next_action"] or "")


@pytest.mark.parametrize(
    "argv",
    [
        [
            "record",
            "--type",
            "decision",
            "--summary",
            "use redis for rate limits",
            "--detail",
            "rationale=it is already deployed, and the counters are cheap",
        ],
        [
            "record",
            "--type",
            "preference",
            "--summary",
            "always jitter retries",
            "--detail",
            "policy_kind=resilience",
        ],
    ],
    ids=["decision", "preference"],
)
def test_the_headline_record_types_are_executable(real_host, argv):
    _common.set_json(True)

    result = CliRunner().invoke(_app(), argv)

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["accepted"] is True
    assert payload["mutations_applied"] >= 1
    # Not just a receipt: the claim is in the graph.
    assert _claims(real_host)


def test_a_repeated_detail_key_builds_the_list_shaped_fields(real_host):
    """``alternatives_rejected`` is a list, and a shell has no list literal."""
    _common.set_json(True)

    result = CliRunner().invoke(
        _app(),
        [
            "record",
            "--type",
            "decision",
            "--summary",
            "use redis for rate limits",
            "--detail",
            "rationale=already deployed",
            "--detail",
            "alternatives_rejected=postgres",
            "--detail",
            "alternatives_rejected=memcached",
        ],
    )

    assert result.exit_code == 0, result.stdout
    pot = real_host.pots.active_pot()
    decisions = [
        dict(node.properties or {})
        for node in real_host.backend.inspection.slice(
            pot_id=pot.pot_id, filter_=ClaimQueryFilter(pot_id=pot.pot_id)
        ).nodes
        if "Decision" in (node.labels or ())
    ]
    assert [d.get("alternatives_rejected") for d in decisions] == [
        ["postgres", "memcached"]
    ]


def test_a_malformed_detail_is_refused_instead_of_dropped(real_host):
    _common.set_json(True)

    result = CliRunner().invoke(
        _app(),
        [
            "record",
            "--type",
            "decision",
            "--summary",
            "use redis for rate limits",
            "--detail",
            "rationale",
        ],
    )

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "validation_error"
    assert "--detail" in payload["message"]
    assert _claims(real_host) == []


def test_a_record_the_store_refuses_is_not_reported_as_a_write(
    real_host, monkeypatch
) -> None:
    """The write did not land, so the command must not answer as though it did.

    Only the storage adapter is faked — everything from the CLI down through the
    record→semantic bridge, validation and the mutation plan is the real thing;
    the claim store is the one part that has to refuse. The receipt already
    carried ``accepted=False`` and the store's reason in ``detail``; the CLI
    dropped both and printed ``rejected: <id> (0 mutations)`` at exit 0.
    """

    def _refuse(self, plan, *, expected_pot_id, **_kwargs):
        return MutationResult(
            ok=False,
            mutation_id="m_refused",
            mutation_summary=MutationSummary(),
            error="claim store refused the batch",
        )

    monkeypatch.setattr(in_memory_backend._Mutation, "apply", _refuse)
    _common.set_json(True)

    result = CliRunner().invoke(
        _app(),
        [
            "record",
            "--type",
            "decision",
            "--summary",
            "use redis for rate limits",
            "--detail",
            "rationale=already deployed",
        ],
    )

    assert result.exit_code != 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["accepted"] is False
    assert payload["ok"] is False
    assert payload["status"] == "rejected"
    assert payload["mutations_applied"] == 0
    assert "refused" in (payload["detail"] or "")
