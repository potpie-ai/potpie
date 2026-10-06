"""CLI contract for the three agent doors: ``resolve`` / ``search`` / ``record``.

Pins what each door carries to the engine and what it shows back:

- ``--include`` and ``--intent`` are discoverable from ``--help``, unknown
  includes and typo'd closed-vocabulary values are refused before anything is
  read, and an unset intent stays unset so Potpie can infer it;
- ``--limit`` is a total result budget passed to the engine;
- the read envelope is emitted by its own serializer, and the human view
  renders each claim as ``subject PREDICATE object · fact`` with the cut
  announced;
- ``record --detail key=value`` makes the structured record types executable,
  and a refused write is neither reported as stored nor exits 0.

The write tests run against a real local runtime over the in-memory graph
backend: the record→semantic bridge and its validation are what is under test,
so only the storage adapter is faked, in the one test that needs the store to
refuse.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from potpie.cli.commands import _common, query
from potpie.runtime.composition import build_local_runtime
from potpie_context_engine.adapters.outbound.graph.backends import in_memory_backend
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.agent_context_port import (
    CONTEXT_INTENTS,
    DEFAULT_INTENT_INCLUDES,
    READER_BACKED_INCLUDES,
)
from potpie_context_engine.core.agent_envelope import (
    AgentEnvelope,
    CoverageReport,
    EvidenceItem,
)
from potpie_context_engine.core.context_records import REQUIRED_DETAIL_KEYS
from potpie_context_engine.core.ontology import PUBLIC_RECORD_TYPES
from potpie_context_engine.core.ports.agent_context import RecordReceipt
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationResult, MutationSummary
from potpie_context_engine.core.source_references import RESOLVE_MODES

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_json(False)


class _Pot:
    pot_id = "p"
    name = "default"
    active = True


class _Pots:
    def active_pot(self):
        return _Pot()

    def list_pots(self):
        return [_Pot()]

    def list_sources(self, *, pot_id):
        return []


class _AgentContext:
    """Records the request the engine boundary built; answers a minimal envelope."""

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
        return RecordReceipt(
            pot_id=request.pot_id,
            record_type=request.record_type,
            accepted=True,
            record_id="rec_1",
            mutations_applied=1,
        )


class _Host:
    def __init__(self) -> None:
        self.pots = _Pots()
        self.agent_context = _AgentContext()
        self.graph = SimpleNamespace(catalog=lambda request: SimpleNamespace(views=()))
        self.graph_workbench = SimpleNamespace()
        self.backend = SimpleNamespace(profile="memory")
        self.nudge = SimpleNamespace()


def _app() -> typer.Typer:
    app = typer.Typer()
    query.register(app)
    return app


def _host() -> _Host:
    host = _Host()
    _common.set_runtime(host)
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


def test_search_limit_is_a_total_budget_passed_to_the_engine():
    host = _host()
    result = CliRunner().invoke(_app(), ["search", "graph", "--limit", "1"])

    assert result.exit_code == 0, result.stdout
    assert host.agent_context.requests[0].max_items == 1


@pytest.mark.parametrize("command", ["resolve", "search"])
def test_a_non_positive_limit_is_refused(command):
    host = _host()
    _common.set_json(True)

    result = CliRunner().invoke(_app(), [command, "graph", "--limit", "0"])

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    assert "--limit must be >= 1" in json.loads(result.stdout)["message"]
    assert host.agent_context.requests == []


def test_search_leaves_the_intent_unset_by_default():
    """Unset, not the string 'unknown': choosing the intent is Potpie's job."""
    host = _host()

    result = CliRunner().invoke(_app(), ["search", "liability cap"])

    assert result.exit_code == 0, result.stdout
    assert host.agent_context.requests[0].intent is None


def test_bare_search_asks_for_documents():
    """The default include set behind a bare search has to contain ``docs``."""
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
    """An empty argument is an absent request, not a broader one."""
    host = _host()
    _common.set_json(True)

    result = CliRunner().invoke(_app(), argv)

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "validation_error"
    assert "cannot be empty" in payload["message"]
    assert host.agent_context.requests == []


def test_a_malformed_scope_is_refused_instead_of_dropped():
    """``record --scope service`` (no colon) must not write an unscoped claim."""
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


# --- a typo in a closed vocabulary is refused, never quietly normalised ------


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
    """``--mode blanced`` would read as ``fast`` and ``--intent debuging`` as
    ``unknown``: a different read, reported as the one asked for."""
    host = _host()
    _common.set_json(True)

    result = CliRunner().invoke(_app(), argv)

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "validation_error"
    assert argument in payload["message"]
    allowed = RESOLVE_MODES if argument == "--mode" else CONTEXT_INTENTS
    assert set(payload["detail"]["allowed"]) == set(allowed)
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


# --- the envelope serialises itself -------------------------------------------


def test_the_read_envelope_is_emitted_by_its_own_serializer():
    """Parity with ``to_dict()`` rather than a field list, so ``candidate_key``
    (what an agent dedupes on) and every later envelope field reach the caller."""
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


# --- writes, against a real local runtime ---------------------------------------


@pytest.fixture()
def real_runtime(tmp_path, monkeypatch):
    """A real local runtime on a temp home over a real in-memory graph backend."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runtime = build_local_runtime(backend=InMemoryGraphBackend())
    runtime.engine.pots.create_pot(name="p", use=True)
    _common.set_runtime(runtime)
    return runtime


def _claims(runtime) -> list:
    """Every claim in the active pot, read straight off the store."""
    pot = runtime.engine.pots.active_pot()
    return list(
        runtime.engine.backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=pot.pot_id)
        )
    )


def test_a_decision_without_its_required_detail_names_the_flag(real_runtime):
    """``decision`` validates a non-empty ``rationale``; the refusal names the
    flag that supplies it."""
    _common.set_json(True)

    result = CliRunner().invoke(
        _app(),
        ["record", "--type", "decision", "--summary", "use redis for rate limits"],
    )

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    payload = json.loads(result.stdout)
    assert "rationale" in payload["message"]
    assert "--detail" in (payload["recommended_next_action"] or "")
    assert _claims(real_runtime) == []


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
def test_the_structured_record_types_are_executable(real_runtime, argv):
    _common.set_json(True)

    result = CliRunner().invoke(_app(), argv)

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["accepted"] is True
    assert payload["mutations_applied"] >= 1
    # Not just a receipt: the claim is in the graph.
    assert _claims(real_runtime)


def test_a_repeated_detail_key_builds_the_list_shaped_fields(real_runtime):
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
    pot = real_runtime.engine.pots.active_pot()
    decisions = [
        dict(node.properties or {})
        for node in real_runtime.engine.backend.inspection.slice(
            pot_id=pot.pot_id, filter_=ClaimQueryFilter(pot_id=pot.pot_id)
        ).nodes
        if "Decision" in (node.labels or ())
    ]
    assert [d.get("alternatives_rejected") for d in decisions] == [
        ["postgres", "memcached"]
    ]


def test_a_malformed_detail_is_refused_instead_of_dropped(real_runtime):
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
    assert _claims(real_runtime) == []


def test_a_record_the_store_refuses_is_not_reported_as_a_write(
    real_runtime, monkeypatch
) -> None:
    """The receipt carries ``accepted=False`` and the store's reason; the CLI
    shows both and exits non-zero instead of printing a receipt at exit 0."""

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
    assert payload["status"] == "rejected"
    assert payload["mutations_applied"] == 0
    assert "refused" in (payload["detail"] or "")


# --- the human rendering carries the triple, the cut, and the source ----------


def _claim(
    include: str,
    *,
    key: str,
    subject: str = "service:inventory-svc",
    predicate: str = "USES",
    obj: str = "database:postgres",
    fact: str | None = "inventory keeps its counts in postgres",
    score: float = 0.82,
) -> EvidenceItem:
    return EvidenceItem(
        include=include,
        candidate_key=key,
        score=score,
        payload={
            "claim_key": key,
            "subject_key": subject,
            "predicate": predicate,
            "object_key": obj,
            "truth": "agent_claim",
            "fact": fact,
        },
        coverage_status="complete",
    )


def _envelope(*items: EvidenceItem, **metadata) -> AgentEnvelope:
    return AgentEnvelope(
        pot_id="p",
        intent="debugging",
        items=tuple(items),
        coverage=(),
        overall_confidence="medium",
        metadata=metadata,
    )


def test_resolve_leaves_the_intent_unset_by_default():
    """Unset, so Potpie infers it from the task and says so. A ``feature``
    default hid ``prior_bugs`` and ``timeline`` from every why-question."""
    host = _host()

    result = CliRunner().invoke(_app(), ["resolve", "why is stock stale"])

    assert result.exit_code == 0, result.stdout
    assert host.agent_context.requests[0].intent is None


def test_resolve_help_says_the_intent_is_inferred():
    result = CliRunner().invoke(_app(), ["resolve", "--help"])

    assert "inferred" in " ".join(result.stdout.split())


def test_a_claim_line_carries_the_triple_and_the_fact():
    text = query._envelope_human(_envelope(_claim("infra_topology", key="claim:1")))

    assert text.splitlines()[1] == (
        "  • [infra_topology] service:inventory-svc USES database:postgres"
        " · inventory keeps its counts in postgres (agent_claim, 0.82)"
    )


def test_every_claim_line_states_a_triple():
    """No claim row without subject, predicate and object; the evidence note
    alone is often a file path or a quoted README line."""
    env = _envelope(
        *[_claim("prior_bugs", key=f"claim:{n}", fact=None) for n in range(4)]
    )

    lines = query._envelope_human(env).splitlines()[1:]
    assert len(lines) == 4
    for line in lines:
        assert " USES " in line, line


def test_a_claim_two_families_found_is_one_line_naming_both():
    env = _envelope(
        _claim("prior_bugs", key="claim:same"),
        _claim("timeline", key="claim:same", score=0.4),
    )

    lines = query._envelope_human(env).splitlines()
    assert lines[0].endswith("items=2")
    assert len(lines) == 2
    assert lines[1].startswith("  • [prior_bugs, timeline] ")


def test_the_cut_is_announced_not_silent():
    env = _envelope(*[_claim("decisions", key=f"claim:{n}") for n in range(13)])

    lines = query._envelope_human(env).splitlines()
    assert len([line for line in lines if line.startswith("  • ")]) == 10
    assert lines[-1] == "  … +3 more (use --json)"


def test_human_envelope_renders_bounded_kind_details_and_fetch_guidance():
    claim = _claim("prior_bugs", key="claim:fix")
    item = EvidenceItem(
        include=claim.include,
        candidate_key=claim.candidate_key,
        score=claim.score,
        payload={
            **dict(claim.payload),
            "details": {
                "root_cause": "Leaked retry sockets",
                "fix_steps": ["Close sockets in finally"],
                "omitted": {"fix_steps": {"items": 2}},
            },
            "follow_up_commands": {
                "full_entity": "potpie graph neighborhood --entity fix:checkout --pot p"
            },
        },
        coverage_status=claim.coverage_status,
    )

    text = query._envelope_human(_envelope(item))

    assert "root_cause: Leaked retry sockets" in text
    assert 'fix_steps: ["Close sockets in finally"]' in text
    assert "fetch_more:" in text
    assert "potpie graph neighborhood --entity fix:checkout --pot p" in text


def test_budget_and_match_notices_are_rendered_in_the_header():
    env = _envelope(
        _claim("decisions", key="claim:1"),
        searched_families=["decisions", "docs"],
        match_status="no_exact_match",
        exact_identifier={"display": "PR #42"},
        total_result_budget=1,
        omitted_by_total_budget={"decisions": 0, "docs": 2},
        more_results_available=True,
    )

    text = query._envelope_human(env)

    assert "searched=decisions, docs match=no_exact_match" in text
    assert "total limit 1 omitted 2 results (docs 2)" in text
    assert "more results available" in text
    assert "! no exact match for PR #42" in text


def test_an_inferred_intent_is_marked_in_the_header():
    text = query._envelope_human(_envelope(intent_source="inferred"))

    assert text.startswith("pot=p intent=debugging (inferred) ")


def test_an_explicit_intent_is_not_marked():
    text = query._envelope_human(_envelope(intent_source="explicit"))

    assert text.startswith("pot=p intent=debugging confidence=")


def test_a_repeated_bug_summary_prints_once():
    """A bug recorded with the same text for summary and symptom printed twice."""
    env = _envelope(
        _claim(
            "prior_bugs",
            key="claim:bug",
            subject="bug_pattern:stale-stock",
            predicate="REPRODUCES",
            obj="service:inventory-svc",
            fact="stock counts stale after deploy • stock counts stale after deploy",
        )
    )

    line = query._envelope_human(env).splitlines()[1]
    assert line.count("stock counts stale after deploy") == 1


def test_an_item_with_nothing_to_say_is_neither_printed_nor_counted():
    empty = EvidenceItem(
        include="owners",
        candidate_key="",
        score=0.1,
        payload={},
        coverage_status="sparse",
    )
    env = _envelope(_claim("decisions", key="claim:1"), empty)

    lines = query._envelope_human(env).splitlines()
    assert len(lines) == 2
    assert "more" not in lines[-1]


def test_record_help_names_the_required_detail_per_type():
    result = CliRunner().invoke(_app(), ["record", "--help"])

    assert result.exit_code == 0
    # Rich wraps long option help across panel borders; drop the box glyphs
    # before joining so a phrase split over two lines still matches.
    text = " ".join(result.stdout.replace("│", " ").split())
    for record_type, keys in REQUIRED_DETAIL_KEYS.items():
        assert f"{record_type} (needs {', '.join(keys) or 'no detail'})" in text


@pytest.mark.parametrize("command", ["resolve", "search"])
@pytest.mark.parametrize("include", ["documents", "protocols"])
def test_unknown_include_is_a_validation_failure(command, include):
    host = _host()
    result = CliRunner().invoke(_app(), [command, "rollback", "--include", include])
    assert result.exit_code == 1
    assert "Unknown include families" in result.output
    assert "--include docs" in result.output
    assert host.agent_context.requests == []


@pytest.mark.parametrize("command", ["resolve", "search"])
def test_extension_include_uses_the_engine_catalog(command):
    host = _host()
    catalog_requests = []

    def catalog(request):
        catalog_requests.append(request)
        return SimpleNamespace(views=({"v1_include": "protocols", "backed": True},))

    host.graph.catalog = catalog
    result = CliRunner().invoke(_app(), [command, "status 2", "--include", "protocols"])
    assert result.exit_code == 0, result.output
    assert [request.pot_id for request in catalog_requests] == ["p"]
    assert host.agent_context.requests[0].include == ("protocols",)


@pytest.mark.parametrize("command", ["resolve", "search"])
def test_standard_include_does_not_fetch_catalog(command):
    host = _host()

    def unexpected_catalog(request):
        pytest.fail("ordinary include must not add a catalog roundtrip")

    host.graph.catalog = unexpected_catalog
    result = CliRunner().invoke(_app(), [command, "rollback", "--include", "docs"])
    assert result.exit_code == 0, result.output


def test_invalid_traversal_direction_fails_before_pot_resolution():
    from potpie.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "--json",
            "graph",
            "read",
            "--subgraph",
            "infra_topology",
            "--view",
            "service_neighborhood",
            "--direction",
            "sideways",
        ],
    )
    assert result.exit_code == 1
    assert "--direction must be one of: out, in, both" in result.output


# --- the record-type vocabulary is discoverable from the CLI --------------------


def test_record_help_advertises_exactly_the_public_record_types():
    result = CliRunner().invoke(_app(), ["record", "--help"])

    assert result.exit_code == 0
    text = " ".join(result.stdout.replace("│", " ").split())
    for record_type in PUBLIC_RECORD_TYPES:
        assert record_type in text, record_type
    assert "Structured" in text and "Free-form" in text


def test_record_unknown_type_is_refused_before_the_engine_is_asked():
    _common.set_json(True)
    host = _host()

    result = CliRunner().invoke(
        _app(), ["record", "--type", "decsion", "--summary", "use jittered backoff"]
    )

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert host.agent_context.requests == []
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert payload["detail"]["candidates"][0] == "decision"
    assert payload["detail"]["corrected_template"] == (
        "potpie record --type decision --summary '<summary>' "
        "--detail rationale=<rationale>"
    )
    assert "potpie record --type decision" in payload["recommended_next_action"]
    assert "decision" in payload["detail"]["structured_types"]
    assert "runbook_note" in payload["detail"]["free_form_types"]


def test_record_type_case_is_folded_quietly():
    host = _host()

    result = CliRunner().invoke(
        _app(), ["record", "--type", "Fix", "--summary", "retries need jitter"]
    )

    assert result.exit_code == 0, result.output
    assert host.agent_context.requests[0].record_type == "fix"
