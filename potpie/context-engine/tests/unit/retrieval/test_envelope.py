"""Canonical agent envelope (rebuild plan P8).

Intent/include vocabulary is the single canonical set in
``potpie_context_engine.core.agent_context_port``; the envelope carries plain strings.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from potpie_context_engine.application.readers._common import ReadResponse
from potpie_context_engine.application.services.envelope_builder import (
    EnvelopeBuilder,
    IncludeResult,
    envelope_to_dict,
)
from potpie_context_engine.core.agent_context_port import (
    CONTEXT_INTENTS,
    DEFAULT_INTENT_INCLUDES,
)
from potpie_context_engine.core.agent_envelope import (
    AgentEnvelope,
    CoverageReport,
    derive_overall_confidence,
)
from potpie_context_engine.domain.ranking import Candidate, RankedItem


_NOW = datetime(2026, 5, 20, tzinfo=timezone.utc)


def _ranked_item(*, key: str, score: float, payload: dict) -> RankedItem:
    return RankedItem(
        candidate=Candidate(candidate_key=key, payload=payload),
        score=score,
        breakdown={"strength": score, "recency": score},
    )


def _resp(
    *, family: str, items: list[RankedItem], coverage_status: str, pool: int = 0
) -> ReadResponse:
    return ReadResponse(
        family=family,
        items=tuple(items),
        coverage_status=coverage_status,
        meta={"candidate_pool": pool},
    )


def test_derive_overall_confidence_all_complete_gives_high() -> None:
    coverage = [
        CoverageReport(include="owners", status="complete"),
        CoverageReport(include="decisions", status="complete"),
    ]
    assert derive_overall_confidence(coverage=coverage) == "high"


def test_derive_overall_confidence_one_empty_caps_to_low() -> None:
    coverage = [
        CoverageReport(include="owners", status="complete"),
        CoverageReport(include="prior_bugs", status="empty"),
    ]
    assert derive_overall_confidence(coverage=coverage) == "low"


def test_derive_overall_confidence_empty_input_returns_unknown() -> None:
    assert derive_overall_confidence(coverage=[]) == "unknown"


def test_derive_overall_confidence_unmeasured_relevance_leaves_coverage_verdict_alone() -> (
    None
):
    """A family that cannot measure relevance is judged exactly as before."""
    coverage = [CoverageReport(include="owners", status="complete")]
    assert derive_overall_confidence(coverage=coverage) == "high"


def test_derive_overall_confidence_weak_evidence_downgrades_a_complete_page() -> None:
    """The regression this exists for: a full page of irrelevant hits.

    ``complete`` only says the family filled its page. Before relevance was
    carried, an unanswerable query against a working index reported "high".
    """
    coverage = [
        CoverageReport(include="resources", status="complete", best_relevance=0.21)
    ]
    assert derive_overall_confidence(coverage=coverage) == "low"


def test_derive_overall_confidence_middling_evidence_reports_medium() -> None:
    coverage = [
        CoverageReport(include="resources", status="complete", best_relevance=0.40)
    ]
    assert derive_overall_confidence(coverage=coverage) == "medium"


def test_derive_overall_confidence_strong_evidence_keeps_high() -> None:
    coverage = [
        CoverageReport(include="resources", status="complete", best_relevance=0.62)
    ]
    assert derive_overall_confidence(coverage=coverage) == "high"


def test_derive_overall_confidence_relevance_never_upgrades_a_weak_coverage_verdict() -> (
    None
):
    """The two signals cap each other; neither promotes."""
    coverage = [
        CoverageReport(include="resources", status="sparse", best_relevance=0.95)
    ]
    assert derive_overall_confidence(coverage=coverage) == "low"


def test_derive_overall_confidence_best_relevance_wins_across_families() -> None:
    """One family finding a strong answer is not dragged down by another's
    weak one — the relevance channel takes the best measured, while status
    still takes the worst."""
    coverage = [
        CoverageReport(include="resources", status="complete", best_relevance=0.62),
        CoverageReport(include="docs", status="complete", best_relevance=0.10),
    ]
    assert derive_overall_confidence(coverage=coverage) == "high"


def test_derive_overall_confidence_best_relevance_is_serialised() -> None:
    envelope = AgentEnvelope(
        pot_id="pot_1",
        intent="unknown",
        items=(),
        coverage=(
            CoverageReport(include="resources", status="complete", best_relevance=0.42),
        ),
    )
    assert envelope.to_dict()["coverage"][0]["best_relevance"] == 0.42


def test_envelope_builder_cross_include_ranking() -> None:
    builder = EnvelopeBuilder()
    prefs_resp = _resp(
        family="preferences",
        items=[_ranked_item(key="pref-a", score=0.4, payload={"src": "pref"})],
        coverage_status="partial",
    )
    bugs_resp = _resp(
        family="prior_fixes",
        items=[_ranked_item(key="bug-a", score=0.9, payload={"src": "bug"})],
        coverage_status="complete",
    )
    envelope = builder.build(
        pot_id="pot-1",
        intent="debugging",
        results=[
            IncludeResult(include="preferences", response=prefs_resp),
            IncludeResult(include="prior_fixes", response=bugs_resp),
        ],
        requested_includes=["preferences", "prior_fixes"],
    )
    # Cross-leg sort: bug (0.9) ahead of preference (0.4)
    assert envelope.items[0].candidate_key == "bug-a"
    assert envelope.items[1].candidate_key == "pref-a"


def test_envelope_builder_docs_include_is_demoted_against_equal_prior_bugs() -> None:
    # P9: equal raw scores must still surface project memory ahead of
    # section claims when both includes are requested together.
    builder = EnvelopeBuilder()
    envelope = builder.build(
        pot_id="pot-1",
        intent="unknown",
        results=[
            IncludeResult(
                include="docs",
                response=_resp(
                    family="docs",
                    items=[
                        _ranked_item(
                            key="section-a",
                            score=0.9,
                            payload={"src": "section"},
                        )
                    ],
                    coverage_status="complete",
                ),
            ),
            IncludeResult(
                include="prior_bugs",
                response=_resp(
                    family="prior_bugs",
                    items=[
                        _ranked_item(key="bug-a", score=0.9, payload={"src": "bug"})
                    ],
                    coverage_status="complete",
                ),
            ),
        ],
        requested_includes=["docs", "prior_bugs"],
    )
    assert envelope.items[0].candidate_key == "bug-a"
    assert envelope.items[1].candidate_key == "section-a"
    assert envelope.items[0].score > envelope.items[1].score


def test_envelope_builder_default_includes_when_none_requested() -> None:
    builder = EnvelopeBuilder()
    envelope = builder.build(pot_id="pot-1", intent="feature", results=[])
    # No requested includes → no unsupported entries for the intent defaults.
    assert envelope.unsupported_includes == ()
    assert envelope.intent == "feature"


def test_envelope_builder_unknown_intent_normalizes() -> None:
    builder = EnvelopeBuilder()
    envelope = builder.build(pot_id="pot-1", intent="not-a-real-intent", results=[])
    assert envelope.intent == "unknown"


def test_envelope_builder_coverage_carries_graph_view_forward_pointer() -> None:
    builder = EnvelopeBuilder()
    envelope = builder.build(
        pot_id="pot-1",
        intent="debugging",
        results=[
            IncludeResult(
                include="prior_bugs",
                response=_resp(
                    family="prior_bugs",
                    items=[],
                    coverage_status="empty",
                ),
            ),
        ],
        requested_includes=["prior_bugs"],
    )
    assert envelope.coverage[0].graph_view == "debugging.prior_occurrences"
    assert (
        envelope.to_dict()["coverage"][0]["graph_view"] == "debugging.prior_occurrences"
    )


def test_envelope_builder_unsupported_includes_propagate() -> None:
    builder = EnvelopeBuilder()
    envelope = builder.build(
        pot_id="pot-1",
        intent="feature",
        results=[],
        requested_includes=["decisions", "bogus"],
    )
    names = [u.name for u in envelope.unsupported_includes]
    assert names == ["bogus"]


def test_envelope_builder_serialisation_to_dict() -> None:
    builder = EnvelopeBuilder()
    prefs_resp = _resp(
        family="preferences",
        items=[_ranked_item(key="p", score=0.5, payload={"a": 1})],
        coverage_status="complete",
        pool=3,
    )
    envelope = builder.build(
        pot_id="pot-1",
        intent="feature",
        results=[IncludeResult(include="preferences", response=prefs_resp)],
        requested_includes=["preferences"],
        as_of=_NOW,
    )
    out = envelope_to_dict(envelope)
    assert envelope.to_dict() == out
    assert out["pot_id"] == "pot-1"
    assert out["intent"] == "feature"
    assert out["items"][0]["include"] == "preferences"
    assert out["items"][0]["candidate_key"] == "p"
    assert out["coverage"][0]["status"] == "complete"
    assert out["coverage"][0]["candidate_pool"] == 3
    assert out["overall_confidence"] == "high"
    assert out["as_of"] == _NOW.isoformat()


def test_envelope_builder_metadata_passthrough() -> None:
    builder = EnvelopeBuilder()
    envelope = builder.build(
        pot_id="pot-1",
        intent="feature",
        results=[],
        metadata={"trace_id": "t-1"},
    )
    assert envelope.metadata == {"trace_id": "t-1"}


def _agent_contract_emit():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts/generate_agent_contract.py"
    spec = importlib.util.spec_from_file_location("engine_test_agent_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.emit


@pytest.mark.source_contract
def test_agent_contract_generator_generator_emits_canonical_catalog() -> None:
    import io

    emit = _agent_contract_emit()

    buf = io.StringIO()
    emit(buf)
    out = buf.getvalue()
    assert "## Intents" in out
    assert "## Includes" in out
    # Every canonical intent appears in the generated contract.
    for intent in CONTEXT_INTENTS:
        assert f"`{intent}`" in out
    # A representative default include appears.
    assert "decisions" in out
    assert DEFAULT_INTENT_INCLUDES  # sanity: the source table is non-empty


@pytest.mark.source_contract
def test_agent_contract_generator_generator_output_is_deterministic() -> None:
    import io

    emit = _agent_contract_emit()

    buf_a = io.StringIO()
    buf_b = io.StringIO()
    emit(buf_a)
    emit(buf_b)
    assert buf_a.getvalue() == buf_b.getvalue()


def test_exact_passage_survives_cross_family_demotion_without_promoting_weak_hits():
    builder = EnvelopeBuilder()
    resources = _resp(
        family="resources",
        coverage_status="partial",
        items=[
            _ranked_item(
                key="exact-token",
                score=0.55,
                payload={"retrieval": {"lexical_rank": 1, "term_coverage": 1.0}},
            ),
            _ranked_item(
                key="one-word-overlap",
                score=0.7,
                payload={"retrieval": {"lexical_rank": 2, "term_coverage": 0.2}},
            ),
            _ranked_item(
                key="no-lexical-match",
                score=0.65,
                payload={"retrieval": {"lexical_rank": None, "term_coverage": 0.0}},
            ),
        ],
    )
    timeline = _resp(
        family="timeline",
        coverage_status="complete",
        items=[_ranked_item(key="recent-but-unrelated", score=0.8, payload={})],
    )
    env = builder.build(
        pot_id="p",
        intent="unknown",
        requested_includes=["resources", "timeline"],
        results=[
            IncludeResult("resources", resources),
            IncludeResult("timeline", timeline),
        ],
    )
    assert [item.candidate_key for item in env.items] == [
        "exact-token",
        "recent-but-unrelated",
        "one-word-overlap",
        "no-lexical-match",
    ]
    assert env.items[0].breakdown["lexical_coverage_floor"] == 1.0
