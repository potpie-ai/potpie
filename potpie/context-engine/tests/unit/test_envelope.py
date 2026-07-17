"""Canonical agent envelope (rebuild plan P8).

Intent/include vocabulary is the single canonical set in
``potpie_context_core.domain.agent_context_port``; the envelope carries plain strings.
"""

from __future__ import annotations

from datetime import datetime, timezone

from potpie_context_engine.application.readers._common import ReadResponse
from potpie_context_engine.application.services.envelope_builder import (
    EnvelopeBuilder,
    IncludeResult,
    envelope_to_dict,
)
from potpie_context_core.domain.agent_context_port import CONTEXT_INTENTS, DEFAULT_INTENT_INCLUDES
from potpie_context_core.domain.agent_envelope import CoverageReport, derive_overall_confidence
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


class TestDeriveOverallConfidence:
    def test_all_complete_gives_high(self) -> None:
        coverage = [
            CoverageReport(include="owners", status="complete"),
            CoverageReport(include="decisions", status="complete"),
        ]
        assert derive_overall_confidence(coverage=coverage) == "high"

    def test_one_empty_caps_to_low(self) -> None:
        coverage = [
            CoverageReport(include="owners", status="complete"),
            CoverageReport(include="prior_bugs", status="empty"),
        ]
        assert derive_overall_confidence(coverage=coverage) == "low"

    def test_empty_input_returns_unknown(self) -> None:
        assert derive_overall_confidence(coverage=[]) == "unknown"


class TestEnvelopeBuilder:
    def test_cross_include_ranking(self) -> None:
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

    def test_default_includes_when_none_requested(self) -> None:
        builder = EnvelopeBuilder()
        envelope = builder.build(pot_id="pot-1", intent="feature", results=[])
        # No requested includes → no unsupported entries for the intent defaults.
        assert envelope.unsupported_includes == ()
        assert envelope.intent == "feature"

    def test_unknown_intent_normalizes(self) -> None:
        builder = EnvelopeBuilder()
        envelope = builder.build(pot_id="pot-1", intent="not-a-real-intent", results=[])
        assert envelope.intent == "unknown"

    def test_coverage_carries_graph_view_forward_pointer(self) -> None:
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
            envelope.to_dict()["coverage"][0]["graph_view"]
            == "debugging.prior_occurrences"
        )

    def test_unsupported_includes_propagate(self) -> None:
        builder = EnvelopeBuilder()
        envelope = builder.build(
            pot_id="pot-1",
            intent="feature",
            results=[],
            requested_includes=["decisions", "bogus"],
        )
        names = [u.name for u in envelope.unsupported_includes]
        assert names == ["bogus"]

    def test_serialisation_to_dict(self) -> None:
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

    def test_metadata_passthrough(self) -> None:
        builder = EnvelopeBuilder()
        envelope = builder.build(
            pot_id="pot-1",
            intent="feature",
            results=[],
            metadata={"trace_id": "t-1"},
        )
        assert envelope.metadata == {"trace_id": "t-1"}


class TestAgentContractGenerator:
    def test_generator_emits_canonical_catalog(self) -> None:
        import io

        from scripts.generate_agent_contract import emit

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

    def test_generator_output_is_deterministic(self) -> None:
        import io

        from scripts.generate_agent_contract import emit

        buf_a = io.StringIO()
        buf_b = io.StringIO()
        emit(buf_a)
        emit(buf_b)
        assert buf_a.getvalue() == buf_b.getvalue()
