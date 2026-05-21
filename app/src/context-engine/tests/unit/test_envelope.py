"""Canonical agent envelope (rebuild plan P8)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from application.readers._common import ReadResponse
from application.services.envelope_builder import (
    EnvelopeBuilder,
    IncludeResult,
    envelope_to_dict,
)
from domain.agent_envelope import (
    AgentInclude,
    AgentIntent,
    INTENT_INCLUDES,
    derive_overall_confidence,
    resolve_includes,
)
from domain.ranking import Candidate, RankedItem


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


class TestResolveIncludes:
    def test_default_includes_for_intent(self) -> None:
        matched, unsupported = resolve_includes(intent=AgentIntent.FEATURE, requested=None)
        assert unsupported == []
        assert AgentInclude.CODING_PREFERENCES in matched
        assert matched == list(INTENT_INCLUDES[AgentIntent.FEATURE])

    def test_unknown_include_produces_unsupported(self) -> None:
        matched, unsupported = resolve_includes(
            intent=AgentIntent.FEATURE, requested=["coding_preferences", "bogus_include"]
        )
        assert AgentInclude.CODING_PREFERENCES in matched
        assert len(unsupported) == 1
        assert unsupported[0].name == "bogus_include"
        assert unsupported[0].reason == "unknown_include"

    def test_string_request_promoted(self) -> None:
        matched, _ = resolve_includes(
            intent=AgentIntent.DEBUGGING, requested=["prior_bugs"]
        )
        assert matched == [AgentInclude.PRIOR_BUGS]


class TestDeriveOverallConfidence:
    def test_all_complete_gives_high(self) -> None:
        from domain.agent_envelope import CoverageReport

        coverage = [
            CoverageReport(include=AgentInclude.OWNERS, status="complete"),
            CoverageReport(include=AgentInclude.DECISIONS, status="complete"),
        ]
        assert derive_overall_confidence(coverage=coverage) == "high"

    def test_one_empty_caps_to_low(self) -> None:
        from domain.agent_envelope import CoverageReport

        coverage = [
            CoverageReport(include=AgentInclude.OWNERS, status="complete"),
            CoverageReport(include=AgentInclude.PRIOR_BUGS, status="empty"),
        ]
        assert derive_overall_confidence(coverage=coverage) == "low"

    def test_empty_input_returns_unknown(self) -> None:
        assert derive_overall_confidence(coverage=[]) == "unknown"


class TestEnvelopeBuilder:
    def test_cross_include_ranking(self) -> None:
        builder = EnvelopeBuilder()
        prefs_resp = _resp(
            family="coding_preferences",
            items=[_ranked_item(key="pref-a", score=0.4, payload={"src": "pref"})],
            coverage_status="partial",
        )
        bugs_resp = _resp(
            family="prior_bugs",
            items=[_ranked_item(key="bug-a", score=0.9, payload={"src": "bug"})],
            coverage_status="complete",
        )
        envelope = builder.build(
            pot_id="pot-1",
            intent=AgentIntent.DEBUGGING,
            results=[
                IncludeResult(include=AgentInclude.CODING_PREFERENCES, response=prefs_resp),
                IncludeResult(include=AgentInclude.PRIOR_BUGS, response=bugs_resp),
            ],
            requested_includes=["coding_preferences", "prior_bugs"],
        )
        # Cross-leg sort: bug (0.9) ahead of preference (0.4)
        assert envelope.items[0].candidate_key == "bug-a"
        assert envelope.items[1].candidate_key == "pref-a"

    def test_unsupported_includes_propagate(self) -> None:
        builder = EnvelopeBuilder()
        envelope = builder.build(
            pot_id="pot-1",
            intent=AgentIntent.FEATURE,
            results=[],
            requested_includes=["coding_preferences", "bogus"],
        )
        names = [u.name for u in envelope.unsupported_includes]
        assert names == ["bogus"]

    def test_serialisation_to_dict(self) -> None:
        builder = EnvelopeBuilder()
        prefs_resp = _resp(
            family="coding_preferences",
            items=[_ranked_item(key="p", score=0.5, payload={"a": 1})],
            coverage_status="complete",
            pool=3,
        )
        envelope = builder.build(
            pot_id="pot-1",
            intent=AgentIntent.FEATURE,
            results=[
                IncludeResult(
                    include=AgentInclude.CODING_PREFERENCES,
                    response=prefs_resp,
                )
            ],
            as_of=_NOW,
        )
        out = envelope_to_dict(envelope)
        assert out["pot_id"] == "pot-1"
        assert out["intent"] == "feature"
        assert out["items"][0]["include"] == "coding_preferences"
        assert out["coverage"][0]["status"] == "complete"
        assert out["overall_confidence"] == "high"
        assert out["as_of"] == _NOW.isoformat()

    def test_metadata_passthrough(self) -> None:
        builder = EnvelopeBuilder()
        envelope = builder.build(
            pot_id="pot-1",
            intent=AgentIntent.FEATURE,
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
        # Every intent + include appears
        for intent in AgentIntent:
            assert intent.value in out
        for include in AgentInclude:
            assert include.value in out

    def test_generator_output_is_deterministic(self) -> None:
        import io

        from scripts.generate_agent_contract import emit

        buf_a = io.StringIO()
        buf_b = io.StringIO()
        emit(buf_a)
        emit(buf_b)
        assert buf_a.getvalue() == buf_b.getvalue()
