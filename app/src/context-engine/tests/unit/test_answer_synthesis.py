"""LLM answer synthesis for ``goal=answer`` replaces the count-string summary.

Covers the synthesis step added to close the canned-summary gap
(docs/context-graph/implementation-next-steps.md #4, reviewed 2026-04-22).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from adapters.outbound.synthesis.null import NullAnswerSynthesizer
from adapters.outbound.synthesis.prompt import (
    build_synthesis_payload,
    build_synthesis_prompt,
)
from domain.agent_context_port import bundle_to_agent_envelope
from domain.graph_quality import GraphQualityReport
from domain.intelligence_models import (
    ChangeRecord,
    ContextResolutionRequest,
    ContextScope,
    CoverageReport,
    DecisionRecord,
    IntelligenceBundle,
    ProjectContextRecord,
    ResolutionMeta,
)
from domain.source_references import (
    FreshnessReport,
    SourceFallback,
    SourceReferenceRecord,
)
from domain.source_resolution import SourceResolutionResult

pytestmark = pytest.mark.unit


def _bundle(
    *,
    decisions: list[DecisionRecord] | None = None,
    changes: list[ChangeRecord] | None = None,
    project_map: list[ProjectContextRecord] | None = None,
    fallbacks: list[SourceFallback] | None = None,
    source_refs: list[SourceReferenceRecord] | None = None,
    coverage_status: str = "complete",
    missing: list[str] | None = None,
    query: str = "why did we choose Neo4j?",
) -> IntelligenceBundle:
    scope = ContextScope(repo_name="potpie", file_path="app/x.py")
    request = ContextResolutionRequest(pot_id="pot-1", query=query, scope=scope)
    return IntelligenceBundle(
        request=request,
        semantic_hits=[],
        artifacts=[],
        changes=changes or [],
        decisions=decisions or [],
        discussions=[],
        ownership=[],
        project_map=project_map or [],
        debugging_memory=[],
        causal_chain=[],
        source_refs=source_refs or [],
        coverage=CoverageReport(status=coverage_status, missing=missing or []),
        freshness=FreshnessReport(),
        quality=GraphQualityReport(),
        fallbacks=fallbacks or [],
        source_resolution=SourceResolutionResult(),
        open_conflicts=[],
        recommended_next_actions=[],
        errors=[],
        meta=ResolutionMeta(provider="test"),
    )


# --- Envelope override -----------------------------------------------------------


def test_envelope_falls_back_to_count_string_without_override() -> None:
    bundle = _bundle(
        decisions=[DecisionRecord(decision="d", rationale="r", pr_number=1)]
    )
    env = bundle_to_agent_envelope(bundle)
    assert env["answer"]["summary"].startswith("Resolved ")


def test_envelope_uses_synthesized_override_when_provided() -> None:
    bundle = _bundle()
    env = bundle_to_agent_envelope(bundle, answer_summary="We picked Neo4j because…")
    assert env["answer"]["summary"] == "We picked Neo4j because…"


def test_envelope_falls_back_when_override_is_empty_string() -> None:
    bundle = _bundle()
    env = bundle_to_agent_envelope(bundle, answer_summary="")
    assert env["answer"]["summary"].startswith("No matching") or env["answer"][
        "summary"
    ].startswith("Resolved ")


# --- Null synthesizer ------------------------------------------------------------


async def test_null_synthesizer_returns_none() -> None:
    synth = NullAnswerSynthesizer()
    assert await synth.synthesize(_bundle()) is None


# --- Prompt builder --------------------------------------------------------------


def test_prompt_payload_contains_key_evidence_sections() -> None:
    bundle = _bundle(
        decisions=[
            DecisionRecord(
                decision="Use Neo4j", rationale="temporal edges", pr_number=7
            )
        ],
        changes=[
            ChangeRecord(pr_number=42, title="Switch worker", summary="to Hatchet")
        ],
        project_map=[
            ProjectContextRecord(
                family="service_map",
                kind="Service",
                entity_key="service:x",
                name="context-engine",
                summary="core service",
            )
        ],
        fallbacks=[
            SourceFallback(
                code="semantic_fallback",
                message="decisions via semantic seeds",
            )
        ],
        source_refs=[
            SourceReferenceRecord(
                ref="7",
                source_type="pr",
                verification_state="needs_verification",
            )
        ],
        coverage_status="partial",
        missing=["change_history"],
    )
    payload = build_synthesis_payload(bundle)

    assert payload["query"] == "why did we choose Neo4j?"
    assert payload["coverage"] == {"status": "partial", "missing": ["change_history"]}
    assert len(payload["evidence"]["decisions"]) == 1
    assert payload["evidence"]["decisions"][0]["decision"] == "Use Neo4j"
    assert payload["evidence"]["recent_changes"][0]["pr_number"] == 42
    assert payload["evidence"]["project_map"][0]["name"] == "context-engine"
    assert payload["fallbacks"] == [
        {"code": "semantic_fallback", "message": "decisions via semantic seeds"}
    ]
    assert payload["source_refs"][0]["verification_state"] == "needs_verification"


def test_prompt_truncates_long_lists() -> None:
    many_decisions = [
        DecisionRecord(decision=f"d{i}", rationale=f"r{i}", pr_number=i)
        for i in range(20)
    ]
    payload = build_synthesis_payload(_bundle(decisions=many_decisions))
    assert len(payload["evidence"]["decisions"]) == 8


def test_prompt_is_json_serializable() -> None:
    bundle = _bundle(
        decisions=[DecisionRecord(decision="d", rationale="r", pr_number=1)],
    )
    text = build_synthesis_prompt(bundle)
    assert '"query"' in text
    assert '"evidence"' in text


def test_prompt_handles_request_with_no_scope() -> None:
    request = ContextResolutionRequest(pot_id="pot-1", query="q", scope=None)
    bundle = IntelligenceBundle(
        request=request,
        semantic_hits=[],
        artifacts=[],
        changes=[],
        decisions=[],
        discussions=[],
        ownership=[],
        project_map=[],
        debugging_memory=[],
        causal_chain=[],
        source_refs=[],
        coverage=CoverageReport(status="complete"),
        freshness=FreshnessReport(),
        quality=GraphQualityReport(),
        fallbacks=[],
        source_resolution=SourceResolutionResult(),
        open_conflicts=[],
        recommended_next_actions=[],
        errors=[],
        meta=ResolutionMeta(provider="test"),
    )
    payload = build_synthesis_payload(bundle)
    assert payload["scope"] == {}


# --- Adapter wiring --------------------------------------------------------------


@dataclass
class _RecordingSynth:
    calls: int = 0
    return_value: str | None = "Chose Neo4j for temporal edges (pr:7)."

    async def synthesize(self, bundle: IntelligenceBundle) -> str | None:
        self.calls += 1
        _ = bundle
        return self.return_value


async def test_answer_async_uses_synthesizer_summary() -> None:
    from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
    from domain.graph_query import (
        ContextGraphGoal,
        ContextGraphQuery,
        ContextGraphStrategy,
    )

    class _Episodic:
        enabled = True

    class _Structural:
        pass

    captured_bundle = _bundle(
        decisions=[DecisionRecord(decision="Use Neo4j", rationale="r", pr_number=7)]
    )

    class _ResolutionService:
        async def resolve(self, _request):
            return captured_bundle

    synth = _RecordingSynth()
    adapter = GraphitiContextGraphAdapter(
        episodic=_Episodic(),  # type: ignore[arg-type]
        structural=_Structural(),  # type: ignore[arg-type]
        resolution_service=_ResolutionService(),
        answer_synthesizer=synth,
    )
    request = ContextGraphQuery(
        pot_id="pot-1",
        goal=ContextGraphGoal.ANSWER,
        strategy=ContextGraphStrategy.AUTO,
        query="why Neo4j?",
    )
    result = await adapter.query_async(request)

    assert synth.calls == 1
    assert result.result["answer"]["summary"] == "Chose Neo4j for temporal edges (pr:7)."
    assert result.meta["answer_summary_source"] == "synthesized"


async def test_answer_async_falls_back_when_synthesizer_returns_none() -> None:
    from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
    from domain.graph_query import (
        ContextGraphGoal,
        ContextGraphQuery,
        ContextGraphStrategy,
    )

    class _Episodic:
        enabled = True

    class _Structural:
        pass

    class _ResolutionService:
        async def resolve(self, _request):
            return _bundle(
                decisions=[DecisionRecord(decision="d", rationale="r", pr_number=1)]
            )

    synth = _RecordingSynth(return_value=None)
    adapter = GraphitiContextGraphAdapter(
        episodic=_Episodic(),  # type: ignore[arg-type]
        structural=_Structural(),  # type: ignore[arg-type]
        resolution_service=_ResolutionService(),
        answer_synthesizer=synth,
    )
    request = ContextGraphQuery(
        pot_id="pot-1",
        goal=ContextGraphGoal.ANSWER,
        strategy=ContextGraphStrategy.AUTO,
        query="why?",
    )
    result = await adapter.query_async(request)

    assert synth.calls == 1
    assert result.result["answer"]["summary"].startswith("Resolved ")
    assert result.meta["answer_summary_source"] == "counts"
