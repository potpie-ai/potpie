"""LLM answer synthesis for ``goal=answer`` replaces the fallback summary.

The synthesizer + prompt now consume the canonical :class:`AgentEnvelope`.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from adapters.outbound.graph.context_graph_service import ContextGraphService
from adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore
from adapters.outbound.synthesis.null import NullAnswerSynthesizer
from adapters.outbound.synthesis.prompt import (
    build_synthesis_payload,
    build_synthesis_prompt,
)
from application.services.read_orchestrator import ReadOrchestrator
from domain.agent_envelope import AgentEnvelope, CoverageReport, EvidenceItem, UnsupportedInclude
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphStrategy,
)

pytestmark = pytest.mark.unit


def _item(include: str, key: str, **payload) -> EvidenceItem:
    return EvidenceItem(
        include=include,
        candidate_key=key,
        score=0.5,
        payload=payload,
        coverage_status="complete",
    )


def _env(
    *,
    items: list[EvidenceItem] | None = None,
    coverage: list[CoverageReport] | None = None,
    unsupported: list[UnsupportedInclude] | None = None,
    query: str = "why did we choose Neo4j?",
    confidence: str = "high",
) -> AgentEnvelope:
    return AgentEnvelope(
        pot_id="pot-1",
        intent="feature",
        items=tuple(items or []),
        coverage=tuple(coverage or []),
        unsupported_includes=tuple(unsupported or []),
        overall_confidence=confidence,
        metadata={"query": query},
    )


# --- Prompt builder --------------------------------------------------------------


def test_prompt_payload_contains_key_sections() -> None:
    env = _env(
        items=[
            _item("infra_topology", "k1", fact="web depends on auth", subject_key="service:web"),
            _item("prior_bugs", "k2", fact="pool exhausted", symptom="connection pool"),
        ],
        coverage=[
            CoverageReport(include="infra_topology", status="partial"),
            CoverageReport(include="prior_bugs", status="complete"),
        ],
        unsupported=[UnsupportedInclude(name="owners", reason="not_implemented")],
        confidence="medium",
    )
    payload = build_synthesis_payload(env)

    assert payload["query"] == "why did we choose Neo4j?"
    assert payload["confidence"] == "medium"
    assert {"include": "infra_topology", "status": "partial"} in payload["coverage"]
    assert payload["unsupported_includes"] == ["owners"]
    assert len(payload["evidence"]["infra_topology"]) == 1
    assert payload["evidence"]["infra_topology"][0]["fact"] == "web depends on auth"
    assert payload["evidence"]["prior_bugs"][0]["symptom"] == "connection pool"


def test_prompt_truncates_long_lists() -> None:
    items = [_item("infra_topology", f"k{i}", fact=f"f{i}") for i in range(20)]
    payload = build_synthesis_payload(_env(items=items))
    assert len(payload["evidence"]["infra_topology"]) == 8


def test_prompt_is_json_serializable() -> None:
    text = build_synthesis_prompt(
        _env(items=[_item("prior_bugs", "k", fact="x")])
    )
    assert '"query"' in text
    assert '"evidence"' in text


# --- Null synthesizer ------------------------------------------------------------


async def test_null_synthesizer_returns_none() -> None:
    synth = NullAnswerSynthesizer()
    assert await synth.synthesize(_env()) is None


# --- Adapter wiring --------------------------------------------------------------


@dataclass
class _RecordingSynth:
    calls: int = 0
    return_value: str | None = "Chose Neo4j for temporal edges."

    async def synthesize(self, envelope: AgentEnvelope) -> str | None:
        self.calls += 1
        _ = envelope
        return self.return_value


def _adapter(synth) -> ContextGraphService:
    episodic = MagicMock()
    episodic.enabled = True
    return ContextGraphService(
        graph_writer=episodic,
        orchestrator=ReadOrchestrator(claim_query=InMemoryClaimQueryStore()),
        answer_synthesizer=synth,
    )


def _answer_request() -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id="pot-1",
        goal=ContextGraphGoal.ANSWER,
        strategy=ContextGraphStrategy.AUTO,
        query="why Neo4j?",
    )


async def test_answer_async_uses_synthesizer_summary() -> None:
    synth = _RecordingSynth()
    result = await _adapter(synth).query_async(_answer_request())
    assert synth.calls == 1
    assert result.result["answer"]["summary"] == "Chose Neo4j for temporal edges."
    assert result.meta["answer_summary_source"] == "synthesized"


async def test_answer_async_falls_back_when_synthesizer_returns_none() -> None:
    synth = _RecordingSynth(return_value=None)
    result = await _adapter(synth).query_async(_answer_request())
    assert synth.calls == 1
    # Empty claim store → deterministic fallback summary.
    assert result.result["answer"]["summary"]
    assert result.meta["answer_summary_source"] == "fallback"
