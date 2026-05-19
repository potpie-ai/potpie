"""Read-side agentic query loop (``goal=investigate``).

Covers the agent that drives retrieval over the shared read tools, the
answer-envelope-shaped result it produces, and graceful degradation onto the
deterministic resolve path when the agent is unconfigured, empty, or failing.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from adapters.outbound.graphiti.context_graph import (
    GraphitiContextGraphAdapter,
    _investigate_envelope,
)
from adapters.outbound.query_agent.null_agent import NullQueryAgent
from adapters.outbound.query_agent.pydantic_query_agent import (
    PydanticQueryAgent,
    _compact_for_llm,
    _describe,
    _RunState,
)
from application.services.context_reader_registry import ContextReaderRegistry
from bootstrap.container import _build_query_agent
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphResult,
    ContextGraphScope,
    ContextGraphStrategy,
)
from domain.graph_quality import GraphQualityReport
from domain.intelligence_models import (
    ContextResolutionRequest,
    CoverageReport,
    DecisionRecord,
    IntelligenceBundle,
    ResolutionMeta,
)
from domain.ports.query_agent import QueryAgentResult, QueryAgentStep
from domain.ports.reconciliation_tools import ToolDescriptor
from domain.source_references import FreshnessReport
from domain.source_resolution import SourceResolutionResult

pytestmark = pytest.mark.unit


class _Episodic:
    enabled = True


class _Structural:
    pass


def _adapter(*, readers=None, query_agent=None, resolution_service=None):
    return GraphitiContextGraphAdapter(
        episodic=_Episodic(),  # type: ignore[arg-type]
        structural=_Structural(),  # type: ignore[arg-type]
        readers=readers or ContextReaderRegistry(),
        resolution_service=resolution_service,
        query_agent=query_agent,
    )


def _investigate_request(**kw) -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id="pot-1",
        query=kw.pop("query", "who owns the auth module?"),
        goal=ContextGraphGoal.INVESTIGATE,
        strategy=ContextGraphStrategy.AUTO,
        scope=ContextGraphScope(repo_name="potpie-ai/potpie"),
        **kw,
    )


# --- Pure helpers ----------------------------------------------------------------


def test_run_state_accumulates_evidence_steps_and_dedupes_refs() -> None:
    state = _RunState()
    state.record(
        "context_search",
        {"query": "auth"},
        {
            "kind": "semantic_search",
            "result": [
                {"uuid": "u1", "source_refs": ["gh:pr:1"]},
                {"uuid": "u2", "source_refs": ["gh:pr:1", "gh:pr:2"]},
            ],
        },
    )
    state.record(
        "context_graph_overview",
        {},
        {"kind": "graph_overview", "result": {"entities": 12}},
    )
    assert [s.tool for s in state.steps] == [
        "context_search",
        "context_graph_overview",
    ]
    assert state.steps[0].result_count == 2
    assert state.steps[1].result_count == 1
    assert len(state.evidence) == 3  # 2 hits + 1 overview dict
    assert state.source_refs == ["gh:pr:1", "gh:pr:2"]  # deduped, ordered


def test_compact_for_llm_bounds_payload() -> None:
    big = {"kind": "semantic_search", "result": [{"x": "y" * 999} for _ in range(50)]}
    out = _compact_for_llm(big)
    assert out["count"] == 50
    assert len(out["items"]) == 6
    assert len(out["items"][0]["x"]) <= 400
    assert _compact_for_llm({"error": "boom", "kind": "error"}) == {"error": "boom"}


def test_describe_appends_schema() -> None:
    d = ToolDescriptor(
        name="context_search",
        category="context_lookup",
        description="Semantic search.",
        json_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    text = _describe(d)
    assert text.startswith("Semantic search.")
    assert "Arguments JSON schema" in text


def test_investigate_envelope_is_answer_shaped_superset() -> None:
    env = _investigate_envelope(
        QueryAgentResult(
            answer="The auth module is owned by team-platform.",
            steps=[
                QueryAgentStep("context_search", {"query": "auth"}, "semantic_search", 3)
            ],
            evidence=[{"uuid": "u1"}],
            source_refs=["gh:pr:1"],
            confidence=0.84,
            iterations=1,
            usage={"model": "m"},
        )
    )
    assert env["ok"] is True
    assert env["answer"]["summary"].startswith("The auth module")
    assert env["confidence"] == 0.84
    assert env["coverage"]["status"] == "complete"
    assert env["agent"]["iterations"] == 1
    assert env["agent"]["steps"][0]["tool"] == "context_search"


def test_investigate_envelope_empty_marks_watch() -> None:
    env = _investigate_envelope(QueryAgentResult(answer="No project context found."))
    assert env["coverage"]["status"] == "empty"
    assert env["quality"]["status"] == "watch"
    assert env["confidence"] == 0.2


# --- Null agent ------------------------------------------------------------------


async def test_null_query_agent_returns_none() -> None:
    agent = NullQueryAgent()
    out = await agent.investigate(
        _investigate_request(), tools=[], run_tool=lambda *_a, **_k: None  # type: ignore[arg-type]
    )
    assert out is None


# --- Adapter dispatch + tool loop ------------------------------------------------


class _FakeAgent:
    """Drives ``run_tool`` like a real agent would, then returns a result."""

    def __init__(self) -> None:
        self.tool_names: list[str] = []
        self.results: list[dict] = []

    async def investigate(self, request, *, tools, run_tool):
        self.tool_names = [t.name for t in tools]
        self.results = [
            await run_tool("context_search", {"query": "auth", "limit": 3}),
            await run_tool("context_search", {}),  # missing query -> error
            await run_tool("context_graph_overview", {}),
            await run_tool("bogus_tool", {}),  # unknown -> error
        ]
        return QueryAgentResult(
            answer="team-platform owns auth (gh:pr:1).",
            steps=[
                QueryAgentStep("context_search", {"query": "auth"}, "semantic_search", 1)
            ],
            evidence=[{"uuid": "u1"}],
            source_refs=["gh:pr:1"],
            confidence=0.9,
            iterations=1,
            usage={"model": "m", "latency_ms": 5},
        )


async def test_investigate_runs_tool_loop_and_wraps_result() -> None:
    readers = MagicMock()
    readers.execute.return_value = ContextGraphResult(
        kind="semantic_search",
        goal="retrieve",
        strategy="semantic",
        result=[{"uuid": "u1", "source_refs": ["gh:pr:1"]}],
    )
    agent = _FakeAgent()
    adapter = _adapter(readers=readers, query_agent=agent)

    out = await adapter.query_async(_investigate_request())

    # Reuses the shared 4-tool catalog.
    assert agent.tool_names == [
        "context_search",
        "context_recent_changes",
        "context_file_owners",
        "context_graph_overview",
    ]
    # Only valid tool calls reach the readers (bad-arg + unknown short-circuit).
    assert readers.execute.call_count == 2
    valid_search_preset = readers.execute.call_args_list[0].args[0]
    assert valid_search_preset.goal == ContextGraphGoal.RETRIEVE
    assert valid_search_preset.include == ["semantic_search"]
    assert valid_search_preset.scope.repo_name == "potpie-ai/potpie"
    assert agent.results[1] == {"error": "query_required", "kind": "error"}
    assert agent.results[3] == {"error": "unknown_tool:bogus_tool", "kind": "error"}

    assert out.kind == "query_agent"
    assert out.goal == "investigate"
    assert out.result["answer"]["summary"].startswith("team-platform owns auth")
    assert out.result["confidence"] == 0.9
    assert out.result["agent"]["iterations"] == 1
    assert out.meta["path"] == "investigate"
    assert out.meta["cost"]["query_agent"]["model"] == "m"


async def test_investigate_falls_back_when_agent_not_configured() -> None:
    adapter = _adapter(query_agent=None)  # resolution_service also None
    out = await adapter.query_async(_investigate_request())
    assert out.meta["path"] == "investigate_fallback"
    assert out.meta["fallback"] == "query_agent_not_configured"
    assert out.error == "resolution_service_unavailable"


async def test_investigate_falls_back_when_agent_returns_none() -> None:
    class _NoneAgent:
        async def investigate(self, *a, **k):
            return None

    adapter = _adapter(query_agent=_NoneAgent())
    out = await adapter.query_async(_investigate_request())
    assert out.meta["fallback"] == "query_agent_unavailable"


async def test_investigate_falls_back_when_agent_raises() -> None:
    class _BoomAgent:
        async def investigate(self, *a, **k):
            raise RuntimeError("model exploded")

    adapter = _adapter(query_agent=_BoomAgent())
    out = await adapter.query_async(_investigate_request())
    assert out.meta["fallback"] == "query_agent_error"


async def test_investigate_fallback_yields_resolve_answer_envelope() -> None:
    bundle = IntelligenceBundle(
        request=ContextResolutionRequest(pot_id="pot-1", query="q", scope=None),
        semantic_hits=[],
        artifacts=[],
        changes=[],
        decisions=[DecisionRecord(decision="Use Neo4j", rationale="r", pr_number=7)],
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

    class _ResolutionService:
        async def resolve(self, _request):
            return bundle

    class _NoneAgent:
        async def investigate(self, *a, **k):
            return None

    adapter = _adapter(
        query_agent=_NoneAgent(), resolution_service=_ResolutionService()
    )
    out = await adapter.query_async(_investigate_request())

    assert out.kind == "resolve_context"
    assert out.meta["path"] == "investigate_fallback"
    assert out.meta["fallback"] == "query_agent_unavailable"
    assert "decision" in out.result["answer"]["summary"].lower()


# --- Container gating ------------------------------------------------------------


def test_build_query_agent_env_gating() -> None:
    saved = {
        k: os.environ.pop(k, None)
        for k in (
            "CONTEXT_ENGINE_QUERY_AGENT_MODEL",
            "CONTEXT_ENGINE_ANSWER_SYNTHESIS_MODEL",
        )
    }
    try:
        assert isinstance(_build_query_agent(), NullQueryAgent)
        os.environ["CONTEXT_ENGINE_QUERY_AGENT_MODEL"] = "openai-responses:gpt-5.4-mini"
        assert isinstance(_build_query_agent(), PydanticQueryAgent)
    finally:
        os.environ.pop("CONTEXT_ENGINE_QUERY_AGENT_MODEL", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
