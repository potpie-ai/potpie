"""Context-aware Ingestion Agent: bounded tools, snapshot, evidence warnings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import pytest

from potpie_context_engine.adapters.outbound.reconciliation.context_graph_tools import (
    ContextGraphReconciliationTools,
    build_initial_context_snapshot,
)
from potpie_context_engine.application.services.reconciliation_validation import (
    validate_reconciliation_plan,
)
from potpie_context_engine.domain.agent_envelope import AgentEnvelope
from potpie_context_engine.domain.context_events import ContextEvent, EventRef
from potpie_context_engine.domain.graph_mutations import EntityUpsert
from potpie_context_engine.domain.llm_reconciliation import ReconciliationRequest
from potpie_context_engine.domain.ports.agent_context import ResolveRequest
from potpie_context_engine.domain.reconciliation import ReconciliationPlan


def _make_request(
    *,
    pot_id: str = "pot-1",
    repo_name: str = "owner/repo",
    event_type: str = "pull_request",
    source_id: str = "owner/repo#42",
    payload: dict[str, Any] | None = None,
) -> ReconciliationRequest:
    ev = ContextEvent(
        event_id="evt-1",
        pot_id=pot_id,
        provider="github",
        provider_host="github.com",
        repo_name=repo_name,
        source_system="github",
        event_type=event_type,
        action="opened",
        source_id=source_id or "",
        source_event_id="src-1",
        payload=payload or {"title": "Add telemetry to ingestion"},
        occurred_at=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
        received_at=datetime(2026, 4, 22, 12, 1, tzinfo=timezone.utc),
    )
    return ReconciliationRequest(event=ev, pot_id=pot_id, repo_name=repo_name)


class _FakeBackend:
    enabled = True


class _FakeGraph:
    def __init__(self) -> None:
        self.backend = _FakeBackend()
        self.calls: list[ResolveRequest] = []

    def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        self.calls.append(request)
        return AgentEnvelope(
            pot_id=request.pot_id,
            intent=request.intent or "unknown",
            items=(),
            coverage=(),
            metadata={"max_items": request.max_items},
        )


def test_tools_adapter_scopes_query_to_request_pot_and_repo() -> None:
    graph = _FakeGraph()
    tools = ContextGraphReconciliationTools(graph)
    req = _make_request(pot_id="pot-42", repo_name="acme/api")
    out = tools.execute_read_tool(req, "context_search", {"query": "telemetry"})
    assert out["kind"] == "resolve"
    assert graph.calls[-1].pot_id == "pot-42"
    assert graph.calls[-1].scope["repo_name"] == "acme/api"
    assert graph.calls[-1].task == "telemetry"
    assert graph.calls[-1].include == ()  # generic search routes by intent


def test_tools_adapter_rejects_empty_query() -> None:
    graph = _FakeGraph()
    tools = ContextGraphReconciliationTools(graph)
    req = _make_request()
    out = tools.execute_read_tool(req, "context_search", {"query": "   "})
    assert out["kind"] == "error"
    assert out["error"] == "query_required"
    assert graph.calls == []


def test_tools_adapter_lists_expected_tool_catalog() -> None:
    graph = _FakeGraph()
    tools = ContextGraphReconciliationTools(graph)
    names = {t.name for t in tools.list_tools(_make_request())}
    assert names == {
        "context_search",
        "context_coding_preferences",
        "context_infra_topology",
        "context_timeline",
        "context_prior_bugs",
    }


def test_read_tool_includes_are_orchestrator_backed() -> None:
    """Every targeted read tool must name an include the orchestrator backs;
    this guard stops the tool surface from drifting from the readers again."""
    from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
        InMemoryClaimQueryStore,
    )
    from potpie_context_engine.adapters.outbound.reconciliation.context_graph_tools import (
        READ_TOOL_INCLUDE,
    )
    from potpie_context_engine.application.services.read_orchestrator import (
        ReadOrchestrator,
    )

    backed = ReadOrchestrator(claim_query=InMemoryClaimQueryStore()).backed_includes
    targeted = {inc for inc in READ_TOOL_INCLUDE.values() if inc is not None}
    assert targeted <= backed, f"unbacked read-tool includes: {targeted - backed}"


def test_tools_adapter_returns_disabled_when_graph_off() -> None:
    graph = _FakeGraph()
    graph.backend.enabled = False
    tools = ContextGraphReconciliationTools(graph)
    out = tools.execute_read_tool(_make_request(), "context_search", {"query": "x"})
    assert out == {"error": "context_graph_disabled", "kind": "error"}


def test_timeline_runs_without_a_scope_target() -> None:
    graph = _FakeGraph()
    tools = ContextGraphReconciliationTools(graph)
    out = tools.execute_read_tool(_make_request(), "context_timeline", {})
    assert out["kind"] == "resolve"
    assert graph.calls[-1].include == ("timeline",)


def test_unknown_tool_name_returns_error() -> None:
    graph = _FakeGraph()
    tools = ContextGraphReconciliationTools(graph)
    out = tools.execute_read_tool(_make_request(), "context_magic", {})
    assert out["kind"] == "error"
    assert out["error"].startswith("unknown_tool")


def test_build_initial_context_snapshot_bundles_topology_and_seeds() -> None:
    graph = _FakeGraph()
    tools = ContextGraphReconciliationTools(graph)
    snap = build_initial_context_snapshot(
        tools,
        _make_request(payload={"title": "Investigate alert storm"}),
        semantic_seed="Investigate alert storm",
    )
    assert "infra_topology" in snap
    assert "semantic_seed" in snap
    assert "source_ref_hits" in snap  # event has source_id


def test_snapshot_omits_semantic_seed_when_none() -> None:
    graph = _FakeGraph()
    tools = ContextGraphReconciliationTools(graph)
    req = _make_request(source_id="", payload={})
    req.event.source_event_id = None  # no source ref at all
    snap = build_initial_context_snapshot(tools, req, semantic_seed=None)
    assert "infra_topology" in snap
    assert "semantic_seed" not in snap
    assert "source_ref_hits" not in snap


def _plan_with_n_entities(n: int) -> ReconciliationPlan:
    return ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="github", pot_id="pot-1"),
        summary="s",
        entity_upserts=[
            EntityUpsert(
                entity_key=f"obs:{i}",
                labels=("Entity", "Observation"),
                properties={
                    "summary": f"obs {i}",
                    "observed_at": "2026-04-22T00:00:00+00:00",
                },
            )
            for i in range(n)
        ],
    )


def test_validation_emits_evidence_warning_for_material_plans() -> None:
    plan = _plan_with_n_entities(4)
    validate_reconciliation_plan(plan, "pot-1")
    assert any("evidence" in w for w in plan.warnings)


def test_validation_no_warning_for_small_plans() -> None:
    plan = _plan_with_n_entities(2)
    validate_reconciliation_plan(plan, "pot-1")
    assert not any("evidence" in w for w in plan.warnings)


def test_validation_no_warning_when_confidence_set() -> None:
    plan = _plan_with_n_entities(4)
    plan.confidence = 0.8
    validate_reconciliation_plan(plan, "pot-1")
    assert not any("evidence" in w for w in plan.warnings)


def test_pydantic_deep_agent_exposes_tools_setter() -> None:
    pytest.importorskip("pydantic_deep")
    from potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
        PydanticDeepReconciliationAgent,
    )

    agent = PydanticDeepReconciliationAgent()
    base_meta = agent.capability_metadata()
    assert base_meta["toolset_version"] == "batch-tools-v1"
    assert base_meta["has_context_tools"] is False
    agent.set_context_tools(ContextGraphReconciliationTools(_FakeGraph()))
    after = agent.capability_metadata()
    assert after["has_context_tools"] is True
