"""Tests for the PydanticDeepReconciliationAgent batch interface (no LLM call)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import ModuleType

import anyio
import pytest

from potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
    PydanticDeepReconciliationAgent,
)
from potpie_context_engine.bootstrap import sentry_metrics_runtime
from potpie_context_engine.core.context_events import ContextEvent
from potpie_context_engine.domain.reconciliation_batch import BatchAgentContext

pytestmark = pytest.mark.unit


def _event(eid: str) -> ContextEvent:
    return ContextEvent(
        event_id=eid,
        source_system="test",
        event_type="x",
        action="y",
        pot_id="pot-1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
        source_id=f"src-{eid}",
        occurred_at=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )


def test_capability_metadata_marks_batch_tools_v1() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()
    meta = agent.capability_metadata()
    assert meta["agent"] == "pydantic-deep"
    assert meta["toolset_version"] == "batch-tools-v1"
    assert meta["has_graph_service"] is False


def test_run_batch_without_context_graph_returns_error() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()
    ctx = BatchAgentContext(
        batch_id="b1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[_event("e1")],
    )
    out = agent.run_batch(ctx)
    assert out.ok is False
    assert out.error == "graph_service_unavailable"


def test_compose_instructions_includes_playbooks_for_batch_kinds() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()

    bootstrap = _event("e-boot")
    bootstrap.source_system = "github"
    bootstrap.event_type = "repository"
    bootstrap.action = "added"

    pr = _event("e-pr")
    pr.source_system = "github"
    pr.event_type = "pull_request"
    pr.action = "merged"

    ctx = BatchAgentContext(
        batch_id="b1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[bootstrap, pr],
    )
    text = agent._compose_instructions(ctx)
    # Both kinds present.
    assert "github / repository / added" in text
    assert "github / pull_request / merged" in text
    # Tool budget line uses the higher of the two playbook budgets.
    assert "TOOL BUDGET" in text
    # Base instructions still present.
    assert "context-graph ingestion agent" in text


def test_compose_instructions_exposes_the_full_public_ingestion_ontology() -> None:
    from potpie_context_engine.core.ontology import EDGE_TYPES, ENTITY_TYPES

    agent = PydanticDeepReconciliationAgent()
    ctx = BatchAgentContext(
        batch_id="b1", pot_id="pot-1", repo_name="o/r", events=[_event("e1")]
    )
    text = agent._compose_instructions(ctx)
    for label, spec in ENTITY_TYPES.items():
        if spec.public:
            assert f"{label} | {spec.key_prefix}: | {spec.description}" in text
    for name, spec in EDGE_TYPES.items():
        if spec.public:
            assert f"{name} | " in text
            for subject, object_ in spec.allowed_pairs:
                assert f"{subject} -> {object_}" in text


def test_compose_instructions_dedupes_repeated_event_kinds() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()

    e1 = _event("e1")
    e2 = _event("e2")
    for ev in (e1, e2):
        ev.source_system = "github"
        ev.event_type = "pull_request"
        ev.action = "merged"

    ctx = BatchAgentContext(
        batch_id="b1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[e1, e2],
    )
    text = agent._compose_instructions(ctx)
    # The kind header appears once, not twice.
    assert text.count("github / pull_request / merged") == 1


def test_run_batch_short_circuits_for_empty_event_list() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()
    # Wire a fake graph so the early return is the empty-events guard, not the
    # context-graph check.

    class _Graph:
        async def mutate_async(self, *_args, **_kwargs):
            raise AssertionError("must not be called for empty batch")

    agent.set_graph_service(_Graph())
    ctx = BatchAgentContext(
        batch_id="b1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[],
    )
    out = agent.run_batch(ctx)
    assert out.ok is True
    assert out.completed_event_ids == []


def test_run_batch_timeout_mirrors_sentry_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts: list[tuple[str, int, dict[str, str]]] = []
    monkeypatch.setattr(
        sentry_metrics_runtime,
        "count",
        lambda name, value=1, *, attributes=None, unit=None: counts.append(
            (name, value, dict(attributes or {}))
        ),
    )
    monkeypatch.setenv("CONTEXT_ENGINE_AGENT_RUN_TIMEOUT_SECS", "0.001")

    class _SlowAgent:
        def tool_plain(self, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

        async def run(self, *_args, **_kwargs):
            await anyio.sleep(1)

    fake_deep = ModuleType("pydantic_deep")
    fake_deep.CheckpointMiddleware = lambda **_kwargs: object()
    fake_deep.create_deep_agent = lambda **_kwargs: _SlowAgent()
    fake_deep.create_default_deps = lambda: object()
    monkeypatch.setitem(sys.modules, "pydantic_deep", fake_deep)
    monkeypatch.setitem(
        sys.modules, "pydantic_ai.usage", ModuleType("pydantic_ai.usage")
    )

    agent = PydanticDeepReconciliationAgent()
    agent.set_graph_service(object())
    monkeypatch.setattr(agent, "_build_read_tools", lambda _ctx: [])
    monkeypatch.setattr(agent, "_build_mutation_tools", lambda _state: [])
    monkeypatch.setattr(agent, "_build_control_tools", lambda _state: [])
    ctx = BatchAgentContext(
        batch_id="batch-1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[_event("e1")],
    )

    out = agent.run_batch(ctx)

    assert out.ok is False
    assert ("ce.agent.timeout_total", 1, {"result": "timeout"}) in counts


async def test_mutation_tool_applies_canonical_operations_to_batch_pot_and_replays():
    from potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
        _BatchRunState,
    )
    from potpie_context_engine.testing import build_test_graph_runtime
    from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter

    runtime = build_test_graph_runtime()
    state = _BatchRunState("pot-1", "o/r", {"e1": _event("e1")}, runtime.graph)
    agent = PydanticDeepReconciliationAgent()
    apply = agent._build_mutation_tools(state)[0].function
    plan = {
        "pot_id": "wrong-pot",
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "infra_topology",
                "subject": {"key": "service:api", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:db", "type": "Service"},
                "truth": "source_observation",
                "description": "API invokes DB",
                "evidence": [{"ref": "repo:manifest"}],
            }
        ],
    }
    first = await apply(plan, "e1", "dependency")
    assert first["ok"], first
    assert first["mutation_counts"]["edge_upserts_applied"] == 1
    second = await apply(plan, "e1", "dependency")
    assert second["ok"] and second["mutation_id"] == first["mutation_id"]
    assert (
        len(runtime.backend.claim_query.find_claims(ClaimQueryFilter(pot_id="pot-1")))
        == 1
    )
    assert (
        runtime.backend.claim_query.find_claims(ClaimQueryFilter(pot_id="wrong-pot"))
        == []
    )
    later_group = await apply(
        {
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {"key": "service:worker", "type": "Service"},
                }
            ]
        },
        "e1",
        "next group",
    )
    assert later_group["ok"], later_group


@pytest.mark.parametrize(
    "with_actor", [False, True], ids=["backfill", "authenticated-actor"]
)
@pytest.mark.parametrize(
    "explicit_times", [False, True], ids=["event-defaults", "explicit-claim-times"]
)
async def test_mutation_tool_freezes_trusted_event_provenance_across_restart(
    tmp_path, with_actor, explicit_times
):
    from dataclasses import replace
    from potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
        _BatchRunState,
    )
    from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
        EmbeddedGraphBackend,
    )
    from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
        LocalJsonGraphPlanStore,
    )
    from potpie_context_engine.core.actor import Actor
    from potpie_context_engine.core.runtime import build_graph_runtime
    from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
    from potpie_context_engine.core.graph_plans import GraphMutationPlanRecord
    from potpie_context_engine.core.mutation_execution import request_fingerprint
    from potpie_context_engine.testing import InMemoryGraphInboxStore

    occurred = datetime(2025, 1, 1, tzinfo=timezone.utc)
    received = datetime(2025, 1, 2, tzinfo=timezone.utc)
    actor = (
        Actor("trusted-user", "http", "trusted-client", "session")
        if with_actor
        else None
    )
    event = replace(
        _event("e1"),
        source_system="github",
        event_type="pull_request",
        source_id="pr-123",
        occurred_at=occurred,
        received_at=received,
        actor=actor,
    )

    def open_runtime():
        return build_graph_runtime(
            EmbeddedGraphBackend(home=tmp_path / "graph"),
            LocalJsonGraphPlanStore(home=tmp_path / "plans"),
            InMemoryGraphInboxStore(),
        )

    operation = {
        "op": "link_entities",
        "subgraph": "infra_topology",
        "subject": {
            "key": "service:api",
            "type": "Service",
            "properties": {
                "prov_source_ref": "forged-source",
                "actor_user_id": "forged-user",
            },
        },
        "predicate": "DEPENDS_ON",
        "object": {"key": "service:db", "type": "Service"},
        "truth": "source_observation",
        "description": "Imported deployment dependency",
        "evidence": [{"ref": "pr-123"}],
    }
    if explicit_times:
        operation.update(
            valid_from="2024-12-01T00:00:00+00:00",
            observed_at="2024-12-02T00:00:00+00:00",
        )
    payload = {
        "operations": [operation],
        "created_by": {
            "user": "forged-user",
            "surface": "cli",
            "harness": "forged-agent",
        },
        "provenance_context": {
            "source_ref": "forged-source",
            "actor_user_id": "forged-user",
        },
        "_trusted_provenance": {"source_ref": "forged-source"},
    }
    from potpie_context_engine.core.semantic_mutations import SemanticMutationRequest

    assert (
        SemanticMutationRequest.parse(payload, pot_id="pot-1").provenance_context
        is None
    )
    runtime = open_runtime()
    agent = PydanticDeepReconciliationAgent()
    state = _BatchRunState("pot-1", "o/r", {"e1": event}, runtime.graph)
    tool = agent._build_mutation_tools(state)[0].function
    first = await tool(payload, "e1", "imported dependency")
    assert first["ok"], first
    records = runtime.plan_store.list(pot_id="pot-1")
    assert len(records) == 1
    record = records[0]
    expected = {
        "prov_source_event_id": "e1",
        "prov_source_system": "github",
        "prov_source_kind": "pull_request",
        "prov_source_ref": "pr-123",
        "prov_event_occurred_at": occurred.isoformat(),
        "prov_event_received_at": received.isoformat(),
        "prov_created_by_agent": "pydantic-deep",
    }
    if actor:
        expected.update(actor.to_properties())
    row = runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="pot-1", include_invalidated=True)
    )[0]
    assert {key: row.properties.get(key) for key in expected} == expected
    entity_properties = runtime.backend.claim_query.entity_properties(
        pot_id="pot-1", entity_key="service:api"
    )
    assert {key: entity_properties.get(key) for key in expected} == expected
    assert row.source_system == "github" and row.source_ref == "pr-123"
    assert row.valid_at == (
        datetime(2024, 12, 1, tzinfo=timezone.utc) if explicit_times else occurred
    )
    assert row.observed_at == (
        datetime(2024, 12, 2, tzinfo=timezone.utc) if explicit_times else received
    )
    assert record.provenance.event_occurred_at == occurred
    assert record.provenance.event_received_at == received
    assert record.prepared.request_fingerprint == request_fingerprint(
        record.original_payload
    )
    assert record.original_payload["_trusted_provenance"]["source_ref"] == "pr-123"
    assert (
        GraphMutationPlanRecord.from_dict(record.to_dict()).provenance
        == record.provenance
    )
    if actor is None:
        assert (
            "actor_user_id" not in row.properties
            and "actor_auth_method" not in row.properties
        )
        assert "actor_user_id" not in entity_properties
    restarted = open_runtime()
    replay_state = _BatchRunState("pot-1", "o/r", {"e1": event}, restarted.graph)
    replay = await agent._build_mutation_tools(replay_state)[0].function(
        payload, "e1", "retry"
    )
    assert replay["ok"] and replay["mutation_id"] == first["mutation_id"]
    changed = replace(event, source_id="different-trusted-source")
    changed_state = _BatchRunState("pot-1", "o/r", {"e1": changed}, restarted.graph)
    refused = await agent._build_mutation_tools(changed_state)[0].function(
        payload, "e1", "changed source"
    )
    assert refused["ok"] is False
    assert "different content" in refused["error"]
    assert restarted.backend.mutation.current_version("pot-1") == 1
