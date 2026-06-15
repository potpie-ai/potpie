from __future__ import annotations

import json

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from adapters.outbound.graph.plan_stores.local_json import LocalJsonGraphPlanStore
from application.services.graph_workbench import GraphWorkbenchService
from domain.graph_plans import GraphMutationPlanRecord
from domain.ports.claim_query import ClaimQueryFilter

pytestmark = pytest.mark.unit

POT = "p"


class _MemoryPlanStore:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], GraphMutationPlanRecord] = {}

    def save(self, record: GraphMutationPlanRecord) -> None:
        self.records[(record.pot_id, record.plan_id)] = record

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        return self.records.get((pot_id, plan_id))


def _service() -> tuple[GraphWorkbenchService, InMemoryGraphBackend, _MemoryPlanStore]:
    backend = InMemoryGraphBackend()
    store = _MemoryPlanStore()
    return GraphWorkbenchService(backend=backend, plan_store=store), backend, store


def _link_payload(subject: str = "service:payments-api", object_: str = "service:ledger-api") -> dict:
    return {
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "infra_topology",
                "subject": {"key": subject, "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": object_, "type": "Service"},
                "truth": "source_observation",
                "evidence": [{"source_ref": "repo:manifest"}],
                "description": f"{subject} depends on {object_}",
            }
        ]
    }


def _end_relation_payload() -> dict:
    return {
        "operations": [
            {
                "op": "end_relation_validity",
                "subgraph": "infra_topology",
                "subject": {"key": "service:payments-api", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:ledger-api", "type": "Service"},
                "reason": "dependency removed",
            }
        ]
    }


def test_propose_persists_plan_without_writing_graph_facts() -> None:
    workbench, backend, store = _service()

    proposal = workbench.propose(_link_payload(), pot_id=POT)

    assert proposal.ok is True
    assert proposal.status == "validated"
    assert store.get(pot_id=POT, plan_id=proposal.plan_id) is not None
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []


def test_commit_applies_exact_stored_plan_by_id() -> None:
    workbench, backend, _store = _service()
    proposal = workbench.propose(_link_payload(), pot_id=POT)

    result = workbench.commit(proposal.plan_id, pot_id=POT)

    assert result.ok is True
    assert result.status == "committed"
    assert result.mutation_id
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    assert len(rows) == 1
    assert rows[0].predicate == "DEPENDS_ON"


def test_commit_rejects_stale_plan_after_graph_version_changes() -> None:
    workbench, _backend, _store = _service()
    stale = workbench.propose(_link_payload(), pot_id=POT)
    fresh = workbench.propose(
        _link_payload(subject="service:api", object_="service:db"),
        pot_id=POT,
    )
    committed = workbench.commit(fresh.plan_id, pot_id=POT)
    assert committed.ok is True

    result = workbench.commit(stale.plan_id, pot_id=POT)

    assert result.ok is False
    assert result.status == "conflict"
    assert result.expected_subgraph_versions["_global"] == 0
    assert result.current_subgraph_versions["_global"] == 1


def test_medium_risk_plan_requires_approval_before_commit() -> None:
    workbench, _backend, _store = _service()
    proposal = workbench.propose(_end_relation_payload(), pot_id=POT)

    assert proposal.ok is True
    assert proposal.status == "review_required"
    blocked = workbench.commit(proposal.plan_id, pot_id=POT)
    assert blocked.ok is False
    assert blocked.status == "review_required"
    assert "approval required" in (blocked.detail or "")

    approved = workbench.commit(
        proposal.plan_id,
        pot_id=POT,
        approved_by="user:alice",
    )

    assert approved.ok is True
    assert approved.status == "committed"
    assert approved.approval is not None
    assert approved.approval.approved_by == "user:alice"


def test_local_json_plan_store_round_trips_lowered_plan(tmp_path) -> None:
    store = LocalJsonGraphPlanStore(home=tmp_path)
    workbench = GraphWorkbenchService(
        backend=InMemoryGraphBackend(),
        plan_store=store,
    )
    proposal = workbench.propose(_link_payload(), pot_id=POT)

    reloaded = LocalJsonGraphPlanStore(home=tmp_path).get(
        pot_id=POT,
        plan_id=proposal.plan_id,
    )

    assert reloaded is not None
    assert reloaded.plan_id == proposal.plan_id
    assert reloaded.lowered_batch is not None
    assert reloaded.lowered_batch.edge_upserts[0].edge_type == "DEPENDS_ON"
    raw = json.loads((tmp_path / "graph_plans.json").read_text(encoding="utf-8"))
    assert proposal.plan_id in raw["plans"][POT]
