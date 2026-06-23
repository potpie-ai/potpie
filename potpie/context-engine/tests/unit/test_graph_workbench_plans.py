from __future__ import annotations

from datetime import datetime
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

    def list(
        self,
        *,
        pot_id: str,
        plan_id: str | None = None,
        mutation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphMutationPlanRecord, ...]:
        records = [record for (pid, _), record in self.records.items() if pid == pot_id]
        if plan_id:
            records = [record for record in records if record.plan_id == plan_id]
        if mutation_id:
            records = [
                record for record in records if record.mutation_id == mutation_id
            ]
        if since or until:
            records = [
                record
                for record in records
                if _record_in_window(record, since=since, until=until)
            ]
        records.sort(
            key=lambda record: record.committed_at or record.created_at,
            reverse=True,
        )
        if limit is not None and limit >= 0:
            records = records[:limit]
        return tuple(records)


def _service() -> tuple[GraphWorkbenchService, InMemoryGraphBackend, _MemoryPlanStore]:
    backend = InMemoryGraphBackend()
    store = _MemoryPlanStore()
    return GraphWorkbenchService(backend=backend, plan_store=store), backend, store


def _link_payload(
    subject: str = "service:payments-api", object_: str = "service:ledger-api"
) -> dict:
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


def _owner_payload(team: str) -> dict:
    return {
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "owners",
                "subject": {"key": "service:payments-api", "type": "Service"},
                "predicate": "OWNED_BY",
                "object": {"key": team, "type": "Team"},
                "truth": "source_observation",
                "evidence": [{"source_ref": "repo:owners"}],
                "description": f"payments-api is owned by {team}",
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


def _patch_entity_payload() -> dict:
    return {
        "operations": [
            {
                "op": "patch_entity",
                "subject": {"key": "service:payments-api", "type": "Service"},
                "patch": {"summary": "Payments API service"},
                "expected_entity_version": "entity-version:1",
                "reason": "tighten display metadata",
            }
        ]
    }


def _transition_state_payload() -> dict:
    return {
        "operations": [
            {
                "op": "transition_state",
                "subject": {"key": "decision:adr-1", "type": "Decision"},
                "from_state": "proposed",
                "to_state": "accepted",
                "expected_entity_version": "entity-version:2",
                "reason": "ADR approved by maintainers",
                "observed_at": "2026-06-15T10:00:00+00:00",
            }
        ]
    }


def _supersede_claim_payload() -> dict:
    return {
        "operations": [
            {
                "op": "supersede_claim",
                "subgraph": "infra_topology",
                "subject": {"key": "service:payments-api", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:ledger-api", "type": "Service"},
                "superseded_by": {"key": "service:ledger-v2", "type": "Service"},
                "reason": "dependency target changed",
                "description": "payments-api now depends on ledger-v2",
            }
        ]
    }


def _merge_duplicate_entities_payload() -> dict:
    return {
        "operations": [
            {
                "op": "merge_duplicate_entities",
                "subject": {"key": "service:payments-api-v1", "type": "Service"},
                "object": {"key": "service:payments-api", "type": "Service"},
                "external_ids": {
                    "losing": {"source": "manifest:payments-api-v1.yaml"},
                    "winning": {"source": "manifest:payments-api.yaml"},
                },
                "reason": "manifests refer to the same deployable service",
                "description": "payments-api-v1 is a duplicate identity for payments-api",
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


def test_commit_verify_reads_back_claim_and_quality_summary() -> None:
    workbench, _backend, _store = _service()
    proposal = workbench.propose(_link_payload(), pot_id=POT)

    result = workbench.commit(proposal.plan_id, pot_id=POT, verify=True)

    assert result.ok is True
    assert result.verification is not None
    assert result.verification.ok is True
    assert result.verification.status == "ok"
    assert result.verification.readback_count == 1
    assert result.verification.readback_claim_keys == proposal.claim_keys
    assert result.verification.missing_claim_keys == ()
    assert result.verification.quality_status == "ok"
    assert result.to_dict()["verification"]["readback_count"] == 1


def test_commit_verify_flags_quality_regression() -> None:
    workbench, _backend, _store = _service()
    first = workbench.propose(_owner_payload("team:platform"), pot_id=POT)
    committed_first = workbench.commit(first.plan_id, pot_id=POT, verify=True)
    assert committed_first.verification is not None
    assert committed_first.verification.ok is True

    second = workbench.propose(_owner_payload("team:product"), pot_id=POT)
    committed_second = workbench.commit(second.plan_id, pot_id=POT, verify=True)

    assert committed_second.ok is True
    assert committed_second.verification is not None
    assert committed_second.verification.ok is False
    assert committed_second.verification.status == "degraded"
    assert "conflicting_claims" in committed_second.verification.quality_regressions
    assert committed_second.recommended_next_action


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


def test_patch_entity_plan_requires_approval_then_updates_entity_metadata() -> None:
    workbench, backend, _store = _service()
    proposal = workbench.propose(_patch_entity_payload(), pot_id=POT)

    assert proposal.ok is True
    assert proposal.status == "review_required"
    assert proposal.risk == "medium"
    assert proposal.diff.entity_upserts == 1
    blocked = workbench.commit(proposal.plan_id, pot_id=POT)
    assert blocked.ok is False
    assert "approval required" in (blocked.detail or "")

    committed = workbench.commit(
        proposal.plan_id,
        pot_id=POT,
        approved_by="user:alice",
    )

    assert committed.ok is True
    props = backend.store.entity_properties(
        pot_id=POT,
        entity_key="service:payments-api",
    )
    assert props["summary"] == "Payments API service"
    assert props["last_semantic_op"] == "patch_entity"
    history = workbench.history(pot_id=POT, entity_key="service:payments-api")
    assert any(entry.plan_id == proposal.plan_id for entry in history.entries)


def test_transition_state_commits_entity_state_and_audit_claim() -> None:
    workbench, backend, _store = _service()
    proposal = workbench.propose(_transition_state_payload(), pot_id=POT)

    assert proposal.ok is True
    assert proposal.status == "review_required"
    assert proposal.claim_keys
    committed = workbench.commit(
        proposal.plan_id,
        pot_id=POT,
        approved_by="user:alice",
    )

    assert committed.ok is True
    props = backend.store.entity_properties(pot_id=POT, entity_key="decision:adr-1")
    assert props["lifecycle_state"] == "accepted"
    assert props["previous_lifecycle_state"] == "proposed"
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    assert len(rows) == 1
    assert rows[0].predicate == "MENTIONS"
    assert rows[0].object_key == "decision:adr-1"
    assert rows[0].truth == "timeline_event"
    history = workbench.history(pot_id=POT, mutation_id=committed.mutation_id)
    assert {entry.kind for entry in history.entries} == {"plan", "claim"}


def test_supersede_claim_requires_approval_then_replaces_live_claim() -> None:
    workbench, backend, _store = _service()
    original = workbench.propose(_link_payload(), pot_id=POT)
    workbench.commit(original.plan_id, pot_id=POT)

    proposal = workbench.propose(_supersede_claim_payload(), pot_id=POT)

    assert proposal.ok is True
    assert proposal.status == "review_required"
    assert proposal.risk == "high"
    assert proposal.diff.edge_upserts == 1
    assert proposal.diff.invalidations == 1
    blocked = workbench.commit(proposal.plan_id, pot_id=POT)
    assert blocked.ok is False
    assert "approval required" in (blocked.detail or "")

    committed = workbench.commit(
        proposal.plan_id,
        pot_id=POT,
        approved_by="user:alice",
    )

    assert committed.ok is True
    live_rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    assert len(live_rows) == 1
    assert live_rows[0].predicate == "DEPENDS_ON"
    assert live_rows[0].object_key == "service:ledger-v2"
    all_rows = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, include_invalidated=True)
    )
    assert len(all_rows) == 2
    assert any(
        row.object_key == "service:ledger-api" and row.invalid_at for row in all_rows
    )
    history = workbench.history(pot_id=POT, entity_key="service:ledger-api")
    assert any(entry.plan_id == proposal.plan_id for entry in history.entries)


def test_merge_duplicate_entities_requires_approval_and_writes_merge_record() -> None:
    workbench, backend, _store = _service()
    proposal = workbench.propose(_merge_duplicate_entities_payload(), pot_id=POT)

    assert proposal.ok is True
    assert proposal.status == "review_required"
    assert proposal.risk == "high"
    assert proposal.diff.edge_upserts == 1
    assert proposal.diff.invalidations == 0
    blocked = workbench.commit(proposal.plan_id, pot_id=POT)
    assert blocked.ok is False
    assert "approval required" in (blocked.detail or "")

    committed = workbench.commit(
        proposal.plan_id,
        pot_id=POT,
        approved_by="user:alice",
    )

    assert committed.ok is True
    props = backend.store.entity_properties(
        pot_id=POT,
        entity_key="service:payments-api-v1",
    )
    assert props["merged_into"] == "service:payments-api"
    assert props["merge_status"] == "merged_duplicate"
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    assert len(rows) == 1
    assert rows[0].predicate == "RELATED_TO"
    assert rows[0].subject_key == "service:payments-api-v1"
    assert rows[0].object_key == "service:payments-api"
    assert rows[0].properties["merge_record"] is True
    history = workbench.history(pot_id=POT, entity_key="service:payments-api-v1")
    assert any(entry.plan_id == proposal.plan_id for entry in history.entries)


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


def test_history_by_plan_shows_validation_commit_and_sources() -> None:
    workbench, _backend, _store = _service()
    proposal = workbench.propose(_link_payload(), pot_id=POT)
    committed = workbench.commit(proposal.plan_id, pot_id=POT)

    history = workbench.history(pot_id=POT, plan_id=proposal.plan_id)

    assert history.ok is True
    plan_entries = [entry for entry in history.entries if entry.kind == "plan"]
    assert len(plan_entries) == 1
    entry = plan_entries[0]
    assert entry.plan_id == proposal.plan_id
    assert entry.status == "committed"
    assert entry.mutation_id == committed.mutation_id
    assert "repo:manifest" in entry.source_refs
    assert entry.payload["validation_issues"] == []
    assert entry.payload["diff"]["claim_keys"] == list(committed.claim_keys)


def test_history_by_mutation_returns_plan_and_claim_rows() -> None:
    workbench, _backend, _store = _service()
    proposal = workbench.propose(_link_payload(), pot_id=POT)
    committed = workbench.commit(proposal.plan_id, pot_id=POT)

    history = workbench.history(pot_id=POT, mutation_id=committed.mutation_id)

    assert {entry.kind for entry in history.entries} == {"plan", "claim"}
    claim_entry = next(entry for entry in history.entries if entry.kind == "claim")
    assert claim_entry.mutation_id == committed.mutation_id
    assert claim_entry.claim_key in committed.claim_keys
    assert claim_entry.source_refs == ("repo:manifest",)


def test_history_by_entity_includes_invalidated_claims() -> None:
    workbench, backend, _store = _service()
    first = workbench.propose(_link_payload(), pot_id=POT)
    workbench.commit(first.plan_id, pot_id=POT)
    end = workbench.propose(_end_relation_payload(), pot_id=POT)
    workbench.commit(end.plan_id, pot_id=POT, approved_by="user:alice")

    before = len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)))
    history = workbench.history(pot_id=POT, entity_key="service:payments-api")
    after = len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)))

    assert before == after
    claim_entries = [entry for entry in history.entries if entry.kind == "claim"]
    assert any(entry.status == "invalidated" for entry in claim_entries)
    assert any("service:payments-api" in entry.entity_keys for entry in history.entries)


def test_local_json_plan_store_lists_history_records(tmp_path) -> None:
    store = LocalJsonGraphPlanStore(home=tmp_path)
    workbench = GraphWorkbenchService(
        backend=InMemoryGraphBackend(),
        plan_store=store,
    )
    proposal = workbench.propose(_link_payload(), pot_id=POT)
    committed = workbench.commit(proposal.plan_id, pot_id=POT)

    records = LocalJsonGraphPlanStore(home=tmp_path).list(
        pot_id=POT,
        mutation_id=committed.mutation_id,
    )

    assert len(records) == 1
    assert records[0].plan_id == proposal.plan_id


def _record_in_window(
    record: GraphMutationPlanRecord,
    *,
    since: datetime | None,
    until: datetime | None,
) -> bool:
    times = (record.created_at, record.committed_at)
    for value in times:
        if value is None:
            continue
        if since is not None and value < since:
            continue
        if until is not None and value > until:
            continue
        return True
    return False
