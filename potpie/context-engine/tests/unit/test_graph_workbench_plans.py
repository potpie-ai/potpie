from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
import threading

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core.workbench_service import (
    GraphWorkbenchService,
)
from potpie_context_engine.core.graph_plans import (
    GraphMutationPlanRecord,
    GraphMutationPlanStatus,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter

pytestmark = pytest.mark.unit

POT = "p"


class _MemoryPlanStore:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], GraphMutationPlanRecord] = {}
        self.lock = threading.Lock()

    def save(self, record: GraphMutationPlanRecord) -> None:
        with self.lock:
            self.records[(record.pot_id, record.plan_id)] = record

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        with self.lock:
            return self.records.get((pot_id, plan_id))

    def compare_and_set(
        self,
        *,
        expected: GraphMutationPlanRecord,
        replacement: GraphMutationPlanRecord,
    ) -> bool:
        key = (expected.pot_id, expected.plan_id)
        with self.lock:
            if self.records.get(key) != expected:
                return False
            self.records[key] = replacement
            return True

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
        with self.lock:
            records = [
                record for (pid, _), record in self.records.items() if pid == pot_id
            ]
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


class _RacingPlanStore(_MemoryPlanStore):
    def __init__(self) -> None:
        super().__init__()
        self.commit_claims = threading.Barrier(2)
        self.claim_rejected = threading.Event()
        self.rejected_claim_observed = threading.Event()

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        record = super().get(pot_id=pot_id, plan_id=plan_id)
        if (
            self.claim_rejected.is_set()
            and record is not None
            and record.status == GraphMutationPlanStatus.committing.value
        ):
            self.rejected_claim_observed.set()
        return record

    def compare_and_set(
        self,
        *,
        expected: GraphMutationPlanRecord,
        replacement: GraphMutationPlanRecord,
    ) -> bool:
        if replacement.status == GraphMutationPlanStatus.committing.value:
            self.commit_claims.wait(timeout=5)
        claimed = super().compare_and_set(
            expected=expected,
            replacement=replacement,
        )
        if replacement.status == GraphMutationPlanStatus.committing.value:
            if claimed:
                assert self.rejected_claim_observed.wait(timeout=5)
            else:
                self.claim_rejected.set()
        return claimed


class _ProcessRacingLocalStore(LocalJsonGraphPlanStore):
    __slots__ = ("loaded",)

    def __init__(self, *, home, loaded) -> None:
        super().__init__(home=home)
        self.loaded = loaded

    def _load(self):
        state = super()._load()
        try:
            self.loaded.wait(timeout=1)
        except threading.BrokenBarrierError:
            pass
        return state


def _local_json_cas_worker(
    home,
    expected,
    replacement,
    ready,
    loaded,
    results,
) -> None:
    store = _ProcessRacingLocalStore(home=home, loaded=loaded)
    ready.wait(timeout=5)
    try:
        changed = store.compare_and_set(
            expected=expected,
            replacement=replacement,
        )
    except Exception as exc:
        results.put(("error", repr(exc)))
    else:
        results.put(("ok", changed))


class _RaiseOnceMutation:
    def __init__(self, inner, *, after_side_effect: bool) -> None:
        self.inner = inner
        self.after_side_effect = after_side_effect
        self.mutation_ids: list[str | None] = []
        self.raised = False

    def apply(self, plan, **kwargs):
        provenance = kwargs.get("provenance_context")
        self.mutation_ids.append(
            provenance.mutation_id if provenance is not None else None
        )
        if not self.raised:
            self.raised = True
            if self.after_side_effect:
                self.inner.apply(plan, **kwargs)
            raise RuntimeError("injected mutation failure")
        return self.inner.apply(plan, **kwargs)

    def lookup_execution(self, plan, **kwargs):
        return self.inner.lookup_execution(plan, **kwargs)


def _service() -> tuple[GraphWorkbenchService, InMemoryGraphBackend, _MemoryPlanStore]:
    backend = InMemoryGraphBackend()
    store = _MemoryPlanStore()
    return GraphWorkbenchService(backend=backend, plan_store=store), backend, store


def test_concurrent_commit_claims_a_plan_before_mutating() -> None:
    mutation_count = 0
    mutation_lock = threading.Lock()

    def count_mutation() -> None:
        nonlocal mutation_count
        with mutation_lock:
            mutation_count += 1

    backend = InMemoryGraphBackend(on_change=count_mutation)
    store = _RacingPlanStore()
    workbench = GraphWorkbenchService(backend=backend, plan_store=store)
    proposal = workbench.propose(_link_payload(), pot_id=POT)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(workbench.commit, proposal.plan_id, pot_id=POT)
        second = pool.submit(workbench.commit, proposal.plan_id, pot_id=POT)
        results = (first.result(), second.result())

    assert sum(result.ok for result in results) == 1
    winner = next(result for result in results if result.ok)
    loser = next(result for result in results if not result.ok)
    assert winner.mutation_id
    assert loser.mutation_id == winner.mutation_id
    assert "concurrent" in (loser.detail or "")
    stored = store.get(pot_id=POT, plan_id=proposal.plan_id)
    assert stored is not None
    assert stored.status == "committed"
    assert mutation_count == 1
    assert {result.mutation_id for result in results if result.mutation_id} == {
        stored.mutation_id
    }


def test_crashed_committing_plan_has_single_stale_recovery_claimer() -> None:
    mutation_count = 0
    mutation_lock = threading.Lock()

    def count_mutation() -> None:
        nonlocal mutation_count
        with mutation_lock:
            mutation_count += 1

    backend = InMemoryGraphBackend(on_change=count_mutation)
    store = _RacingPlanStore()
    workbench = GraphWorkbenchService(
        backend=backend,
        plan_store=store,
        commit_claim_timeout_seconds=1,
    )
    proposal = workbench.propose(_link_payload(), pot_id=POT)
    proposed = store.get(pot_id=POT, plan_id=proposal.plan_id)
    assert proposed is not None
    store.save(
        replace(
            proposed,
            status=GraphMutationPlanStatus.committing.value,
            mutation_id="mutation:crash-recovery",
            commit_attempt_id="attempt:crashed-worker",
            commit_attempt_started_at=datetime.now(timezone.utc)
            - timedelta(seconds=10),
        )
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            workbench.commit,
            proposal.plan_id,
            pot_id=POT,
            recover_stale=True,
        )
        second = pool.submit(
            workbench.commit,
            proposal.plan_id,
            pot_id=POT,
            recover_stale=True,
        )
        results = (first.result(), second.result())

    assert sum(result.ok for result in results) == 1
    assert mutation_count == 1
    assert {result.mutation_id for result in results} == {"mutation:crash-recovery"}
    stored = store.get(pot_id=POT, plan_id=proposal.plan_id)
    assert stored is not None
    assert stored.status == GraphMutationPlanStatus.committed.value
    assert stored.commit_attempt_id is None
    assert stored.commit_attempt_started_at is None


def test_stale_recovery_during_backend_callback_does_not_reapply() -> None:
    callback_entered = threading.Event()
    release_callback = threading.Event()
    mutation_count = 0

    def block_after_mutation() -> None:
        nonlocal mutation_count
        mutation_count += 1
        callback_entered.set()
        assert release_callback.wait(timeout=10)

    backend = InMemoryGraphBackend(on_change=block_after_mutation)
    store = _MemoryPlanStore()
    workbench = GraphWorkbenchService(
        backend=backend,
        plan_store=store,
        commit_claim_timeout_seconds=0,
    )
    proposal = workbench.propose(_link_payload(), pot_id=POT)

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(workbench.commit, proposal.plan_id, pot_id=POT)
        assert callback_entered.wait(timeout=10)
        active = store.get(pot_id=POT, plan_id=proposal.plan_id)
        assert active is not None
        store.save(
            replace(
                active,
                commit_attempt_started_at=datetime.now(timezone.utc)
                - timedelta(seconds=10),
            )
        )
        recovering = pool.submit(
            workbench.commit,
            proposal.plan_id,
            pot_id=POT,
            recover_stale=True,
        ).result(timeout=10)
        store.save(active)
        release_callback.set()
        committed = owner.result(timeout=10)

    assert recovering.ok is False
    assert recovering.status == GraphMutationPlanStatus.committing.value
    assert "still in flight" in (recovering.detail or "")
    assert committed.ok is True
    assert recovering.mutation_id == committed.mutation_id
    assert mutation_count == 1
    assert len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 1


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


def test_retrying_committed_plan_returns_persisted_receipt() -> None:
    workbench, _backend, store = _service()
    proposal = workbench.propose(_link_payload(), pot_id=POT)
    committed = workbench.commit(proposal.plan_id, pot_id=POT)
    stored = store.get(pot_id=POT, plan_id=proposal.plan_id)

    retried = workbench.commit(proposal.plan_id, pot_id=POT)

    assert stored is not None
    assert retried.ok is True
    assert retried.status == GraphMutationPlanStatus.committed.value
    assert retried.mutation_id == committed.mutation_id == stored.mutation_id
    assert retried.applied_at == committed.applied_at == stored.committed_at
    assert retried.expected_subgraph_versions == stored.expected_subgraph_versions
    assert retried.current_subgraph_versions == stored.current_subgraph_versions
    assert retried.new_subgraph_versions == stored.final_subgraph_versions
    assert retried.diff == stored.diff
    assert retried.claim_keys == committed.claim_keys


def test_backend_exception_before_side_effect_releases_plan_for_retry() -> None:
    workbench, backend, store = _service()
    flaky = _RaiseOnceMutation(backend.mutation, after_side_effect=False)
    backend._mutation = flaky
    proposal = workbench.propose(_link_payload(), pot_id=POT)

    failed = workbench.commit(proposal.plan_id, pot_id=POT)
    failed_record = store.get(pot_id=POT, plan_id=proposal.plan_id)

    assert failed.ok is False
    assert failed.status == GraphMutationPlanStatus.error.value
    assert failed.mutation_id
    assert failed_record is not None
    assert failed_record.status == GraphMutationPlanStatus.error.value
    assert failed_record.commit_attempt_id is None
    assert failed_record.commit_attempt_started_at is None
    assert failed.detail == "backend_exception: backend mutation failed unexpectedly"
    assert failed_record.detail == failed.detail
    assert "injected mutation failure" not in failed.detail
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []

    retried = workbench.commit(proposal.plan_id, pot_id=POT)

    assert retried.ok is True
    assert retried.mutation_id == failed.mutation_id
    assert flaky.mutation_ids == [failed.mutation_id, failed.mutation_id]


def test_retry_after_confirmed_absence_rechecks_graph_versions() -> None:
    workbench, backend, _store = _service()
    flaky = _RaiseOnceMutation(backend.mutation, after_side_effect=False)
    backend._mutation = flaky
    stale = workbench.propose(_link_payload(), pot_id=POT)

    failed = workbench.commit(stale.plan_id, pot_id=POT)
    fresh = workbench.propose(
        _link_payload(subject="service:api", object_="service:db"),
        pot_id=POT,
    )
    assert workbench.commit(fresh.plan_id, pot_id=POT).ok

    retried = workbench.commit(stale.plan_id, pot_id=POT)

    assert failed.status == GraphMutationPlanStatus.error.value
    assert retried.ok is False
    assert retried.status == GraphMutationPlanStatus.conflict.value
    assert retried.expected_subgraph_versions["_global"] == 0
    assert retried.current_subgraph_versions["_global"] == 1
    assert flaky.mutation_ids.count(failed.mutation_id) == 1


def test_backend_exception_after_side_effect_retries_idempotently() -> None:
    mutation_count = 0

    def count_mutation() -> None:
        nonlocal mutation_count
        mutation_count += 1

    backend = InMemoryGraphBackend(on_change=count_mutation)
    store = _MemoryPlanStore()
    workbench = GraphWorkbenchService(backend=backend, plan_store=store)
    flaky = _RaiseOnceMutation(backend.mutation, after_side_effect=True)
    backend._mutation = flaky
    proposal = workbench.propose(_link_payload(), pot_id=POT)

    failed = workbench.commit(proposal.plan_id, pot_id=POT)
    rows_after_failure = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    retried = workbench.commit(proposal.plan_id, pot_id=POT)
    rows_after_retry = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))

    assert failed.ok is False
    assert failed.status == GraphMutationPlanStatus.error.value
    assert failed.mutation_id
    assert len(rows_after_failure) == 1
    assert rows_after_failure[0].mutation_id == failed.mutation_id
    assert retried.ok is True
    assert retried.mutation_id == failed.mutation_id
    assert flaky.mutation_ids == [failed.mutation_id]
    assert mutation_count == 1
    assert len(rows_after_retry) == 1
    assert rows_after_retry[0].mutation_id == failed.mutation_id
    stored = store.get(pot_id=POT, plan_id=proposal.plan_id)
    assert stored is not None
    assert stored.status == GraphMutationPlanStatus.committed.value


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


def test_local_json_plan_store_rejects_a_stale_state_transition(tmp_path) -> None:
    first_store = LocalJsonGraphPlanStore(home=tmp_path)
    workbench = GraphWorkbenchService(
        backend=InMemoryGraphBackend(),
        plan_store=first_store,
    )
    proposal = workbench.propose(_link_payload(), pot_id=POT)
    expected = first_store.get(pot_id=POT, plan_id=proposal.plan_id)
    assert expected is not None
    attempt_started_at = datetime.now(timezone.utc)
    claimed = replace(
        expected,
        status=GraphMutationPlanStatus.committing.value,
        mutation_id="mutation:local-cas",
        commit_attempt_id="attempt:local-cas",
        commit_attempt_started_at=attempt_started_at,
    )

    second_store = LocalJsonGraphPlanStore(home=tmp_path)
    assert second_store.compare_and_set(
        expected=expected,
        replacement=claimed,
    )
    assert not first_store.compare_and_set(
        expected=expected,
        replacement=replace(expected, status=GraphMutationPlanStatus.expired.value),
    )
    reloaded = first_store.get(pot_id=POT, plan_id=proposal.plan_id)
    assert reloaded is not None
    assert reloaded.mutation_id == "mutation:local-cas"
    assert reloaded.commit_attempt_id == "attempt:local-cas"
    assert reloaded.commit_attempt_started_at == attempt_started_at


def test_local_json_plan_store_compare_and_set_is_process_safe(tmp_path) -> None:
    store = LocalJsonGraphPlanStore(home=tmp_path)
    workbench = GraphWorkbenchService(
        backend=InMemoryGraphBackend(),
        plan_store=store,
    )
    proposal = workbench.propose(_link_payload(), pot_id=POT)
    expected = store.get(pot_id=POT, plan_id=proposal.plan_id)
    assert expected is not None
    replacements = (
        replace(expected, status=GraphMutationPlanStatus.committing.value),
        replace(expected, status=GraphMutationPlanStatus.expired.value),
    )

    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2)
    loaded = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_local_json_cas_worker,
            args=(tmp_path, expected, replacement, ready, loaded, results),
        )
        for replacement in replacements
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=10)
        outcomes = [results.get(timeout=2) for _ in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
        results.close()

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(outcomes) == [("ok", False), ("ok", True)]
    current = store.get(pot_id=POT, plan_id=proposal.plan_id)
    assert current is not None
    assert current.status in {
        GraphMutationPlanStatus.committing.value,
        GraphMutationPlanStatus.expired.value,
    }


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
