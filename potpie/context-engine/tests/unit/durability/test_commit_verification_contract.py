"""Execution receipts survive current-state verification and its failures."""

import asyncio
import json
import threading
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from potpie_context_engine.core import workbench_service
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.graph_plans import GraphMutationCommitResult
from potpie_context_engine.core.runtime import build_graph_runtime
from potpie_context_engine.results import result_from_payload
from tests.unit.mutations.test_graph_workbench_plans import _service, _link_payload, POT
from tests.unit.durability.test_mutation_recovery_contract import _prepared, _execute

pytestmark = pytest.mark.unit


def _proposal(workbench):
    payload = _link_payload()
    payload["operations"][0]["valid_from"] = "2025-01-01T00:00:00Z"
    proposal = workbench.propose(payload, pot_id=POT)
    assert proposal.ok
    return proposal


@pytest.mark.parametrize(
    "failure", [RuntimeError, TimeoutError, asyncio.CancelledError]
)
def test_post_receipt_failure_does_not_hide_execution(monkeypatch, failure):
    workbench, backend, _ = _service()
    proposal = _proposal(workbench)

    def fail(*args, **kwargs):
        raise failure("injected readback failure")

    monkeypatch.setattr(workbench_service, "_verify_ingestion_commit", fail)
    committed = workbench.commit(proposal.plan_id, pot_id=POT, verify=True)
    assert committed.ok and committed.status == "committed"
    assert committed.verification_state == "deferred"
    assert committed.verification.outcome == "inconclusive"
    assert committed.execution.version_after == 1
    recovered = workbench.commit_status(proposal.plan_id, pot_id=POT)
    assert recovered.execution == committed.execution
    assert recovered.applied_at == committed.applied_at
    assert backend.mutation.current_version(POT) == 1


@pytest.mark.parametrize(
    "verify,defer,state",
    [
        (False, False, "not_requested"),
        (False, True, "not_requested"),
        (True, True, "deferred"),
        (True, False, "completed"),
    ],
)
def test_commit_states_roundtrip_and_replay(verify, defer, state):
    workbench, backend, _ = _service()
    proposal = _proposal(workbench)
    first = workbench.commit(
        proposal.plan_id, pot_id=POT, verify=verify, defer_verification=defer
    )
    assert first.ok and first.verification_state == state
    decoded = result_from_payload(GraphMutationCommitResult, first.to_dict())
    assert json.dumps(decoded.to_dict(), sort_keys=True) == json.dumps(
        first.to_dict(), sort_keys=True
    )
    replay = workbench.commit(
        proposal.plan_id, pot_id=POT, verify=verify, defer_verification=defer
    )
    assert replay.verification_state == state
    assert replay.execution == first.execution
    assert backend.mutation.current_version(POT) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("valid_at", datetime(2025, 2, 1, tzinfo=timezone.utc)),
        ("source_system", "corrupt-source"),
        ("source_ref", "wrong-source"),
        ("record_id", "new-incarnation"),
    ],
)
def test_exact_start_source_and_incarnation_are_verified(field, value):
    workbench, backend, _ = _service()
    proposal = _proposal(workbench)
    receipt = workbench.commit(proposal.plan_id, pot_id=POT)
    backend.store.rows[0] = replace(backend.store.rows[0], **{field: value})
    report = workbench.verify_commit(proposal.plan_id, pot_id=POT)
    assert report.outcome == "mismatched" and not report.ok
    assert report.checked_revision == 1
    assert report.content_readback["mismatches"]
    assert (
        workbench.commit_status(proposal.plan_id, pot_id=POT).execution
        == receipt.execution
    )


@pytest.mark.parametrize("change", ["past", "future", "retired"])
def test_observed_closure_is_distinct_from_original_execution(change):
    workbench, backend, _ = _service()
    proposal = _proposal(workbench)
    receipt = workbench.commit(proposal.plan_id, pot_id=POT)
    patch = (
        {"retired": True}
        if change == "retired"
        else {
            "invalid_at": datetime(
                2025 if change == "past" else 2099, 2, 1, tzinfo=timezone.utc
            )
        }
    )
    backend.store.rows[0] = replace(backend.store.rows[0], **patch)
    report = workbench.verify_commit(proposal.plan_id, pot_id=POT)
    assert report.outcome == "superseded_retracted"
    assert (
        report.content_readback["claim_outcomes"][0]["lifecycle"]["state"]
        == {"past": "closed", "future": "scheduled_close", "retired": "retired"}[change]
    )
    assert (
        workbench.commit_status(proposal.plan_id, pot_id=POT).execution
        == receipt.execution
    )


def test_revision_movement_discards_partial_read_results(monkeypatch):
    workbench, backend, _ = _service()
    proposal = _proposal(workbench)
    receipt = workbench.commit(proposal.plan_id, pot_id=POT)
    original = backend.store.find_claims
    changed = False

    def read(_store, filter_):
        nonlocal changed
        rows = original(filter_)
        if filter_.record_id_in and not changed:
            changed = True
            backend.mutation.reset_pot(POT)
        return rows

    monkeypatch.setattr(type(backend.store), "find_claims", read)
    report = workbench.verify_commit(proposal.plan_id, pot_id=POT)
    assert report.state == "deferred" and report.outcome == "inconclusive"
    assert report.checked_revision is None and report.readback_count == 0
    assert report.content_readback == {} and report.subgraph_versions == {}
    assert backend.mutation.current_version(POT) == 2
    assert (
        workbench.commit_status(proposal.plan_id, pot_id=POT).execution
        == receipt.execution
    )


def test_unsupported_claim_query_is_not_repeated(monkeypatch):
    workbench, backend, _ = _service()
    proposal = _proposal(workbench)
    workbench.commit(proposal.plan_id, pot_id=POT)
    calls = []

    def unavailable(_store, filter_):
        calls.append(filter_)
        raise CapabilityNotImplemented("claim_query.find_claims")

    monkeypatch.setattr(type(backend.store), "find_claims", unavailable)
    result = workbench.commit(proposal.plan_id, pot_id=POT, verify=True)
    assert result.ok and result.verification_state == "unsupported"
    assert result.verification.outcome == "inconclusive"
    assert len(calls) == 1


def test_standalone_verification_recovers_receipt_before_workflow_finalize():
    backend, store, workbench, record = _prepared()
    receipt = _execute(backend, record)
    assert store.get(pot_id=POT, plan_id=record.plan_id).status != "committed"
    report = workbench.verify_commit(record.plan_id, pot_id=POT)
    assert report.outcome == "matching" and report.checked_revision == 1
    assert store.get(pot_id=POT, plan_id=record.plan_id).execution == receipt.execution
    assert backend.mutation.current_version(POT) == 1


def test_workflow_committed_without_receipt_is_not_execution_evidence():
    backend, store, workbench, record = _prepared()
    store.save(replace(record, status="committed"))
    report = workbench.verify_commit(record.plan_id, pot_id=POT)
    assert report.status == "not_committed" and report.outcome == "inconclusive"
    assert backend.mutation.current_version(POT) == 0


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_precommit_baseline_failure_and_cancellation(monkeypatch, error):
    workbench, backend, _ = _service()
    proposal = _proposal(workbench)

    def fail(*args, **kwargs):
        raise error()

    monkeypatch.setattr(workbench_service, "_verification_quality_snapshot", fail)
    if error is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            workbench.commit(proposal.plan_id, pot_id=POT, verify=True)
        assert backend.mutation.current_version(POT) == 0
    else:
        result = workbench.commit(proposal.plan_id, pot_id=POT, verify=True)
        assert result.ok and result.verification_state == "deferred"
        assert backend.mutation.current_version(POT) == 1


def test_active_loop_cancel_during_verification_keeps_receipt(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    workbench, backend, store = _service()
    runtime = build_graph_runtime(backend, store)
    proposal = _proposal(workbench)
    original = workbench_service._verify_ingestion_commit

    def block(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(workbench_service, "_verify_ingestion_commit", block)

    async def journey():
        task = asyncio.create_task(
            runtime.commit_async(proposal.plan_id, pot_id=POT, verify=True)
        )
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 3), 4)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            receipt = await runtime.commit_status_async(proposal.plan_id, pot_id=POT)
            assert receipt.ok and receipt.execution.version_after == 1
        finally:
            release.set()

    asyncio.run(journey())
    assert backend.mutation.current_version(POT) == 1


@pytest.mark.parametrize("version", [None, True, -1, 0, 1.5, "1"])
def test_invalid_revision_never_proves_coherence(monkeypatch, version):
    workbench, backend, _ = _service()
    proposal = _proposal(workbench)
    receipt = workbench.commit(proposal.plan_id, pot_id=POT)
    monkeypatch.setattr(
        type(backend.mutation), "current_version", lambda _self, _pot: version
    )
    report = workbench.verify_commit(proposal.plan_id, pot_id=POT)
    assert report.outcome == "inconclusive" and report.checked_revision is None
    assert (
        workbench.commit_status(proposal.plan_id, pot_id=POT).execution
        == receipt.execution
    )


def test_automatic_predecessor_scope_is_checked():
    from tests.unit.mutations.test_graph_workbench_plans import _owner_payload

    workbench, backend, store = _service()
    first_payload = _owner_payload("team:old")
    first_payload["operations"][0]["valid_from"] = "2025-01-01T00:00:00Z"
    first = workbench.propose(first_payload, pot_id=POT)
    assert workbench.commit(first.plan_id, pot_id=POT).ok
    old = backend.store.rows[0]
    second_payload = _owner_payload("team:new")
    second_payload["operations"][0]["valid_from"] = "2025-02-01T00:00:00Z"
    second = workbench.propose(second_payload, pot_id=POT)
    assert second.ok
    record = store.get(pot_id=POT, plan_id=second.plan_id)
    assert record.lowered_batch.edge_upserts[0].resolution.superseded_record_ids == (
        old.record_id,
    )
    assert workbench.commit(second.plan_id, pot_id=POT).ok
    backend.store.rows[:] = [
        (
            replace(row, source_ref="other:source")
            if row.record_id == old.record_id
            else row
        )
        for row in backend.store.rows
    ]
    report = workbench.verify_commit(second.plan_id, pot_id=POT)
    assert report.outcome == "mismatched"
    assert any(
        item["record_id"] == old.record_id and "source_ref" in item["fields"]
        for item in report.content_readback["mismatches"]
    )


def test_non_temporal_text_is_not_timestamp_canonicalized():
    workbench, backend, _ = _service()
    payload = _link_payload()
    payload["operations"][0]["description"] = "valueZ"
    proposal = workbench.propose(payload, pot_id=POT)
    assert workbench.commit(proposal.plan_id, pot_id=POT).ok
    backend.store.rows[0] = replace(backend.store.rows[0], description="value+00:00")
    assert workbench.verify_commit(proposal.plan_id, pot_id=POT).outcome == "mismatched"


@pytest.mark.parametrize(
    "field,value",
    [
        ("invalid_at", "not-a-date"),
        ("valid_until", "not-a-date"),
        ("observed_at", "2025-01-01"),
        ("valid_at", "2025-02-01T00:00:00Z"),
        ("valid_until", "2025-02-01T00:00:00.000000001Z"),
    ],
)
def test_malformed_stored_times_are_not_proof_of_matching(field, value):
    from potpie_context_engine.core.claim_codec import claim_properties, row_from_record

    workbench, backend, _ = _service()
    proposal = _proposal(workbench)
    committed = workbench.commit(proposal.plan_id, pot_id=POT)
    props = claim_properties(backend.store.rows[0])
    props[field] = value
    backend.store.rows[0] = row_from_record({"props": props})
    checked = workbench.verify_commit(proposal.plan_id, pot_id=POT)
    assert checked.state == "deferred" and checked.outcome == "inconclusive"
    outcome = checked.content_readback["claim_outcomes"][0]
    assert outcome["lifecycle"]["state"] == "unknown"
    assert field in outcome["lifecycle"]["temporal_diagnostics"]
    assert (
        workbench.commit_status(proposal.plan_id, pot_id=POT).execution
        == committed.execution
    )
    backend.store.rows[0] = replace(backend.store.rows[0], source_ref="corruption")
    assert workbench.verify_commit(proposal.plan_id, pot_id=POT).outcome == "mismatched"


@pytest.mark.parametrize(
    "field,value",
    [
        ("predicate", "WRONG"),
        ("subject_key", "service:other"),
        ("object_key", "service:other"),
    ],
)
def test_correction_verifies_its_known_frozen_endpoints(field, value):
    workbench, backend, _ = _service()
    first = _proposal(workbench)
    assert workbench.commit(first.plan_id, pot_id=POT).ok
    op = _link_payload()["operations"][0]
    op.update(
        op="end_relation_validity",
        reason="dependency removed",
        valid_until="2025-02-01T00:00:00Z",
    )
    proposal = workbench.propose(
        {"operations": [op]}, pot_id=POT, approved_by="reviewer"
    )
    assert proposal.ok
    assert workbench.commit(proposal.plan_id, pot_id=POT).ok
    backend.store.rows[0] = replace(backend.store.rows[0], **{field: value})
    checked = workbench.verify_commit(proposal.plan_id, pot_id=POT)
    assert checked.outcome == "mismatched"
    assert field in checked.content_readback["mismatches"][0]["fields"]


def test_legacy_report_does_not_invent_completed_verification():
    from potpie_context_engine.core.graph_plans import GraphIngestionVerificationResult

    decoded = result_from_payload(
        GraphIngestionVerificationResult,
        {"ok": True, "status": "ok", "plan_id": "legacy", "pot_id": POT},
    )
    assert decoded.ok and decoded.status == "ok"
    assert decoded.state == "deferred" and decoded.outcome == "inconclusive"
    assert decoded.checked_revision is None and decoded.verified_at is None


def test_empty_property_projection_requires_existing_entity():
    from potpie_context_engine.core.graph_mutations import EntityUpsert
    from potpie_context_engine.core.reconciliation import MutationBatch
    from potpie_context_engine.core.workbench_service import GraphWorkbenchService
    from tests.unit.mutations.test_graph_workbench_plans import _entity_only_payload

    workbench, backend, store = _service()
    initial = workbench.propose(_entity_only_payload(), pot_id=POT)
    assert workbench.commit(initial.plan_id, pot_id=POT).ok

    def projection(_request, plan):
        plan.batch = MutationBatch(
            entity_upserts=[
                EntityUpsert("feature:widget-cache", (), {}, preserve_labels=True)
            ]
        )

    workbench = GraphWorkbenchService(
        backend=backend, plan_store=store, lowerer=projection
    )
    proposed = workbench.propose(_entity_only_payload(), pot_id=POT)
    assert proposed.ok
    assert workbench.commit(proposed.plan_id, pot_id=POT).ok
    backend.mutation.reset_pot(POT)
    checked = workbench.verify_commit(proposed.plan_id, pot_id=POT)
    assert checked.outcome == "mismatched"
    assert (
        checked.content_readback["mismatches"][0]["fields"]["entity"]["actual"]
        == "missing"
    )


@pytest.mark.parametrize("defer", [False, True])
def test_reconciliation_race_preserves_requested_verification_intent(
    monkeypatch, defer
):
    backend, store, workbench, record = _prepared()
    receipt = _execute(backend, record)

    def other_reconciler(_store, *, expected, replacement):
        store.save(replacement)
        return False

    monkeypatch.setattr(type(store), "compare_and_set", other_reconciler)
    result = workbench.commit(
        record.plan_id, pot_id=POT, verify=True, defer_verification=defer
    )
    assert result.ok and result.execution == receipt.execution
    assert result.verification_state == "deferred"
    assert result.verification.outcome == "inconclusive"
    assert backend.mutation.current_version(POT) == 1
