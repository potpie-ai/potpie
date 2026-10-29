"""Frozen request identity and receipt schema regression contract (POT-2700)."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    CompletedMutationExecution,
    MutationExecutionReuseError,
    complete_execution,
    execution_receipts_from_json,
    execution_receipts_to_json,
    validate_execution_replay,
)
from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.graph_plans import GraphMutationPlanRecord
from potpie_context_engine.core.mutation_execution import (
    ExecutionMetadata,
    MutationPersistenceError,
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
)
from potpie_context_engine.testing import build_test_graph_runtime


def prepared():
    return PreparedMutation(
        "pot",
        "mutation",
        "request",
        batch_fingerprint(MutationBatch()),
        0,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def result():
    return MutationResult(
        ok=True, mutation_id="mutation", mutation_summary=MutationSummary()
    )


def test_legacy_fingerprint_and_receipt_are_preserved():
    digest = batch_fingerprint(MutationBatch())
    assert digest == "3cd0190c5a605bf2af818ce7222c9813f6ee51f2"
    receipt = CompletedMutationExecution("pot", "mutation", digest, result())
    serialized = execution_receipts_to_json([receipt])
    assert "execution" not in serialized[0]["result"]
    assert execution_receipts_from_json(serialized) == (receipt,)
    assert execution_receipts_from_json(serialized)[0].result.execution is None


def test_explicit_null_execution_metadata_is_corruption_not_legacy():
    receipt = CompletedMutationExecution(
        "pot", "mutation", batch_fingerprint(MutationBatch()), result()
    )
    serialized = execution_receipts_to_json([receipt])
    serialized[0]["result"]["execution"] = None
    with pytest.raises(MutationPersistenceError, match="malformed execution receipt"):
        execution_receipts_from_json(serialized)


def test_new_receipt_freezes_execution_metadata_and_rejects_rebased_preparation():
    context = ProvenanceContext(mutation_id="mutation", prepared=prepared())
    completed = complete_execution(
        result(),
        MutationBatch(),
        pot_id="pot",
        version_before=0,
        provenance=context,
        applied_at=context.prepared.prepared_at,
    )
    assert completed.execution.version_after == 0
    assert (
        ExecutionMetadata.from_dict(completed.execution.to_dict())
        == completed.execution
    )
    with pytest.raises(GraphMutationVersionConflict):
        complete_execution(
            result(),
            MutationBatch(),
            pot_id="pot",
            version_before=1,
            provenance=context,
        )
    with pytest.raises(MutationExecutionReuseError):
        validate_execution_replay(
            completed,
            provenance=replace(
                context, prepared=replace(prepared(), request_fingerprint="changed")
            ),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("prepared_at", "2026-01-01"),
        ("expected_version", True),
        ("target_claim_record_ids", "abc"),
        ("request_fingerprint", {}),
        ("schema_version", 99),
        ("schema_version", 4.0),
        ("schema_version", True),
        ("fingerprint_version", True),
        ("fingerprint_version", 1.0),
    ],
)
def test_malformed_prepared_metadata_fails_closed(field, value):
    raw = prepared().to_dict()
    raw[field] = value
    with pytest.raises(MutationPersistenceError):
        PreparedMutation.from_dict(raw)


@pytest.mark.parametrize(
    "patch",
    [
        {"ok": "false"},
        {"mutation_id": 3},
        {"mutation_summary": {"edge_upserts_applied": "1"}},
    ],
)
def test_malformed_receipt_cannot_become_success(patch):
    raw = execution_receipts_to_json(
        [CompletedMutationExecution("pot", "mutation", "digest", result())]
    )
    raw[0]["result"].update(patch)
    with pytest.raises(MutationPersistenceError):
        execution_receipts_from_json(raw)


def test_concurrent_proposals_freeze_one_preparation_and_retain_terminal_identity():
    runtime = build_test_graph_runtime()
    payload = {
        "idempotency_key": "same",
        "operations": [
            {
                "op": "upsert_entity",
                "subject": {"key": "service:api", "type": "Service"},
            }
        ],
    }
    with ThreadPoolExecutor(max_workers=8) as pool:
        proposals = list(
            pool.map(lambda _: runtime.propose(payload, pot_id="pot"), range(8))
        )
    assert len({p.plan_id for p in proposals}) == 1
    record = runtime.plan_store.get(pot_id="pot", plan_id=proposals[0].plan_id)
    assert record.prepared.mutation_id == record.mutation_id
    assert GraphMutationPlanRecord.from_dict(record.to_dict()) == record
    for status in ("conflict", "expired", "error"):
        runtime.plan_store.save(replace(record, status=status))
        replay = runtime.propose(payload, pot_id="pot")
        assert replay.plan_id == record.plan_id and not replay.ok
        assert (
            runtime.plan_store.get(pot_id="pot", plan_id=record.plan_id).prepared
            == record.prepared
        )


def test_changed_frozen_plan_is_rejected_before_execution():
    runtime = build_test_graph_runtime()
    proposal = runtime.propose(
        {
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {"key": "service:api", "type": "Service"},
                }
            ]
        },
        pot_id="pot",
    )
    record = runtime.plan_store.get(pot_id="pot", plan_id=proposal.plan_id)
    runtime.plan_store.save(replace(record, mutation_id="tampered"))
    with pytest.raises(MutationPersistenceError):
        runtime.commit(proposal.plan_id, pot_id="pot")
    assert runtime.backend.mutation.current_version("pot") == 0


def test_mutated_administrative_batch_is_rejected_before_apply() -> None:
    from potpie_context_engine.core.reconciliation_validation import (
        validate_reconciliation_plan,
    )
    from potpie_context_engine.core.errors import MutationBatchValidationError
    from potpie_context_engine.core.reconciliation import MutationBatch
    from potpie_context_engine.core.graph_mutations import EntityUpsert

    batch = MutationBatch(reset_pot=True)
    batch.entity_upserts.append(EntityUpsert("service:a", ("Entity", "Service"), {}))
    with pytest.raises(MutationBatchValidationError, match="cannot share"):
        validate_reconciliation_plan(batch, "p")


def test_raw_snapshot_batch_requires_canonical_frozen_identities() -> None:
    from potpie_context_engine.core.reconciliation_validation import (
        validate_reconciliation_plan,
    )
    from potpie_context_engine.core.errors import MutationBatchValidationError
    from potpie_context_engine.core.reconciliation import MutationBatch

    batch = MutationBatch(
        snapshot_import={
            "format_version": "2",
            "pot_id": "p",
            "claims": [],
            "entities": [{"key": "service:a", "labels": ["Entity"], "properties": {}}],
        }
    )
    with pytest.raises(MutationBatchValidationError, match="canonical"):
        validate_reconciliation_plan(batch, "p")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2.0),
        ("fingerprint_version", True),
        ("fingerprint_version", 1.0),
    ],
)
def test_receipt_versions_require_exact_integers(field, value):
    execution = ExecutionMetadata("pot", datetime.now(timezone.utc), 0, 1, "digest")
    raw = execution.to_dict()
    raw[field] = value
    with pytest.raises(MutationPersistenceError, match="version"):
        ExecutionMetadata.from_dict(raw)


@pytest.mark.parametrize(
    "field,value", [("schema_version", 4.0), ("fingerprint_version", True)]
)
def test_corrupt_persisted_prepared_version_cannot_execute(tmp_path, field, value):
    import json
    from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
        LocalJsonGraphPlanStore,
    )
    from potpie_context_engine.testing import InMemoryGraphBackend
    from potpie_context_engine.core.runtime import build_graph_runtime

    store = LocalJsonGraphPlanStore(tmp_path)
    backend = InMemoryGraphBackend()
    runtime = build_graph_runtime(backend=backend, plan_store=store)
    proposal = runtime.propose(
        {
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {"key": "service:a", "type": "Service"},
                }
            ]
        },
        pot_id="pot",
    )
    raw = json.loads(store._path.read_text())
    raw["plans"]["pot"][proposal.plan_id]["prepared"][field] = value
    store._path.write_text(json.dumps(raw))
    before = store._path.read_bytes()
    with pytest.raises(MutationPersistenceError):
        runtime.commit(proposal.plan_id, pot_id="pot")
    assert backend.mutation.current_version("pot") == 0
    assert store._path.read_bytes() == before
