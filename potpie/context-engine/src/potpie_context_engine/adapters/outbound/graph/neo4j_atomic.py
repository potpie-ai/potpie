"""Datastore-wide revision check, graph write and receipt in one Neo4j transaction."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

from potpie_context_core.definition import GraphDefinition
from potpie_context_core.errors import GraphMutationVersionConflict
from potpie_context_core.graph_mutations import ProvenanceContext
from potpie_context_core.reconciliation import MutationBatch, MutationResult
from potpie_context_core.reconciliation_config import ReconciliationConfig
from potpie_context_core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    CompletedMutationExecution,
    MutationExecutionReuseError,
    execution_receipts_from_json,
    execution_receipts_to_json,
    mutation_batch_fingerprint,
)
from potpie_context_engine.adapters.outbound.graph.apply_plan import (
    apply_mutation_batch,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter


_CONSTRAINT = "CREATE CONSTRAINT potpie_revision_unique IF NOT EXISTS FOR (r:PotpieRevision) REQUIRE r.pot_id IS UNIQUE"
_REVISION = "MATCH (r:PotpieRevision {pot_id: $pot}) RETURN r.version AS version"
_RECEIPT = "MATCH (r:PotpieMutationReceipt {pot_id: $pot, mutation_id: $mutation}) RETURN r.payload AS payload"
_LOCK_REVISION = (
    "MERGE (r:PotpieRevision {pot_id: $pot}) "
    "ON CREATE SET r.version = 0, r.lock_sequence = 0 "
    "SET r.lock_sequence = r.lock_sequence + 1 RETURN r.version AS version"
)


async def _one(session: Any, query: str, **parameters: Any) -> Mapping[str, Any] | None:
    result = await session.run(query, **parameters)
    record = await result.single()
    await result.consume()
    return record


async def _ensure_constraint(transaction: Any) -> None:
    result = await transaction.run(_CONSTRAINT)
    await result.consume()


async def current_version(writer: Neo4jGraphWriter, pot_id: str) -> int:
    driver = writer._new_driver()
    if driver is None:
        raise RuntimeError("neo4j_unavailable")
    try:
        async with driver.session() as session:
            record = await _one(session, _REVISION, pot=pot_id)
            return int(record["version"] or 0) if record else 0
    finally:
        await driver.close()


def _decode_receipt(
    record: Mapping[str, Any] | None, fingerprint: str
) -> CompletedMutationExecution | None:
    if record is None:
        return None
    receipts = execution_receipts_from_json(json.loads(record["payload"]))
    if len(receipts) != 1:
        raise RuntimeError("invalid durable Neo4j mutation receipt")
    receipt = receipts[0]
    if receipt.batch_fingerprint != fingerprint:
        raise MutationExecutionReuseError(
            "mutation ID was already used with a different batch"
        )
    return receipt


async def lookup_execution(
    writer: Neo4jGraphWriter,
    plan: MutationBatch,
    *,
    expected_pot_id: str,
    mutation_id: str,
) -> MutationExecutionLookup:
    fingerprint = mutation_batch_fingerprint(plan)
    driver = writer._new_driver()
    if driver is None:
        raise RuntimeError("neo4j_unavailable")
    try:
        async with driver.session() as session:
            record = await _one(
                session, _RECEIPT, pot=expected_pot_id, mutation=mutation_id
            )
        receipt = _decode_receipt(record, fingerprint)
        return MutationExecutionLookup(
            state=MutationExecutionState.completed.value
            if receipt
            else MutationExecutionState.absent.value,
            mutation_id=mutation_id,
            batch_fingerprint=fingerprint,
            result=receipt.result if receipt else None,
        )
    finally:
        await driver.close()


async def apply_atomic(
    writer: Neo4jGraphWriter,
    plan: MutationBatch,
    *,
    expected_pot_id: str,
    provenance_context: ProvenanceContext,
    definition: GraphDefinition,
    reconciliation_config: ReconciliationConfig | None = None,
    expected_version: int | None = None,
) -> MutationResult:
    """Write-lock the pot revision before checking it; roll back every verb on error."""
    fingerprint = mutation_batch_fingerprint(plan)
    mutation_id = provenance_context.mutation_id
    if not mutation_id:
        raise ValueError("atomic mutation requires a reserved mutation ID")
    driver = writer._new_driver()
    if driver is None:
        raise RuntimeError("neo4j_unavailable")
    try:
        async with driver.session() as session:
            # Concurrent first writers can deadlock while creating schema;
            # managed transactions retry that transient failure safely.
            await session.execute_write(_ensure_constraint)
            async with await session.begin_transaction() as transaction:
                # A read-dependent SET acquires Neo4j's node write lock before
                # reading its old value. Every canonical mutation takes it.
                revision = await _one(
                    transaction,
                    _LOCK_REVISION,
                    pot=expected_pot_id,
                )
                current = int(revision["version"])
                cached = _decode_receipt(
                    await _one(
                        transaction, _RECEIPT, pot=expected_pot_id, mutation=mutation_id
                    ),
                    fingerprint,
                )
                if cached:
                    await transaction.commit()
                    return cached.result
                if expected_version is not None and current != expected_version:
                    raise GraphMutationVersionConflict(
                        expected=expected_version, current=current
                    )
                applied = await apply_mutation_batch(
                    writer.in_transaction(transaction),
                    deepcopy(plan),
                    expected_pot_id=expected_pot_id,
                    provenance_context=provenance_context,
                    definition=definition,
                    reconciliation_config=reconciliation_config,
                )
                receipt = CompletedMutationExecution(
                    expected_pot_id, mutation_id, fingerprint, applied
                )
                payload = json.dumps(
                    execution_receipts_to_json([receipt]), sort_keys=True
                )
                result = await transaction.run(
                    "MATCH (r:PotpieRevision {pot_id: $pot}) SET r.version = r.version + 1 "
                    "CREATE (:PotpieMutationReceipt {pot_id: $pot, mutation_id: $mutation, payload: $payload})",
                    pot=expected_pot_id,
                    mutation=mutation_id,
                    payload=payload,
                )
                await result.consume()
                await transaction.commit()
                return applied
    finally:
        await driver.close()


async def reset_atomic(writer: Neo4jGraphWriter, pot_id: str) -> dict[str, Any]:
    """Reset canonical data while invalidating every outstanding proposal."""
    from potpie_context_engine.adapters.outbound.graph.cypher import (
        _require_valid_pot_id,
    )

    _require_valid_pot_id(pot_id)
    driver = writer._new_driver()
    if driver is None:
        raise RuntimeError("neo4j_unavailable")
    try:
        async with driver.session() as session:
            await session.execute_write(_ensure_constraint)
            async with await session.begin_transaction() as transaction:
                await _one(transaction, _LOCK_REVISION, pot=pot_id)
                before = await _one(
                    transaction,
                    "MATCH (n {group_id: $pot}) WITH collect(n) AS nodes "
                    "FOREACH (n IN nodes | DETACH DELETE n) RETURN size(nodes) AS count",
                    pot=pot_id,
                )
                result = await transaction.run(
                    "MATCH (r:PotpieRevision {pot_id: $pot}) SET r.version = r.version + 1",
                    pot=pot_id,
                )
                await result.consume()
                await transaction.commit()
                return {
                    "ok": True,
                    "group_id_nodes_before": before["count"],
                    "group_id_nodes_remaining": 0,
                }
    finally:
        await driver.close()
