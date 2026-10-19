"""Datastore-wide revision check, graph write and receipt in one Neo4j transaction."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from potpie_context_engine.core.definition import GraphDefinition
from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.reconciliation import MutationBatch, MutationResult
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig
from potpie_context_engine.core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    CompletedMutationExecution,
    execution_receipts_to_json,
    mutation_batch_fingerprint,
    complete_execution,
    validate_execution_replay,
    validate_prepared_execution,
    validate_new_preparation,
)
from potpie_context_engine.adapters.outbound.graph.apply_plan import (
    apply_mutation_batch,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter
from potpie_context_engine.adapters.outbound.graph.neo4j_execution_state import (
    _one,
    lock_execution_state,
    read_execution_state,
)


_CONSTRAINTS = (
    "CREATE CONSTRAINT potpie_revision_unique IF NOT EXISTS "
    "FOR (r:PotpieRevision) REQUIRE r.pot_id IS UNIQUE",
    "CREATE CONSTRAINT potpie_receipt_unique IF NOT EXISTS "
    "FOR (r:PotpieMutationReceipt) REQUIRE (r.pot_id, r.mutation_id) IS UNIQUE",
)
_AFFECTED_CLAIMS = (
    "MATCH ()-[r:RELATES_TO {group_id: $pot, last_mutation_id: $mutation}]->() "
    "RETURN collect(coalesce(r.record_id, r.uuid)) AS record_ids, count(r) AS count"
)


async def _ensure_constraints(transaction: Any) -> None:
    for query in _CONSTRAINTS:
        result = await transaction.run(query)
        await result.consume()


async def current_version(writer: Neo4jGraphWriter, pot_id: str) -> int:
    from potpie_context_engine.adapters.outbound.graph.cypher import (
        _require_valid_pot_id,
    )

    _require_valid_pot_id(pot_id)
    driver = writer._new_driver()
    if driver is None:
        raise RuntimeError("neo4j_unavailable")
    try:
        async with driver.session() as session:
            # Writer routing keeps these authoritative reads off lagging replicas.
            state = await session.execute_write(read_execution_state, pot_id)
            return state.version
    finally:
        await driver.close()


async def lookup_execution(
    writer: Neo4jGraphWriter,
    plan: MutationBatch,
    *,
    expected_pot_id: str,
    mutation_id: str,
) -> MutationExecutionLookup:
    from potpie_context_engine.adapters.outbound.graph.cypher import (
        _require_valid_pot_id,
    )

    _require_valid_pot_id(expected_pot_id)
    if not mutation_id:
        raise ValueError("execution lookup requires a reserved mutation ID")
    fingerprint = mutation_batch_fingerprint(plan)
    driver = writer._new_driver()
    if driver is None:
        raise RuntimeError("neo4j_unavailable")
    try:
        async with driver.session() as session:
            state = await session.execute_write(read_execution_state, expected_pot_id)
        receipt = state.completed(mutation_id, fingerprint)
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
    from potpie_context_engine.adapters.outbound.graph.cypher import (
        _require_valid_pot_id,
    )

    _require_valid_pot_id(expected_pot_id)
    plan = deepcopy(plan)
    if plan.reset_pot or plan.snapshot_import is not None:
        from potpie_context_engine.core.reconciliation_validation import (
            validate_reconciliation_plan,
        )

        validate_reconciliation_plan(
            plan, expected_pot_id, definition=definition, config=reconciliation_config
        )
    mutation_id = provenance_context.mutation_id
    if not mutation_id:
        raise ValueError("atomic mutation requires a reserved mutation ID")
    validate_prepared_execution(
        plan,
        pot_id=expected_pot_id,
        mutation_id=mutation_id,
        provenance=provenance_context,
    )
    fingerprint = mutation_batch_fingerprint(plan)
    driver = writer._new_driver()
    if driver is None:
        raise RuntimeError("neo4j_unavailable")
    try:
        async with driver.session() as session:
            # Concurrent first writers can deadlock while creating schema;
            # managed transactions retry that transient failure safely.
            await session.execute_write(_ensure_constraints)

            async def execute(transaction: Any) -> MutationResult:
                state = await lock_execution_state(transaction, expected_pot_id)
                current = state.version
                cached = state.completed(mutation_id, fingerprint)
                if cached:
                    validate_execution_replay(
                        cached.result, provenance=provenance_context
                    )
                    return cached.result
                validate_new_preparation(provenance_context)
                if expected_version is not None and current != expected_version:
                    raise GraphMutationVersionConflict(
                        expected=expected_version, current=current
                    )
                prepared = provenance_context.prepared
                if prepared is not None and current != prepared.expected_version:
                    raise GraphMutationVersionConflict(
                        expected=prepared.expected_version, current=current
                    )
                if plan.reset_pot or plan.snapshot_import is not None:
                    from .neo4j_snapshot_write import merge_snapshot, reset_graph

                    if plan.reset_pot:
                        applied, record_ids = await reset_graph(
                            transaction, pot_id=expected_pot_id, mutation_id=mutation_id
                        )
                    else:
                        applied, record_ids = await merge_snapshot(
                            transaction,
                            pot_id=expected_pot_id,
                            mutation_id=mutation_id,
                            incoming=plan.snapshot_import,
                        )
                else:
                    applied = await apply_mutation_batch(
                        writer.in_transaction(transaction),
                        deepcopy(plan),
                        expected_pot_id=expected_pot_id,
                        provenance_context=provenance_context,
                        definition=definition,
                        reconciliation_config=reconciliation_config,
                    )
                    affected = await _one(
                        transaction,
                        _AFFECTED_CLAIMS,
                        pot=expected_pot_id,
                        mutation=mutation_id,
                    )
                    record_ids = affected["record_ids"] if affected is not None else []
                    if (
                        affected is None
                        or len(record_ids) != affected["count"]
                        or any(
                            not isinstance(value, str) or not value
                            for value in record_ids
                        )
                    ):
                        raise MutationPersistenceError(
                            "affected Neo4j claim has no record identity"
                        )
                applied = complete_execution(
                    applied,
                    plan,
                    pot_id=expected_pot_id,
                    version_before=current,
                    provenance=provenance_context,
                    record_ids=record_ids,
                )
                receipt = CompletedMutationExecution(
                    expected_pot_id, mutation_id, fingerprint, applied
                )
                payload = json.dumps(
                    execution_receipts_to_json([receipt]), sort_keys=True
                )
                result = await transaction.run(
                    "MATCH (r:PotpieRevision {pot_id: $pot}) SET r.version = $version "
                    "CREATE (:PotpieMutationReceipt {pot_id: $pot, mutation_id: $mutation, payload: $payload})",
                    pot=expected_pot_id,
                    mutation=mutation_id,
                    payload=payload,
                    version=applied.execution.version_after,
                )
                await result.consume()
                return applied

            return await session.execute_write(execute)
    finally:
        await driver.close()
