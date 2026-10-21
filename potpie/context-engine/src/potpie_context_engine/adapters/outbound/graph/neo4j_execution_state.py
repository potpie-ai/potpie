"""Validate Neo4j revisions against the authoritative execution receipts.

Receipt metadata lives in canonical JSON payloads. Each validation scans the
pot's receipts so even older modern receipts establish their revision floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any, Mapping

from potpie_context_engine.core.mutation_execution import MutationPersistenceError

from ._local_json_atomic import strict_json_load
from ._mutation_execution import (
    CompletedMutationExecution,
    MutationExecutionReuseError,
    execution_receipts_from_json,
)

_REVISION = "MATCH (r:PotpieRevision {pot_id: $pot}) RETURN r.version AS version"
_RECEIPTS = (
    "MATCH (r:PotpieMutationReceipt {pot_id: $pot}) "
    "RETURN r.mutation_id AS mutation_id, r.payload AS payload"
)
_LOCK_REVISION = (
    "MERGE (r:PotpieRevision {pot_id: $pot}) "
    "ON CREATE SET r.version = 0, r.lock_sequence = 0 "
    "SET r.lock_sequence = r.lock_sequence + 1 RETURN r.version AS version"
)


@dataclass(frozen=True, slots=True)
class Neo4jExecutionState:
    version: int
    receipts: Mapping[str, CompletedMutationExecution]

    def completed(
        self, mutation_id: str, fingerprint: str
    ) -> CompletedMutationExecution | None:
        receipt = self.receipts.get(mutation_id)
        if receipt is not None and receipt.batch_fingerprint != fingerprint:
            raise MutationExecutionReuseError(
                "mutation ID was already used with a different batch"
            )
        return receipt


async def _one(session: Any, query: str, **parameters: Any) -> Mapping[str, Any] | None:
    result = await session.run(query, **parameters)
    records = await result.fetch(2)
    await result.consume()
    if len(records) > 1:
        raise MutationPersistenceError("duplicate durable Neo4j execution state")
    return records[0] if records else None


def _decode_receipt(
    record: Mapping[str, Any], pot_id: str
) -> CompletedMutationExecution:
    try:
        receipts = execution_receipts_from_json(
            strict_json_load(StringIO(record["payload"]))
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MutationPersistenceError(
            "invalid durable Neo4j mutation receipt"
        ) from exc
    if len(receipts) != 1:
        raise MutationPersistenceError("invalid durable Neo4j mutation receipt")
    receipt = receipts[0]
    if receipt.pot_id != pot_id or receipt.mutation_id != record.get("mutation_id"):
        raise MutationPersistenceError(
            "Neo4j mutation receipt identity differs from its key"
        )
    return receipt


async def _read_receipts(
    transaction: Any, pot_id: str
) -> dict[str, CompletedMutationExecution]:
    result = await transaction.run(_RECEIPTS, pot=pot_id)
    rows = await result.data()
    await result.consume()
    receipts = {}
    for row in rows:
        receipt = _decode_receipt(row, pot_id)
        if receipt.mutation_id in receipts:
            raise MutationPersistenceError("duplicate durable Neo4j execution state")
        receipts[receipt.mutation_id] = receipt
    return receipts


def _validated_state(
    revision: Mapping[str, Any] | None,
    receipts: Mapping[str, CompletedMutationExecution],
    *,
    revision_existed: bool,
) -> Neo4jExecutionState:
    current = revision["version"] if revision is not None else 0
    if type(current) is not int or current < 0:
        raise MutationPersistenceError("invalid durable Neo4j pot revision")
    # None means genuinely unavailable legacy metadata. Modern empty receipts
    # establish a zero floor and still prove a revision record must exist.
    floor = max(
        (
            item.result.execution.version_after
            for item in receipts.values()
            if item.result.execution is not None
        ),
        default=None,
    )
    if floor is not None and (not revision_existed or current < floor):
        raise MutationPersistenceError(
            "Neo4j pot revision is missing or below durable execution receipt floor"
        )
    return Neo4jExecutionState(current, receipts)


async def read_execution_state(transaction: Any, pot_id: str) -> Neo4jExecutionState:
    # Canonical revisions only advance. Reading receipts first, then the
    # revision, allows a concurrent atomic commit without pairing its new
    # receipt with an earlier revision under Neo4j's read-committed isolation.
    receipts = await _read_receipts(transaction, pot_id)
    revision = await _one(transaction, _REVISION, pot=pot_id)
    return _validated_state(revision, receipts, revision_existed=revision is not None)


async def lock_execution_state(transaction: Any, pot_id: str) -> Neo4jExecutionState:
    result = await transaction.run(_LOCK_REVISION, pot=pot_id)
    records = await result.fetch(2)
    summary = await result.consume()
    created = summary.counters.nodes_created
    if len(records) != 1 or type(created) is not int or created not in (0, 1):
        raise MutationPersistenceError("invalid durable Neo4j revision lock result")
    receipts = await _read_receipts(transaction, pot_id)
    # MERGE and the read-dependent SET own the native lock before this scan.
    # A concurrent first writer consequently sees the committed node as
    # existing; a missing node with modern receipts is rolled back, never
    # silently bootstrapped over acknowledged history.
    return _validated_state(records[0], receipts, revision_existed=created == 0)
