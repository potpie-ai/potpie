"""Read and validate native revision state against durable execution receipts."""

from __future__ import annotations

from dataclasses import dataclass

from falkordb import QueryResult
from redis.exceptions import ResponseError

from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.reconciliation import MutationResult

from ._mutation_execution import MutationExecutionReuseError
from .falkordb_receipts import RECEIPT_COLUMNS, decode_receipt

_SNAPSHOT = (
    "OPTIONAL MATCH (v:PotpieRevision {pot_id:$pot}) "
    "WITH count(v) AS revision_count,collect(v.version) AS versions "
    "OPTIONAL MATCH (r:PotpieMutationReceipt {pot_id:$pot}) "
    "RETURN revision_count,versions,collect(CASE WHEN r IS NULL THEN null "
    f"ELSE [r.mutation_id,{RECEIPT_COLUMNS}] END)"
)


@dataclass(frozen=True, slots=True)
class NativeRevision:
    version: int | None
    receipts: dict[str, tuple[str, MutationResult]]

    def execution(self, mutation_id: str, fingerprint: str) -> MutationResult | None:
        receipt = self.receipts.get(mutation_id)
        if receipt is None:
            return None
        stored, result = receipt
        if stored != fingerprint:
            raise MutationExecutionReuseError(
                "mutation ID was already used with a different batch"
            )
        return result


def read_revision(graph, pot_id: str, pipe=None) -> NativeRevision:
    """One native query prevents mixing receipt and revision generations."""
    try:
        if pipe is None:
            rows = graph.ro_query(_SNAPSHOT, params={"pot": pot_id}).result_set
        else:
            response = pipe.execute_command(
                "GRAPH.RO_QUERY",
                graph.name,
                graph._build_params_header({"pot": pot_id}) + _SNAPSHOT,
                "--compact",
            )
            rows = QueryResult(graph, response).result_set
    except ResponseError as exc:
        if "empty key" not in str(exc).lower():
            raise
        return NativeRevision(None, {})
    if len(rows) != 1:
        raise MutationPersistenceError("incomplete native revision snapshot")
    count, versions, stored = rows[0]
    if count > 1:
        raise MutationPersistenceError("duplicate native pot revision")
    if len(versions) != count or any(
        type(value) is not int or value < 0 for value in versions
    ):
        raise MutationPersistenceError("invalid native pot revision")
    version = versions[0] if count else None
    receipts, floor = {}, None
    for mutation_id, *row in stored:
        fingerprint = row[0]
        if (
            not isinstance(mutation_id, str)
            or not mutation_id
            or not isinstance(fingerprint, str)
            or not fingerprint
        ):
            raise MutationPersistenceError("incomplete native receipt identity")
        if mutation_id in receipts:
            raise MutationPersistenceError("duplicate native execution receipt")
        result = decode_receipt(
            row, pot_id=pot_id, mutation_id=mutation_id, fingerprint=fingerprint
        )
        receipts[mutation_id] = (fingerprint, result)
        if result.execution is not None:
            after = result.execution.version_after
            floor = after if floor is None else max(floor, after)
    if floor is not None and (version is None or version < floor):
        raise MutationPersistenceError(
            f"native pot revision is missing or regressed below durable execution floor {floor}"
        )
    return NativeRevision(version, receipts)
