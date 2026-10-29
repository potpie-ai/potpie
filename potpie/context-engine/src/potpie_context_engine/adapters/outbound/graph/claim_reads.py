"""Bounded claim read sets for raw writer lifecycle preparation."""

from __future__ import annotations

from typing import Any, Iterable

from potpie_context_engine.core.graph_mutations import EdgeDelete, InvalidationOp
from potpie_context_engine.core.reconciliation import MutationBatch


def candidate_scope(batch: MutationBatch) -> dict[str, Any]:
    """Select reviewed identities, assertion histories and unresolved corrections."""
    identities = set()
    corrections = []
    for edge in batch.edge_upserts:
        if edge.resolution is not None:
            identities.add(edge.resolution.record_id)
            identities.update(edge.resolution.superseded_record_ids)
        if edge.metadata_review_record_id is not None:
            identities.add(edge.metadata_review_record_id)
    for op in (*batch.edge_deletes, *batch.invalidations):
        if op.closures is not None:
            identities.update(closure.target.record_id for closure in op.closures)
        else:
            corrections.append(op)
    return {
        "record_ids": tuple(sorted(identities)),
        "history_scopes": tuple(
            sorted(
                {(edge.from_entity_key, edge.edge_type) for edge in batch.edge_upserts}
            )
        ),
        "corrections": tuple(corrections),
    }


def claim_read_query(
    pot_id: str,
    *,
    record_ids: Iterable[str] = (),
    history_scopes: Iterable[tuple[str, str]] = (),
    corrections: Iterable[EdgeDelete | InvalidationOp] = (),
) -> tuple[str, dict[str, Any]] | None:
    """Build a parameterized read with no fallback to an unbounded pot scan.

    All incarnations, including closed/retired rows, are returned in the selected
    scopes. Correction axes retain their intersection semantics.
    """
    params: dict[str, Any] = {"pot": pot_id}
    selectors = []
    for index, (subject, predicate) in enumerate(sorted(set(history_scopes))):
        params.update({f"subject{index}": subject, f"predicate{index}": predicate})
        selectors.append(
            f"(r.subject_key=$subject{index} AND r.name=$predicate{index})"
        )
    identities = sorted(set(record_ids))
    if identities:
        params["identities"] = identities
        selectors.append("coalesce(r.record_id,r.uuid) IN $identities")
    for index, op in enumerate(corrections):
        if isinstance(op, EdgeDelete):
            edge = (op.edge_type, op.from_entity_key, op.to_entity_key)
        else:
            edge = op.target_edge
        axes = []
        if edge is not None:
            predicate, subject, target = edge
            params.update(
                {f"p{index}": predicate, f"s{index}": subject, f"o{index}": target}
            )
            axes.append(
                f"r.name=$p{index} AND r.subject_key=$s{index} AND r.object_key=$o{index}"
            )
        if isinstance(op, InvalidationOp):
            if op.target_entity_key is not None:
                params[f"e{index}"] = op.target_entity_key
                axes.append(f"(r.subject_key=$e{index} OR r.object_key=$e{index})")
            if op.target_claim_keys is not None:
                params[f"k{index}"] = list(op.target_claim_keys)
                axes.append(f"r.claim_key IN $k{index}")
            if op.target_claim_record_ids is not None:
                params[f"r{index}"] = list(op.target_claim_record_ids)
                axes.append(f"coalesce(r.record_id,r.uuid) IN $r{index}")
        if axes:
            selectors.append("(" + " AND ".join(axes) + ")")
    if not selectors:
        return None
    return (
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() WHERE "
        + " OR ".join(selectors)
        + " RETURN properties(r) AS props",
        params,
    )
