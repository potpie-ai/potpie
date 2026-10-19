"""Normalize assertion inputs and freeze physical lifecycle choices."""

from __future__ import annotations

from datetime import datetime
import uuid

from .claim_codec import _as_utc, iso, row_from_record, parse_dt
from .claim_lifecycle import (
    ClaimLifecycleConflict,
    apply_claim_to_rows,
    resolve_claim_assertion,
)
from .definition import GraphDefinition
from .graph_mutations import EdgeUpsert
from .ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow
from .reconciliation import MutationBatch


def assertion_row(
    edge: EdgeUpsert,
    *,
    pot_id: str,
    record_id: str,
    anchor: datetime,
    source_system: str | None = None,
    source_ref: str | None = None,
    stable_source: bool = False,
) -> ClaimRow:
    """Normalize once before policy; generated provenance never implies stability."""
    props = dict(edge.properties)
    for name in ("valid_at", "valid_from", "invalid_at", "valid_until", "observed_at"):
        if props.get(name) is not None and parse_dt(props[name]) is None:
            raise ClaimLifecycleConflict(f"invalid assertion timestamp: {name}")
    props.setdefault(
        "source_identity_stable",
        bool(stable_source or props.get("source_ref") and props.get("source_system")),
    )
    props.setdefault("source_system", source_system or "agent")
    props.setdefault("source_ref", source_ref)
    props.setdefault(
        "fact", f"{edge.from_entity_key} {edge.edge_type} {edge.to_entity_key}"
    )
    props.setdefault("valid_at", props.get("valid_from") or iso(anchor))
    props.setdefault("observed_at", iso(anchor))
    props.update(
        group_id=pot_id,
        name=edge.edge_type,
        subject_key=edge.from_entity_key,
        object_key=edge.to_entity_key,
        record_id=record_id,
    )
    row = row_from_record({"props": props})
    from dataclasses import replace

    return replace(
        row,
        **{
            name: (
                _as_utc(getattr(row, name)) if getattr(row, name) is not None else None
            )
            for name in ("valid_at", "invalid_at", "valid_until", "observed_at")
        },
    )


def prepare_claim_assertions(
    batch: MutationBatch,
    *,
    pot_id: str,
    mutation_id: str,
    anchor: datetime,
    query: ClaimQueryPort,
    definition: GraphDefinition,
) -> None:
    """Freeze candidates within the caller's revision-checked read bracket."""
    if not batch.edge_upserts:
        return
    rows = query.find_claims(
        ClaimQueryFilter(
            pot_id=pot_id,
            predicate_in=tuple(
                dict.fromkeys(edge.edge_type for edge in batch.edge_upserts)
            ),
            subject_key_in=tuple(
                dict.fromkeys(edge.from_entity_key for edge in batch.edge_upserts)
            ),
            include_invalidated=True,
            include_retired=True,
        )
    )
    for index, edge in enumerate(batch.edge_upserts):
        incoming = assertion_row(
            edge,
            pot_id=pot_id,
            anchor=anchor,
            record_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL, f"potpie:claim:{pot_id}:{mutation_id}:{index}"
                )
            ),
        )
        row, resolution = resolve_claim_assertion(
            incoming,
            rows,
            singleton=edge.edge_type in definition.singleton_predicates,
            anchor=anchor,
            metadata_record_id=edge.metadata_review_record_id,
        )
        edge.resolution = resolution
        # Resolution owns lifecycle; observation and provenance are frozen here.
        edge.properties["observed_at"] = iso(incoming.observed_at)
        edge.properties["source_identity_stable"] = incoming.properties[
            "source_identity_stable"
        ]
        rows = apply_claim_to_rows(rows, row, resolution)
