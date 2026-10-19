"""Normalize assertion inputs and freeze physical lifecycle choices."""

from __future__ import annotations

from datetime import datetime
from dataclasses import replace
import hashlib
import uuid

from .claim_codec import _as_utc, iso, row_from_record, parse_dt
from .claim_corrections import (
    SupersessionBinding,
    prepare_claim_closures,
    bind_supersession_references,
)
from .claim_lifecycle import (
    ClaimLifecycleConflict,
    apply_claim_to_rows,
    resolve_claim_assertion,
    validate_incarnation_history,
    claim_scope,
)
from .definition import GraphDefinition
from .graph_mutations import EdgeUpsert, ProvenanceRef
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
    if not (batch.edge_upserts or batch.edge_deletes or batch.invalidations):
        return
    rows = query.find_claims(
        ClaimQueryFilter(
            pot_id=pot_id,
            include_invalidated=True,
            include_retired=True,
        )
    )
    bindings: list[SupersessionBinding] = []
    rows = prepare_claim_closures(
        batch,
        rows,
        anchor=anchor,
        bindings=bindings,
        singleton_predicates=definition.singleton_predicates,
    )
    prepare_assertions_from_rows(
        batch,
        rows,
        pot_id=pot_id,
        mutation_id=mutation_id,
        anchor=anchor,
        definition=definition,
        bindings=bindings,
    )


def stable_source_ref(
    *, predicate: str, from_key: str, to_key: str, provenance: ProvenanceRef
) -> str:
    """Preserve the historical raw-source identity encoding."""
    if provenance.source_ref:
        return provenance.source_ref
    digest = hashlib.sha256(
        "\0".join((provenance.source_event_id, predicate, from_key, to_key)).encode()
    ).hexdigest()
    return f"event:{provenance.source_event_id}:{digest[:12]}"


def prepare_assertions_from_rows(
    batch: MutationBatch,
    rows: list[ClaimRow],
    *,
    pot_id: str,
    mutation_id: str,
    anchor: datetime,
    definition: GraphDefinition,
    bindings: list[SupersessionBinding],
    provenance: ProvenanceRef | None = None,
) -> list[ClaimRow]:
    """Freeze raw assertions or validate prepared ones over post-closure history."""
    for index, edge in enumerate(batch.edge_upserts):
        props = dict(edge.properties)
        source_ref = None
        if provenance is not None:
            props.setdefault(
                "valid_at",
                props.get("valid_from")
                or provenance.event_occurred_at
                or provenance.valid_from
                or anchor,
            )
            props.setdefault("observed_at", provenance.event_received_at or anchor)
            if provenance.confidence is not None:
                props.setdefault("confidence", provenance.confidence)
            source_ref = stable_source_ref(
                predicate=edge.edge_type,
                from_key=edge.from_entity_key,
                to_key=edge.to_entity_key,
                provenance=provenance,
            )
        incoming = assertion_row(
            replace(edge, properties=props),
            pot_id=pot_id,
            anchor=anchor,
            record_id=(
                edge.resolution.record_id
                if edge.resolution
                else str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"potpie:claim:{pot_id}:{mutation_id}:{index}",
                    )
                )
            ),
            source_system=provenance.source_system if provenance else None,
            source_ref=source_ref,
            stable_source=bool(
                provenance and provenance.source_system and provenance.source_ref
            ),
        )
        row, resolution = resolve_claim_assertion(
            replace(incoming, mutation_id=mutation_id),
            rows,
            singleton=edge.edge_type in definition.singleton_predicates,
            anchor=anchor,
            resolution=edge.resolution,
            metadata_record_id=edge.metadata_review_record_id,
        )
        edge.resolution = resolution
        edge.properties = {
            **props,
            "valid_at": iso(incoming.valid_at),
            "observed_at": iso(incoming.observed_at),
            "source_system": incoming.source_system,
            "source_ref": incoming.source_ref,
            "source_identity_stable": incoming.properties["source_identity_stable"],
        }
        rows = apply_claim_to_rows(rows, row, resolution)
    bind_supersession_references(bindings)
    identities = {edge.resolution.record_id for edge in batch.edge_upserts}
    identities.update(
        identity
        for edge in batch.edge_upserts
        for identity in edge.resolution.superseded_record_ids
    )
    identities.update(
        c.target.record_id
        for op in (*batch.edge_deletes, *batch.invalidations)
        for c in (op.closures or ())
    )
    scopes = {claim_scope(row) for row in rows if row.record_id in identities}
    validate_incarnation_history(
        (row for row in rows if claim_scope(row) in scopes and not row.retired),
        singleton_predicates=definition.singleton_predicates,
    )
    return rows
