"""Native claim persistence; lifecycle decisions belong to the shared core."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
import uuid

from potpie_context_engine.core.claim_codec import claim_properties, row_from_record
from potpie_context_engine.core.claim_lifecycle import (
    ClaimLifecycleConflict,
    close_claim_at,
    resolve_claim_assertion,
)
from potpie_context_engine.core.claim_preparation import assertion_row
from potpie_context_engine.core.definition import GraphDefinition
from potpie_context_engine.core.graph_mutations import EdgeUpsert, ProvenanceRef


async def write_claim(
    session: Any,
    *,
    pot_id: str,
    edge: EdgeUpsert,
    provenance: ProvenanceRef,
    definition: GraphDefinition,
    embedder: Any | None = None,
) -> None:
    # These are encoding/enrichment helpers, not lifecycle policy.
    from .cypher import _coerce_props_for_neo4j, _embedding_props, _stable_source_ref

    anchor = provenance.graph_updated_at or datetime.now(timezone.utc)
    props = dict(edge.properties)
    props.setdefault(
        "valid_at",
        props.get("valid_from")
        or provenance.event_occurred_at
        or provenance.valid_from
        or anchor,
    )
    if "confidence" not in props and provenance.confidence is not None:
        props["confidence"] = provenance.confidence
    incoming = assertion_row(
        replace(edge, properties=props),
        pot_id=pot_id,
        anchor=anchor,
        record_id=edge.resolution.record_id if edge.resolution else str(uuid.uuid4()),
        source_system=provenance.source_system,
        source_ref=_stable_source_ref(
            predicate=edge.edge_type,
            from_key=edge.from_entity_key,
            to_key=edge.to_entity_key,
            provenance=provenance,
        ),
        stable_source=bool(provenance.source_system and provenance.source_ref),
    )
    incoming = replace(incoming, mutation_id=provenance.mutation_id)
    identities = [incoming.record_id]
    if edge.resolution:
        identities.extend(edge.resolution.superseded_record_ids)
    if edge.metadata_review_record_id:
        identities.append(edge.metadata_review_record_id)
    result = await session.run(
        "MATCH ()-[r:RELATES_TO {group_id: $gid}]->() "
        "WHERE (r.subject_key=$subject AND r.name=$predicate) "
        "OR coalesce(r.record_id,r.uuid) IN $identities RETURN properties(r) AS props",
        gid=pot_id,
        subject=edge.from_entity_key,
        predicate=edge.edge_type,
        identities=identities,
    )
    candidates = [row_from_record(record) async for record in result]
    row, resolution = resolve_claim_assertion(
        incoming,
        candidates,
        singleton=edge.edge_type in definition.singleton_predicates,
        anchor=anchor,
        resolution=edge.resolution,
        metadata_record_id=edge.metadata_review_record_id,
    )
    expected = {key: 1 for key in resolution.superseded_record_ids}
    expected[resolution.record_id] = int(resolution.existing)
    result = await session.run(
        "MATCH (n:Entity {group_id:$gid}) "
        "WHERE coalesce(n.record_id,n.uuid) IN $identities "
        "RETURN coalesce(n.record_id,n.uuid) AS identity, count(n) AS cnt, 'entity' AS kind "
        "UNION ALL MATCH ()-[r:RELATES_TO {group_id:$gid}]->() "
        "WHERE coalesce(r.record_id,r.uuid) IN $identities "
        "RETURN coalesce(r.record_id,r.uuid) AS identity, count(r) AS cnt, 'claim' AS kind",
        gid=pot_id,
        identities=list(expected),
    )
    actual = {key: 0 for key in expected}
    async for record in result:
        if record["kind"] == "entity":
            raise ClaimLifecycleConflict(
                "claim incarnation collides with an entity identity"
            )
        actual[record["identity"]] = int(record["cnt"])
    if actual != expected:
        raise ClaimLifecycleConflict("claim incarnation is missing or ambiguous")
    props = {
        **({} if resolution.metadata_review else edge.properties),
        **claim_properties(row),
    }
    props.pop("uuid", None)
    props.pop("created_at", None)
    props.update(
        provenance_source_event=provenance.source_event_id,
        last_mutation_id=provenance.mutation_id,
    )
    if not resolution.metadata_review:
        props.update(
            _embedding_props(
                embedder=embedder,
                predicate=edge.edge_type,
                from_key=edge.from_entity_key,
                to_key=edge.to_entity_key,
                edge_props=props,
            )
        )
    if resolution.existing:
        query = (
            "MATCH (a:Entity {group_id:$gid,entity_key:$subject})"
            "-[r:RELATES_TO {group_id:$gid}]->"
            "(b:Entity {group_id:$gid,entity_key:$object}) "
            "WHERE coalesce(r.record_id,r.uuid)=$record_id "
            "AND coalesce(a.retired,false)=false AND coalesce(b.retired,false)=false "
            "SET r += $props RETURN count(r) AS cnt"
        )
    else:
        query = (
            "MATCH (a:Entity {group_id:$gid,entity_key:$subject}) "
            "MATCH (b:Entity {group_id:$gid,entity_key:$object}) "
            "WHERE coalesce(a.retired,false)=false AND coalesce(b.retired,false)=false "
            "CREATE (a)-[r:RELATES_TO]->(b) "
            "SET r += $props, r.uuid=$record_id, r.created_at=$now RETURN count(r) AS cnt"
        )
    result = await session.run(
        query,
        gid=pot_id,
        subject=edge.from_entity_key,
        object=edge.to_entity_key,
        record_id=resolution.record_id,
        props=_coerce_props_for_neo4j(props),
        now=anchor.isoformat(),
    )
    record = await result.single()
    await result.consume()
    if record is None or int(record["cnt"]) != 1:
        raise ClaimLifecycleConflict(
            "claim endpoints or incarnation are missing or ambiguous"
        )
    if resolution.superseded_record_ids:
        result = await session.run(
            "MATCH ()-[r:RELATES_TO {group_id:$gid}]->() "
            "WHERE coalesce(r.record_id,r.uuid) IN $identities "
            "SET r.invalid_at=$effective_at, r.expired_at=$now, "
            "r.superseded_by_object=$object, r.supersession_reason='singleton_predicate', "
            "r.last_mutation_id=$mutation_id RETURN count(r) AS cnt",
            gid=pot_id,
            identities=list(resolution.superseded_record_ids),
            effective_at=resolution.effective_at.isoformat(),
            now=anchor.isoformat(),
            object=edge.to_entity_key,
            mutation_id=provenance.mutation_id,
        )
        record = await result.single()
        await result.consume()
        if record is None or int(record["cnt"]) != len(
            resolution.superseded_record_ids
        ):
            raise ClaimLifecycleConflict("singleton predecessor identity changed")


async def close_claims(
    session: Any,
    *,
    pot_id: str,
    predicate: str | None = None,
    subject: str | None = None,
    object_key: str | None = None,
    claim_keys: tuple[str, ...] | None = None,
    record_ids: tuple[str, ...] | None = None,
    point: datetime,
    anchor: datetime,
    properties: dict[str, Any],
) -> int:
    """Resolve a legacy selector or honor a frozen one, preserving past endpoints."""
    result = await session.run(
        "MATCH ()-[r:RELATES_TO {group_id:$gid}]->() "
        "WHERE ($predicate IS NULL OR r.name=$predicate) "
        "AND ($subject IS NULL OR r.subject_key=$subject) "
        "AND ($object IS NULL OR r.object_key=$object) "
        "AND ($claim_keys IS NULL OR r.claim_key IN $claim_keys) "
        "AND ($record_ids IS NULL OR coalesce(r.record_id,r.uuid) IN $record_ids) "
        "RETURN properties(r) AS props",
        gid=pot_id,
        predicate=predicate,
        subject=subject,
        object=object_key,
        claim_keys=list(claim_keys) if claim_keys is not None else None,
        record_ids=list(record_ids) if record_ids is not None else None,
    )
    rows = [row_from_record(record) async for record in result]
    if record_ids is not None and sorted(row.record_id for row in rows) != sorted(
        record_ids
    ):
        raise ClaimLifecycleConflict("correction incarnation is missing or ambiguous")
    selected = []
    for row in rows:
        closed = close_claim_at(row, point, anchor=anchor)
        if closed != row:
            if not row.record_id or row.record_id in selected:
                raise ClaimLifecycleConflict(
                    "closure incarnation identity is missing or ambiguous"
                )
            selected.append(row.record_id)
    if not selected:
        return 0
    result = await session.run(
        "MATCH ()-[r:RELATES_TO {group_id:$gid}]->() "
        "WHERE coalesce(r.record_id,r.uuid) IN $record_ids "
        "SET r += $props RETURN count(r) AS cnt",
        gid=pot_id,
        record_ids=selected,
        props={**properties, "invalid_at": point.isoformat()},
    )
    record = await result.single()
    await result.consume()
    if record is None or int(record["cnt"]) != len(selected):
        raise ClaimLifecycleConflict("closure incarnation identity changed")
    return len(selected)
