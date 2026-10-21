"""Native claim persistence; lifecycle decisions belong to the shared core."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
import uuid

from potpie_context_engine.core.claim_codec import claim_properties, row_from_record
from potpie_context_engine.core.claim_corrections import (
    ClaimClosureResolution,
    apply_claim_closure,
)
from potpie_context_engine.core.claim_lifecycle import (
    ClaimLifecycleConflict,
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
    raw_candidates = [dict(record) async for record in result]
    candidates = [row_from_record(record) for record in raw_candidates]
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
    _require_physical_ids(
        raw_candidates, {key for key, count in expected.items() if count}
    )
    await _validate_claim_identities(session, pot_id, expected)
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


async def _validate_claim_identities(
    session: Any, pot_id: str, expected: dict[str, int]
) -> None:
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


async def close_claims(
    session: Any,
    *,
    pot_id: str,
    closures: tuple[ClaimClosureResolution, ...] | None,
    properties: dict[str, Any],
) -> int:
    """Persist exact reviewed intervals; the shared planner owns all selection."""
    if closures is None:
        raise ClaimLifecycleConflict(
            "closure must be resolved inside the atomic mutation boundary"
        )
    if not closures:
        return 0
    identities = [closure.target.record_id for closure in closures]
    await _validate_claim_identities(session, pot_id, dict.fromkeys(identities, 1))
    result = await session.run(
        "MATCH ()-[r:RELATES_TO {group_id:$gid}]->() "
        "WHERE coalesce(r.record_id,r.uuid) IN $record_ids RETURN properties(r) AS props",
        gid=pot_id,
        record_ids=identities,
    )
    raw_rows = [dict(record) async for record in result]
    _require_physical_ids(raw_rows, set(identities))
    rows = [row_from_record(record) for record in raw_rows]
    if sorted(row.record_id for row in rows) != sorted(identities):
        raise ClaimLifecycleConflict("correction incarnation is missing or ambiguous")
    by_id = {row.record_id: row for row in rows}
    count = 0
    for closure in closures:
        apply_claim_closure(by_id[closure.target.record_id], closure)
        if closure.before == closure.after:
            continue
        result = await session.run(
            "MATCH ()-[r:RELATES_TO {group_id:$gid}]->() "
            "WHERE coalesce(r.record_id,r.uuid)=$record_id "
            "SET r += $props RETURN count(r) AS cnt",
            gid=pot_id,
            record_id=closure.target.record_id,
            props={**properties, **closure.after.to_dict()},
        )
        record = await result.single()
        await result.consume()
        if record is None or int(record["cnt"]) != 1:
            raise ClaimLifecycleConflict("closure incarnation identity changed")
        count += 1
    return count


def _require_physical_ids(records, selected: set[str]) -> None:
    for record in records:
        props = record["props"]
        if (props.get("record_id") or props.get("uuid")) in selected:
            if (
                not isinstance(props.get("record_id"), str)
                or not props["record_id"].strip()
            ):
                raise ClaimLifecycleConflict(
                    "migration_required: selected claim has no physical record_id"
                )
