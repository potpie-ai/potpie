"""Encode shared correction/assertion results as exact native physical writes.

Lifecycle selection, interval decisions and derived effects belong to the core
planner. Both ordinary publication and journal capture consume this one final
write set while WATCH protects its complete read bracket.
"""

from __future__ import annotations

from copy import deepcopy

from potpie_context_engine.core.claim_codec import claim_properties
from potpie_context_engine.core.claim_lifecycle import ClaimLifecycleConflict

from .cypher import _coerce_props_for_neo4j, _embedding_props


def encode_native_claims(
    graph, pipe, plan, params, provenance, embedder, original, rows
):
    """Validate native endpoints/identities and encode the planner's final rows."""
    _validate_endpoints(graph, pipe, plan, params["pot"])
    writes, counts = _encode_writes(plan, params, provenance, embedder, original, rows)
    _validate_write_identities(
        graph, pipe, params["pot"], writes, original, plan.entity_upserts
    )
    native_writes = deepcopy(writes)
    for item in native_writes:
        item["embedding"] = item["props"].pop("fact_embedding", None)
    params.update(
        claim_updates=[item for item in native_writes if item["record_id"] in original],
        claim_creates=[
            item for item in native_writes if item["record_id"] not in original
        ],
        claim_writes=writes,
        claim_affected=[item["record_id"] for item in writes],
        claim_edges=counts["edges"],
        claim_deletes=counts["deletes"],
        claim_invalidations=counts["invalidations"],
    )


def _validate_endpoints(graph, pipe, plan, pot_id):
    from .falkordb_journal import _read

    claim_keys = {
        key
        for edge in plan.edge_upserts
        for key in (edge.from_entity_key, edge.to_entity_key)
    }
    keys = claim_keys | {
        item.entity_key for item in plan.entity_upserts if item.preserve_labels
    }
    if not keys:
        return
    records = _read(
        graph,
        pipe,
        "MATCH (e:Entity {group_id:$pot}) WHERE e.entity_key IN $keys "
        "RETURN e.entity_key,coalesce(e.retired,false)=false",
        {"pot": pot_id, "keys": sorted(keys)},
    )
    if len(records) != len({key for key, _ in records}):
        raise ClaimLifecycleConflict("ambiguous native claim endpoint")
    activity = dict(records)
    present = set(activity)
    for item in plan.entity_upserts:
        if item.preserve_labels and item.entity_key not in present:
            raise ClaimLifecycleConflict(
                "entity projection target is missing or ambiguous"
            )
        present.add(item.entity_key)
        if "retired" in item.properties:
            retired = item.properties["retired"]
            activity[item.entity_key] = retired is None or retired is False
        else:
            activity.setdefault(item.entity_key, True)
    if any(not activity.get(key, False) for key in claim_keys):
        raise ClaimLifecycleConflict("claim endpoint is missing or retired")


def _encode_writes(plan, params, provenance, embedder, original, rows):
    annotations, assertions = {}, {}
    counts = {"edges": len(plan.edge_upserts), "deletes": 0, "invalidations": 0}
    for operations, count_key in (
        (plan.edge_deletes, "deletes"),
        (plan.invalidations, "invalidations"),
    ):
        for op in operations:
            for closure in op.closures:
                if closure.before == closure.after:
                    continue
                props = {"expired_at": params["now"], "last_mutation_id": params["mid"]}
                if count_key == "deletes":
                    props["deleted_by"] = params["mid"]
                else:
                    props.update(
                        invalidation_reason=op.reason,
                        invalidated_by=provenance.source_event_id,
                    )
                annotations.setdefault(closure.target.record_id, {}).update(props)
                counts[count_key] += 1
    for edge in plan.edge_upserts:
        assertions[edge.resolution.record_id] = edge
        for identity in edge.resolution.superseded_record_ids:
            annotations.setdefault(identity, {}).update(
                expired_at=params["now"], last_mutation_id=params["mid"]
            )
    by_id = {row.record_id: row for row in rows}
    writes = []
    for identity in sorted(annotations.keys() | assertions.keys()):
        row = by_id[identity]
        props = deepcopy(
            original.get(identity, {"uuid": identity, "created_at": params["now"]})
        )
        edge = assertions.get(identity)
        if edge is not None and not edge.resolution.metadata_review:
            props.update(
                {
                    key: value
                    for key, value in edge.properties.items()
                    if key not in {"uuid", "created_at"}
                }
            )
            props.update(
                _embedding_props(
                    embedder=embedder,
                    predicate=edge.edge_type,
                    from_key=edge.from_entity_key,
                    to_key=edge.to_entity_key,
                    edge_props=claim_properties(row),
                )
            )
        props.update(claim_properties(row))
        if edge is not None and not edge.resolution.metadata_review:
            props["provenance_source_event"] = provenance.source_event_id
        props.update(annotations.get(identity, {}))
        if edge is not None:
            props["last_mutation_id"] = params["mid"]
        writes.append(
            {
                "record_id": identity,
                "source": row.subject_key,
                "target": row.object_key,
                "props": _coerce_props_for_neo4j(props),
            }
        )
    return writes, counts


def _validate_write_identities(graph, pipe, pot_id, writes, existing, entity_upserts):
    """Validate claim cardinality and separation from final staged entities."""
    if not writes:
        return
    from .falkordb_journal import _read

    identities = [item["record_id"] for item in writes]
    params = {"pot": pot_id, "identities": identities}
    rows = _read(
        graph,
        pipe,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
        "WHERE coalesce(r.record_id,r.uuid) IN $identities "
        "RETURN coalesce(r.record_id,r.uuid),count(r)",
        params,
    )
    counts = dict(rows)
    if any(
        counts.get(identity, 0) != int(identity in existing) for identity in identities
    ):
        raise ClaimLifecycleConflict(
            "missing or ambiguous native claim record identity"
        )
    entity_rows = _read(
        graph,
        pipe,
        "MATCH (e:Entity {group_id:$pot}) "
        "WHERE e.record_id IN $identities OR e.uuid IN $identities OR e.entity_key IN $keys "
        "RETURN e.entity_key,properties(e)",
        {**params, "keys": list({item.entity_key for item in entity_upserts})},
    )
    staged = [(key, dict(props)) for key, props in entity_rows]
    for item in entity_upserts:
        matched = [props for key, props in staged if key == item.entity_key]
        if matched:
            for props in matched:
                props.update(item.properties)
        else:
            staged.append((item.entity_key, dict(item.properties)))
    if any(
        (props.get("record_id") or props.get("uuid")) in identities
        for _, props in staged
    ):
        raise ClaimLifecycleConflict(
            "claim incarnation collides with an entity identity"
        )


def publication_clauses():
    """Physical identity writes; no lifecycle selection occurs inside Cypher."""
    clauses = [
        "CALL { WITH v UNWIND $claim_updates AS item MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
        "WHERE coalesce(r.record_id,r.uuid)=item.record_id SET r+=item.props "
        "FOREACH (_ IN CASE WHEN item.embedding IS NULL THEN [] ELSE [1] END | SET r.fact_embedding=vecf32(item.embedding)) "
        "RETURN count(r) AS updated_claims }",
        "CALL { WITH v UNWIND $claim_creates AS item MATCH (a:Entity {group_id:$pot,entity_key:item.source}) "
        "MATCH (b:Entity {group_id:$pot,entity_key:item.target}) CREATE (a)-[r:RELATES_TO]->(b) "
        "SET r+=item.props "
        "FOREACH (_ IN CASE WHEN item.embedding IS NULL THEN [] ELSE [1] END | SET r.fact_embedding=vecf32(item.embedding)) "
        "RETURN count(r) AS created_claims }",
    ]
    return clauses
