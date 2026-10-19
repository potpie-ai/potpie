"""Resolve scoped native claim effects once, inside the watched boundary.

The shared core owns incarnation and interval policy. This module only reads
native candidates, enriches assertions, and encodes exact physical writes used
by ordinary publication and journal capture alike.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import uuid

from potpie_context_engine.core.claim_codec import (
    claim_properties,
    iso,
    parse_dt,
    row_from_record,
)
from potpie_context_engine.core.claim_lifecycle import (
    ClaimLifecycleConflict,
    apply_claim_to_rows,
    close_claim_at,
    resolution_properties,
    resolve_claim_assertion,
)
from potpie_context_engine.core.claim_preparation import assertion_row
from potpie_context_engine.core.graph_mutations import EdgeUpsert
from potpie_context_engine.core.mutation_execution import MutationPersistenceError

from .cypher import _coerce_props_for_neo4j, _embedding_props, _stable_source_ref


def normalize_assertions(plan, params, provenance, embedder):
    """Enrich authored properties without mistaking generated refs for sources."""
    assertions = []
    anchor = datetime.fromisoformat(params["now"])
    params["delete_effective_at"] = iso(
        provenance.event_occurred_at or provenance.valid_from or anchor
    )
    for index, edge in enumerate(plan.edge_upserts):
        props = dict(edge.properties)
        props.setdefault(
            "valid_at", props.get("valid_from") or provenance.valid_from or anchor
        )
        if "confidence" not in props and provenance.confidence is not None:
            props["confidence"] = provenance.confidence
        incoming = assertion_row(
            replace(edge, properties=props),
            pot_id=params["pot"],
            anchor=anchor,
            record_id=edge.resolution.record_id
            if edge.resolution
            else str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"potpie:claim:{params['pot']}:{params['mid']}:{index}",
                )
            ),
            source_system=provenance.source_system,
            source_ref=_stable_source_ref(
                predicate=edge.edge_type,
                from_key=edge.from_entity_key,
                to_key=edge.to_entity_key,
                provenance=provenance,
            ),
            stable_source=bool(provenance.source_system and provenance.source_ref),
        )
        incoming = replace(incoming, mutation_id=params["mid"])
        raw = {
            key: value
            for key, value in edge.properties.items()
            if key not in {"uuid", "created_at"}
        }
        raw.update(claim_properties(incoming))
        if edge.metadata_review_record_id is None and not (
            edge.resolution and edge.resolution.metadata_review
        ):
            raw.update(
                _embedding_props(
                    embedder=embedder,
                    predicate=edge.edge_type,
                    from_key=edge.from_entity_key,
                    to_key=edge.to_entity_key,
                    edge_props=raw,
                )
            )
        raw["provenance_source_event"] = provenance.source_event_id
        raw["last_mutation_id"] = params["mid"]
        assertions.append((edge, incoming, raw))
    for index, item in enumerate(plan.invalidations):
        params[f"ip{index}"] = {
            "invalid_at": item.valid_to or params["now"],
            "expired_at": params["now"],
            "invalidation_reason": item.reason,
            "invalidated_by": provenance.source_event_id,
            "last_mutation_id": params["mid"],
        }
        old_key = item.target_entity_key or (
            item.target_edge[2] if item.target_edge else None
        )
        if item.superseded_by_key and old_key:
            edge = EdgeUpsert(
                "SUPERSEDES",
                item.superseded_by_key,
                old_key,
                {
                    "fact": item.reason,
                    "reason": item.reason,
                    "valid_at": item.valid_to or params["now"],
                    "source_identity_stable": False,
                },
            )
            incoming = assertion_row(
                edge,
                pot_id=params["pot"],
                anchor=anchor,
                record_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"potpie:supersedes:{params['pot']}:{params['mid']}:{index}",
                    )
                ),
                source_system=provenance.source_system,
                source_ref=_stable_source_ref(
                    predicate="SUPERSEDES",
                    from_key=edge.from_entity_key,
                    to_key=edge.to_entity_key,
                    provenance=provenance,
                ),
            )
            incoming = replace(incoming, mutation_id=params["mid"])
            assertions.append((edge, incoming, claim_properties(incoming)))
    return assertions


def resolve_native_effects(graph, pipe, plan, params, definition, assertions):
    """Read a subject/predicate closure and explicit IDs under the caller's WATCH."""
    from .falkordb_journal import _read

    selectors = []
    identities = set()
    candidate_params = {"pot": params["pot"]}
    for index, (edge, incoming, _) in enumerate(assertions):
        candidate_params.update(
            {
                f"subject{index}": edge.from_entity_key,
                f"predicate{index}": edge.edge_type,
            }
        )
        selectors.append(
            f"(r.subject_key=$subject{index} AND r.name=$predicate{index})"
        )
        identities.add(incoming.record_id)
        if edge.resolution:
            identities.update(edge.resolution.superseded_record_ids)
        if edge.metadata_review_record_id:
            identities.add(edge.metadata_review_record_id)
    for index, edge in enumerate(plan.edge_deletes):
        candidate_params.update(
            {
                f"ds{index}": edge.from_entity_key,
                f"dp{index}": edge.edge_type,
                f"do{index}": edge.to_entity_key,
            }
        )
        selectors.append(
            f"(r.subject_key=$ds{index} AND r.name=$dp{index} AND r.object_key=$do{index})"
        )
    for index, item in enumerate(plan.invalidations):
        if item.target_claim_record_ids is not None:
            identities.update(item.target_claim_record_ids)
        elif item.target_claim_keys is not None:
            candidate_params[f"keys{index}"] = list(item.target_claim_keys)
            selectors.append(f"r.claim_key IN $keys{index}")
        elif item.target_edge:
            predicate, subject, object_key = item.target_edge
            candidate_params.update(
                {
                    f"ip{index}": predicate,
                    f"is{index}": subject,
                    f"io{index}": object_key,
                }
            )
            selectors.append(
                f"(r.name=$ip{index} AND r.subject_key=$is{index} AND r.object_key=$io{index})"
            )
    candidate_params["identities"] = sorted(identities)
    selectors.append("coalesce(r.record_id,r.uuid) IN $identities")
    records = _read(
        graph,
        pipe,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() WHERE "
        + " OR ".join(selectors)
        + " RETURN properties(r)",
        candidate_params,
    )
    props_by_id = {}
    for (props,) in records:
        identity = (
            props.get("record_id")
            if props.get("record_id") is not None
            else props.get("uuid")
        )
        if not isinstance(identity, str) or not identity:
            raise MutationPersistenceError(
                "migration required: touched legacy claim has no valid persistent record identity"
            )
        if identity in props_by_id:
            raise ClaimLifecycleConflict("ambiguous native claim record identity")
        props_by_id[identity] = props
    endpoint_keys = {
        key
        for edge, _, _ in assertions
        for key in (edge.from_entity_key, edge.to_entity_key)
    }
    endpoint_rows = (
        _read(
            graph,
            pipe,
            "MATCH (e:Entity {group_id:$pot}) WHERE e.entity_key IN $keys "
            "RETURN e.entity_key,coalesce(e.retired,false)=false",
            {"pot": params["pot"], "keys": sorted(endpoint_keys)},
        )
        if endpoint_keys
        else []
    )
    existing_keys = [key for key, _ in endpoint_rows]
    if len(existing_keys) != len(set(existing_keys)):
        raise ClaimLifecycleConflict("ambiguous native claim endpoint")
    activity = dict(endpoint_rows)
    for item in plan.entity_upserts:
        if "retired" in item.properties:
            retired = item.properties["retired"]
            activity[item.entity_key] = retired is None or retired is False
        else:
            activity.setdefault(item.entity_key, True)
    entity_keys = {key for key, active in activity.items() if active}
    writes, counts = resolve_effects(
        plan, params, definition, assertions, props_by_id, entity_keys
    )
    _validate_write_identities(
        graph, pipe, params["pot"], writes, props_by_id, plan.entity_upserts
    )
    native_writes = deepcopy(writes)
    for item in native_writes:
        item["embedding"] = item["props"].pop("fact_embedding", None)
    params.update(
        claim_updates=[
            item for item in native_writes if item["record_id"] in props_by_id
        ],
        claim_creates=[
            item for item in native_writes if item["record_id"] not in props_by_id
        ],
        claim_writes=writes,
        claim_affected=[item["record_id"] for item in writes],
        claim_edges=counts["edges"],
        claim_deletes=counts["deletes"],
        claim_invalidations=counts["invalidations"],
    )


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
                if props.get("record_id") is None:
                    props["record_id"] = props.get("uuid")
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


def resolve_effects(plan, params, definition, assertions, props_by_id, entity_keys):
    """Encode shared-policy decisions into exact writes, staging in batch order."""
    rows = [row_from_record({"props": props}) for props in props_by_id.values()]
    final = deepcopy(props_by_id)
    touched = set()
    counts = {"edges": 0, "deletes": 0, "invalidations": 0}
    anchor = datetime.fromisoformat(params["now"])
    for index, (edge, incoming, enriched) in enumerate(assertions):
        if (
            edge.from_entity_key not in entity_keys
            or edge.to_entity_key not in entity_keys
        ):
            raise ClaimLifecycleConflict("claim endpoint is missing or retired")
        row, resolution = resolve_claim_assertion(
            incoming,
            rows,
            singleton=edge.edge_type in definition.singleton_predicates,
            anchor=anchor,
            resolution=edge.resolution,
            metadata_record_id=edge.metadata_review_record_id,
        )
        old = final.get(row.record_id)
        props = (
            dict(old) if old else {"uuid": row.record_id, "created_at": params["now"]}
        )
        if not resolution.metadata_review:
            props.update(enriched)
        props.update(claim_properties(row))
        props.update(resolution_properties(resolution))
        props["last_mutation_id"] = params["mid"]
        final[row.record_id] = props
        touched.add(row.record_id)
        rows = apply_claim_to_rows(rows, row, resolution)
        for identity in resolution.superseded_record_ids:
            previous = next(
                candidate for candidate in rows if candidate.record_id == identity
            )
            final[identity].update(claim_properties(previous))
            final[identity]["expired_at"] = params["now"]
            touched.add(identity)
        if index < len(plan.edge_upserts):
            counts["edges"] += 1
    for edge in plan.edge_deletes:
        selected = [
            row
            for row in rows
            if (row.predicate, row.subject_key, row.object_key)
            == (edge.edge_type, edge.from_entity_key, edge.to_entity_key)
        ]
        rows, changed = _close(
            rows,
            selected,
            parse_dt(params["delete_effective_at"]),
            anchor,
            {"expired_at": params["now"], "deleted_by": params["mid"]},
            final,
            touched,
        )
        counts["deletes"] += changed
    for index, item in enumerate(plan.invalidations):
        if item.target_claim_record_ids is not None:
            selected = [
                row for row in rows if row.record_id in item.target_claim_record_ids
            ]
            if sorted(row.record_id for row in selected) != sorted(
                item.target_claim_record_ids
            ):
                raise ClaimLifecycleConflict(
                    "invalidation incarnation is missing or ambiguous"
                )
        elif item.target_claim_keys is not None:
            selected = [row for row in rows if row.claim_key in item.target_claim_keys]
        elif item.target_edge:
            selected = [
                row
                for row in rows
                if (row.predicate, row.subject_key, row.object_key) == item.target_edge
            ]
        else:
            selected = []
        rows, changed = _close(
            rows,
            selected,
            parse_dt(item.valid_to) or anchor,
            anchor,
            params[f"ip{index}"],
            final,
            touched,
        )
        counts["invalidations"] += changed
    writes = [
        {
            "record_id": identity,
            "source": final[identity]["subject_key"],
            "target": final[identity]["object_key"],
            "props": _coerce_props_for_neo4j(final[identity]),
        }
        for identity in sorted(touched)
    ]
    return writes, counts


def _close(rows, selected, point, anchor, annotations, final, touched):
    changed = 0
    for row in selected:
        closed = close_claim_at(row, point, anchor=anchor)
        if closed == row:
            continue
        final[row.record_id].update(
            annotations, record_id=row.record_id, invalid_at=iso(closed.invalid_at)
        )
        rows = [
            closed if current.record_id == row.record_id else current
            for current in rows
        ]
        touched.add(row.record_id)
        changed += 1
    return rows, changed


def publication_clauses(plan):
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
    for index, item in enumerate(plan.invalidations):
        if item.target_entity_key:
            clauses.append(
                f"CALL {{ WITH v OPTIONAL MATCH (e:Entity {{group_id:$pot,entity_key:$iek{index}}}) "
                f"FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END | SET e += $ip{index}) "
                f"RETURN count(e) AS ic{index} }}"
            )
    return clauses
