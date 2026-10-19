"""Fail closed on legacy claim identities before a watched native publication."""

from __future__ import annotations

from potpie_context_engine.core.mutation_execution import MutationPersistenceError


def validate_touched_claim_identities(graph, pipe, plan, params, definition):
    """Use the compiler's resolved parameters; never assign migration identities."""
    selectors = []
    for i, item in enumerate(plan.edge_upserts):
        selectors.append(
            f"a.group_id=$pot AND b.group_id=$pot AND a.entity_key=$ef{i} AND b.entity_key=$et{i} "
            f"AND r.name=$en{i} AND r.subject_key=$ef{i} AND r.object_key=$et{i} AND r.source_ref=$es{i}"
            + (f" AND r.claim_key=$eclaim{i}" if params[f"eclaim{i}"] else "")
        )
        if (
            item.edge_type in definition.singleton_predicates
            and params[f"eprop{i}"].get("evidence_strength") == "deterministic"
        ):
            selectors.append(
                f"a.group_id=$pot AND a.entity_key=$ef{i} AND r.name=$en{i} "
                f"AND b.entity_key<>$et{i} AND r.invalid_at IS NULL"
            )
    for i, _ in enumerate(plan.edge_deletes):
        selectors.append(
            f"a.group_id=$pot AND a.entity_key=$df{i} AND r.name=$dn{i} AND r.object_key=$dt{i}"
        )
    for i, item in enumerate(plan.invalidations):
        if item.target_claim_keys is not None:
            selectors.append(f"r.claim_key IN $ick{i} AND r.invalid_at IS NULL")
        elif not item.target_entity_key:
            selectors.append(
                f"a.group_id=$pot AND a.entity_key=$if{i} AND r.name=$in{i} "
                f"AND r.object_key=$it{i} AND r.invalid_at IS NULL"
            )
        if f"iss{i}" in params:
            selectors.append(
                f"a.group_id=$pot AND b.group_id=$pot AND a.entity_key=$isn{i} AND b.entity_key=$iso{i} "
                f"AND r.name='SUPERSEDES' AND r.subject_key=$isn{i} "
                f"AND r.object_key=$iso{i} AND r.source_ref=$iss{i}"
            )
    if not selectors:
        return
    from .falkordb_atomic import _rows

    query = (
        "MATCH (a:Entity)-[r:RELATES_TO {group_id:$pot}]->(b) WHERE "
        + " OR ".join(f"({selector})" for selector in selectors)
        + " RETURN r.record_id,r.uuid"
    )
    rows = _rows(
        graph,
        pipe.execute_command(
            "GRAPH.RO_QUERY",
            graph.name,
            graph._build_params_header(params) + query,
            "--compact",
        ),
    )
    for record_id, uid in rows:
        identity = record_id if record_id is not None else uid
        if not isinstance(identity, str) or not identity:
            raise MutationPersistenceError(
                "migration required: touched legacy claim has no valid persistent record identity"
            )
