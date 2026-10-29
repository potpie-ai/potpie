"""Scoped claim histories for the native correction/assertion planner."""

from __future__ import annotations

from potpie_context_engine.core.claim_codec import row_from_record
from potpie_context_engine.core.claim_corrections import (
    SupersessionBinding,
    prepare_claim_closures,
)
from potpie_context_engine.core.claim_lifecycle import ClaimLifecycleConflict
from potpie_context_engine.core.mutation_execution import MutationPersistenceError


def prepare_native_claims(
    graph, pipe, plan, *, pot_id, mutation_id, provenance, definition
):
    """Resolve a private batch over one WATCH-protected scoped read set."""
    from potpie_context_engine.core.claim_preparation import (
        prepare_assertions_from_rows,
    )

    props = _read_candidates(graph, pipe, plan, pot_id)
    props.update(_read_histories(graph, pipe, pot_id, _scopes(props)))
    rows = [row_from_record({"props": value}) for value in props.values()]
    bindings: list[SupersessionBinding] = []
    rows = prepare_claim_closures(
        plan,
        rows,
        anchor=provenance.graph_updated_at,
        effective_at=provenance.event_occurred_at or provenance.valid_from,
        bindings=bindings,
        singleton_predicates=definition.singleton_predicates,
    )
    # Raw closure planning can add ordinary system assertions. Read their
    # histories too, retaining the already staged closure intervals.
    scopes = {(edge.from_entity_key, edge.edge_type) for edge in plan.edge_upserts}
    extra = _read_histories(graph, pipe, pot_id, scopes - _scopes(props))
    rows.extend(
        row_from_record({"props": value})
        for key, value in extra.items()
        if key not in props
    )
    props.update(extra)
    rows = prepare_assertions_from_rows(
        plan,
        rows,
        pot_id=pot_id,
        mutation_id=mutation_id,
        anchor=provenance.graph_updated_at,
        definition=definition,
        bindings=bindings,
        provenance=provenance,
    )
    return props, rows


def _scopes(props):
    return {(value.get("subject_key"), value.get("name")) for value in props.values()}


def _read_candidates(graph, pipe, plan, pot_id):
    selectors, identities, params = [], set(), {"pot": pot_id}
    for index, edge in enumerate(plan.edge_upserts):
        params.update(
            {
                f"subject{index}": edge.from_entity_key,
                f"predicate{index}": edge.edge_type,
            }
        )
        selectors.append(
            f"(r.subject_key=$subject{index} AND r.name=$predicate{index})"
        )
        if edge.resolution:
            identities.add(edge.resolution.record_id)
            identities.update(edge.resolution.superseded_record_ids)
        if edge.metadata_review_record_id:
            identities.add(edge.metadata_review_record_id)
    for index, op in enumerate((*plan.edge_deletes, *plan.invalidations)):
        if op.closures is not None:
            identities.update(closure.target.record_id for closure in op.closures)
            continue
        from potpie_context_engine.core.graph_mutations import EdgeDelete

        if isinstance(op, EdgeDelete):
            params.update(
                {f"dp{index}": op.edge_type, f"ds{index}": op.from_entity_key}
            )
            selectors.append(f"(r.name=$dp{index} AND r.subject_key=$ds{index})")
            continue
        axes = []
        if op.target_edge is not None:
            predicate, subject, target = op.target_edge
            params.update(
                {f"ip{index}": predicate, f"is{index}": subject, f"io{index}": target}
            )
            axes.append(
                f"r.name=$ip{index} AND r.subject_key=$is{index} AND r.object_key=$io{index}"
            )
        if op.target_entity_key is not None:
            params[f"ie{index}"] = op.target_entity_key
            axes.append(f"(r.subject_key=$ie{index} OR r.object_key=$ie{index})")
        if op.target_claim_keys is not None:
            params[f"ik{index}"] = list(op.target_claim_keys)
            axes.append(f"r.claim_key IN $ik{index}")
        if op.target_claim_record_ids is not None:
            params[f"ir{index}"] = list(op.target_claim_record_ids)
            axes.append(f"coalesce(r.record_id,r.uuid) IN $ir{index}")
        if axes:
            selectors.append("(" + " AND ".join(axes) + ")")
    if identities:
        params["identities"] = sorted(identities)
        selectors.append("coalesce(r.record_id,r.uuid) IN $identities")
    return _read(graph, pipe, params, selectors)


def _read_histories(graph, pipe, pot_id, scopes):
    params, selectors = {"pot": pot_id}, []
    for index, (subject, predicate) in enumerate(sorted(scopes)):
        params.update({f"subject{index}": subject, f"predicate{index}": predicate})
        selectors.append(
            f"(r.subject_key=$subject{index} AND r.name=$predicate{index})"
        )
    return _read(graph, pipe, params, selectors)


def _read(graph, pipe, params, selectors):
    if not selectors:
        return {}
    from .falkordb_journal import _read as native_read

    records = native_read(
        graph,
        pipe,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() WHERE "
        + " OR ".join(selectors)
        + " RETURN properties(r)",
        params,
    )
    props = {}
    for (value,) in records:
        identity = (
            value.get("record_id")
            if value.get("record_id") is not None
            else value.get("uuid")
        )
        if not isinstance(identity, str) or not identity:
            raise MutationPersistenceError(
                "migration required: touched legacy claim has no valid persistent record identity"
            )
        if identity in props:
            raise ClaimLifecycleConflict("ambiguous native claim record identity")
        props[identity] = value
    return props
