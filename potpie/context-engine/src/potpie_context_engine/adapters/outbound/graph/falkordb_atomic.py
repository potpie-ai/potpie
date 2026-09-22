"""Atomic FalkorDB mutation batches using one guarded GRAPH.QUERY."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

from falkordb import QueryResult
from redis.exceptions import ResponseError, WatchError

from potpie_context_core.definition import DEFAULT_GRAPH_DEFINITION
from potpie_context_core.errors import GraphMutationVersionConflict
from potpie_context_core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_core.reconciliation import MutationResult, MutationSummary
from potpie_context_core.reconciliation_validation import validate_reconciliation_plan
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
    mutation_batch_fingerprint,
)
from potpie_context_engine.adapters.outbound.graph.apply_plan import _build_provenance
from potpie_context_engine.adapters.outbound.graph.cypher import (
    _coerce_props_for_neo4j,
    _clean_entity_text,
    _embedding_props,
    _iso,
    _render_fact,
    _REVIVE_CLAUSE,
    _conditional_revive_clause,
    _split_edge_properties,
    _stable_source_ref,
    coherent_entity_labels,
    compact_entity_summary,
    evidence_strength_for_truth,
)

_STATE = "PotpieRevision"
_RECEIPT = "PotpieMutationReceipt"


def _rows(graph, response):
    return QueryResult(graph, response).result_set


def current_version(graph, pot_id: str) -> int:
    try:
        result = graph.ro_query(
            f"MATCH (r:{_STATE} {{pot_id:$pot}}) RETURN r.version",
            params={"pot": pot_id},
        )
    except ResponseError as exc:
        if "empty key" in str(exc).lower():
            return 0
        raise
    return int(result.result_set[0][0]) if result.result_set else 0


def _ensure_state(graph, pot_id: str) -> None:
    graph.query(
        f"MERGE (r:{_STATE} {{pot_id:$pot}}) ON CREATE SET r.version=0",
        params={"pot": pot_id},
    )


def reset_pot(graph, pot_id: str) -> dict[str, int | bool]:
    """Delete canonical pot entities and advance revision in one graph command."""
    _ensure_state(graph, pot_id)
    while True:
        pipe = graph.client.pipeline()
        try:
            pipe.watch(graph.name)
            before = current_version(graph, pot_id)
            query = graph._build_params_header({"pot": pot_id, "expected": before}) + (
                f"MATCH (v:{_STATE} {{pot_id:$pot}}) WHERE v.version=$expected "
                "OPTIONAL MATCH (n:Entity {group_id:$pot}) WITH v,collect(n) AS nodes "
                "FOREACH (item IN nodes | DETACH DELETE item) "
                "SET v.version=v.version+1 RETURN size(nodes),v.version"
            )
            pipe.multi()
            pipe.execute_command("GRAPH.QUERY", graph.name, query, "--compact")
            rows = _rows(graph, pipe.execute()[0])
            if not rows:
                continue
            row = rows[0]
            return {
                "ok": True,
                "group_id_nodes_before": int(row[0]),
                "group_id_nodes_remaining": 0,
                "version": int(row[1]),
            }
        except WatchError:
            continue
        finally:
            pipe.reset()


def lookup_execution(graph, plan, *, expected_pot_id, mutation_id):
    fingerprint = mutation_batch_fingerprint(plan)
    current_version(graph, expected_pot_id)
    result = graph.query(
        f"MATCH (r:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid}}) "
        "RETURN r.fingerprint,r.payload,r.entities,r.edges,r.deletes,r.invalidations",
        params={"pot": expected_pot_id, "mid": mutation_id},
    )
    if not result.result_set:
        return MutationExecutionLookup(
            MutationExecutionState.absent.value, mutation_id, fingerprint
        )
    stored, payload, entities, edges, deletes, invalidations = result.result_set[0]
    if stored != fingerprint:
        raise MutationExecutionReuseError(
            "mutation ID was already used with a different batch"
        )
    raw = json.loads(payload)
    summary = raw["mutation_summary"]
    summary.update(
        entity_upserts_applied=int(entities or 0),
        edge_upserts_applied=int(edges or 0),
        edge_deletes_applied=int(deletes or 0),
        invalidations_applied=int(invalidations or 0),
    )
    applied = MutationResult(
        True,
        mutation_id,
        MutationSummary(**summary),
        downgrades=raw.get("downgrades", []),
    )
    return MutationExecutionLookup(
        MutationExecutionState.completed.value, mutation_id, fingerprint, result=applied
    )


def _existing(graph, pipe, pot_id, mutation_id):
    query = graph._build_params_header({"pot": pot_id, "mid": mutation_id}) + (
        f"MATCH (v:{_STATE} {{pot_id:$pot}}) "
        f"OPTIONAL MATCH (r:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid}}) "
        "RETURN v.version,r.fingerprint,r.payload,r.entities,r.edges,r.deletes,r.invalidations"
    )
    response = pipe.execute_command(
        "GRAPH.RO_QUERY",
        graph.name,
        query,
        "--compact",
    )
    rows = _rows(graph, response)
    return rows[0] if rows else (0, None, None, None, None, None, None)


def _compile(
    plan,
    *,
    pot_id,
    expected,
    mutation_id,
    fingerprint,
    provenance,
    definition,
    embedder=None,
):
    params = {
        "pot": pot_id,
        "expected": expected,
        "mid": mutation_id,
        "fingerprint": fingerprint,
        "now": datetime.now(timezone.utc).isoformat(),
    }
    clauses = [f"MATCH (v:{_STATE} {{pot_id:$pot}}) WHERE v.version=$expected"]
    counts = {"entities": 0, "edges": 0, "deletes": 0, "invalidations": 0}
    prov = provenance.to_properties()
    for i, item in enumerate(plan.entity_upserts):
        key, props = f"ek{i}", f"ep{i}"
        labels = coherent_entity_labels(
            item.entity_key,
            item.labels,
            entity_types=definition.entity_types,
        )
        label_text = ":".join(sorted(set(labels)))
        stale = sorted(set(definition.entity_types) - set(labels))
        remove_text = f"REMOVE e:{':'.join(stale)} " if stale else ""
        raw = dict(item.properties)
        authored_name = _clean_entity_text(raw.pop("name", None))
        raw.update(prov)
        raw["group_id"] = pot_id
        raw["provenance_source_event"] = provenance.source_event_id
        summary = compact_entity_summary(
            raw.pop("summary", None),
            raw.get("description"),
            raw.get("title"),
            authored_name,
        )
        description = _clean_entity_text(raw.pop("description", None)) or summary
        params[key], params[props] = item.entity_key, _coerce_props_for_neo4j(raw)
        params[f"ean{i}"] = authored_name
        params[f"eas{i}"] = summary
        params[f"ead{i}"] = description
        clauses.append(
            f"CALL {{ WITH v MERGE (e:Entity {{group_id:$pot,entity_key:${key}}}) "
            f"ON CREATE SET e.uuid=randomUUID(),e.created_at=timestamp() SET e+=${props} "
            f"SET e.name=CASE WHEN $ean{i}<>'' THEN $ean{i} WHEN coalesce(e.name,'')='' THEN ${key} ELSE e.name END, "
            f"e.summary=CASE WHEN $eas{i}<>'' THEN $eas{i} WHEN coalesce(e.summary,'')='' THEN ${key} ELSE e.summary END, "
            f"e.description=CASE WHEN $ead{i}<>'' THEN $ead{i} WHEN coalesce(e.description,'')='' THEN ${key} ELSE e.description END "
            f"{remove_text}"
            f"SET e:{label_text} RETURN count(e) AS ec{i} }}"
        )
        counts["entities"] += 1
    for i, item in enumerate(plan.edge_upserts):
        reserved, extras = _split_edge_properties(item.properties)
        source = reserved.get("source_ref") or _stable_source_ref(
            predicate=item.edge_type,
            from_key=item.from_entity_key,
            to_key=item.to_entity_key,
            provenance=provenance,
        )
        raw = dict(extras)
        valid_at = (
            _iso(reserved.get("valid_at"))
            or _iso(provenance.valid_from)
            or params["now"]
        )
        source_system = (
            reserved.get("source_system") or provenance.source_system or "agent"
        )
        truth = extras.get("truth") if isinstance(extras.get("truth"), str) else None
        raw.setdefault(
            "fact",
            _render_fact(
                predicate=item.edge_type,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                extra=reserved,
            ),
        )
        raw["valid_at"] = valid_at
        raw["source_system"] = source_system
        raw["evidence_strength"] = evidence_strength_for_truth(truth)
        raw["observed_at"] = params["now"]
        if reserved.get("fact_embedding") is not None:
            raw["fact_embedding"] = reserved["fact_embedding"]
        if reserved.get("confidence") is not None:
            raw["confidence"] = float(reserved["confidence"])
        elif provenance.confidence is not None:
            raw["confidence"] = float(provenance.confidence)
        raw.update(
            _embedding_props(
                embedder=embedder,
                predicate=item.edge_type,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                edge_props=raw,
            )
        )
        raw["provenance_source_event"] = provenance.source_event_id
        raw["mutation_id"] = mutation_id
        claim_key = raw.get("claim_key")
        preserve_lifecycle = raw.get("evidence_review_required") is True
        for name, value in (
            (f"ef{i}", item.from_entity_key),
            (f"et{i}", item.to_entity_key),
            (f"en{i}", item.edge_type),
            (f"es{i}", source),
            (f"eprop{i}", _coerce_props_for_neo4j(raw)),
            (f"eclaim{i}", claim_key),
            (f"ev{i}", valid_at),
            (f"epreserve{i}", preserve_lifecycle),
        ):
            params[name] = value
        clauses.append(
            f"CALL {{ WITH v MATCH (a:Entity {{group_id:$pot,entity_key:$ef{i}}}) "
            f"MATCH (b:Entity {{group_id:$pot,entity_key:$et{i}}}) "
            f"MERGE (a)-[r:RELATES_TO {{group_id:$pot,name:$en{i},subject_key:$ef{i},object_key:$et{i},source_ref:$es{i}"
            + (f",claim_key:$eclaim{i}" if claim_key else "")
            + "}]->(b) "
            f"ON CREATE SET r.uuid=randomUUID(),r.created_at=$now "
            f"SET r.revived_at=CASE WHEN r.invalid_at IS NULL OR ($epreserve{i} AND r.invalid_at IS NOT NULL) "
            f"THEN r.revived_at ELSE $now END SET {_conditional_revive_clause(alias='r', preserve_param=f'epreserve{i}')} "
            f"SET r+= $eprop{i} "
            f"RETURN count(r) AS xc{i} }}"
        )
        if (
            item.edge_type in definition.singleton_predicates
            and evidence_strength_for_truth(truth) == "deterministic"
        ):
            clauses.append(
                f"CALL {{ WITH v OPTIONAL MATCH (:Entity {{group_id:$pot,entity_key:$ef{i}}})"
                f"-[old:RELATES_TO {{group_id:$pot,name:$en{i}}}]->(other:Entity) "
                f"WHERE other.entity_key <> $et{i} AND old.invalid_at IS NULL "
                "FOREACH (_ IN CASE WHEN old IS NULL THEN [] ELSE [1] END | "
                f"SET old.invalid_at=$ev{i},old.expired_at=$now,"
                f"old.superseded_by_object=$et{i},old.supersession_reason='singleton_predicate') "
                f"RETURN count(old) AS sc{i} }}"
            )
        counts["edges"] += 1
    for i, item in enumerate(plan.edge_deletes):
        params.update(
            {
                f"df{i}": item.from_entity_key,
                f"dt{i}": item.to_entity_key,
                f"dn{i}": item.edge_type,
            }
        )
        clauses.append(
            f"CALL {{ WITH v OPTIONAL MATCH (:Entity {{group_id:$pot,entity_key:$df{i}}})-[r:RELATES_TO {{group_id:$pot,name:$dn{i},object_key:$dt{i}}}]->() "
            f"FOREACH (_ IN CASE WHEN r IS NULL THEN [] ELSE [1] END | SET r.invalid_at=$now,r.expired_at=$now,r.deleted_by=$mid) RETURN count(r) AS dc{i} }}"
        )
        counts["deletes"] += 1
    for i, item in enumerate(plan.invalidations):
        prop = {
            "invalid_at": item.valid_to or params["now"],
            "expired_at": params["now"],
            "invalidation_reason": item.reason,
            "invalidated_by": provenance.source_event_id,
        }
        params[f"ip{i}"] = prop
        matches = []
        if item.target_claim_keys is not None:
            params[f"ick{i}"] = list(item.target_claim_keys)
            matches.append(
                f"OPTIONAL MATCH ()-[r{i}a:RELATES_TO {{group_id:$pot}}]->() "
                f"WHERE r{i}a.claim_key IN $ick{i} AND r{i}a.invalid_at IS NULL"
            )
        if item.target_entity_key:
            params[f"iek{i}"] = item.target_entity_key
            matches.append(
                f"OPTIONAL MATCH (r{i}b:Entity {{group_id:$pot,entity_key:$iek{i}}})"
            )
        if not matches:
            edge, src, dst = item.target_edge or ("", "", "")
            params.update({f"in{i}": edge, f"if{i}": src, f"it{i}": dst})
            matches.append(
                f"OPTIONAL MATCH (:Entity {{group_id:$pot,entity_key:$if{i}}})-"
                f"[r{i}c:RELATES_TO {{group_id:$pot,name:$in{i},object_key:$it{i}}}]->() "
                f"WHERE r{i}c.invalid_at IS NULL"
            )
        aliases = [f"r{i}{suffix}" for suffix in ("a", "b", "c") if any(f"r{i}{suffix}" in match for match in matches)]
        updates = " ".join(
            f"FOREACH (_ IN CASE WHEN {alias} IS NULL THEN [] ELSE [1] END | SET {alias} += $ip{i})"
            for alias in aliases
        )
        count_expr = "+".join(f"count(DISTINCT {alias})" for alias in aliases)
        clauses.append(
            f"CALL {{ WITH v {' '.join(matches)} {updates} RETURN {count_expr} AS ic{i} }}"
        )
        if item.superseded_by_key:
            old_key = item.target_entity_key or (
                item.target_edge[2] if item.target_edge else None
            )
            if old_key:
                supersedes_revive = _REVIVE_CLAUSE.replace("r.", "s.")
                supersedes_source = _stable_source_ref(
                    predicate="SUPERSEDES",
                    from_key=item.superseded_by_key,
                    to_key=old_key,
                    provenance=provenance,
                )
                params[f"isn{i}"] = item.superseded_by_key
                params[f"iso{i}"] = old_key
                params[f"isr{i}"] = item.reason
                params[f"iss{i}"] = supersedes_source
                params[f"isprop{i}"] = _coerce_props_for_neo4j(
                    {
                        "fact": item.reason,
                        "reason": item.reason,
                        "valid_at": item.valid_to or params["now"],
                        "source_system": provenance.source_system or "agent",
                        "evidence_strength": "deterministic",
                        "observed_at": params["now"],
                        "provenance_source_event": provenance.source_event_id,
                        "mutation_id": mutation_id,
                    }
                )
                clauses.append(
                    f"CALL {{ WITH v MATCH (new:Entity {{group_id:$pot,entity_key:$isn{i}}}) "
                    f"MATCH (old:Entity {{group_id:$pot,entity_key:$iso{i}}}) "
                    "MERGE (new)-[s:RELATES_TO {group_id:$pot,name:'SUPERSEDES',"
                    f"subject_key:$isn{i},object_key:$iso{i},source_ref:$iss{i}}}]->(old) "
                    "ON CREATE SET s.uuid=randomUUID(),s.created_at=$now "
                    "SET s.revived_at=CASE WHEN s.invalid_at IS NULL THEN s.revived_at ELSE $now END "
                    f"SET {supersedes_revive} SET s += $isprop{i} "
                    f"RETURN count(s) AS sic{i} }}"
                )
        counts["invalidations"] += 1
    payload = json.dumps(
        {
            "ok": True,
            "mutation_id": mutation_id,
            "mutation_summary": {
                "entity_upserts_applied": counts["entities"],
                "edge_upserts_applied": counts["edges"],
                "edge_deletes_applied": counts["deletes"],
                "invalidations_applied": counts["invalidations"],
                "stamp_counts": {},
            },
            "error": None,
            "reconciliation_errors": [],
            "downgrades": list(plan.ontology_downgrades),
        },
        sort_keys=True,
    )
    params["payload"] = payload
    entity_expr = "+".join(f"ec{i}" for i in range(len(plan.entity_upserts))) or "0"
    edge_expr = "+".join(f"xc{i}" for i in range(len(plan.edge_upserts))) or "0"
    delete_expr = "+".join(f"dc{i}" for i in range(len(plan.edge_deletes))) or "0"
    invalid_expr = "+".join(f"ic{i}" for i in range(len(plan.invalidations))) or "0"
    clauses += [
        "SET v.version=v.version+1",
        f"CREATE (receipt:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid,fingerprint:$fingerprint,payload:$payload,"
        f"entities:{entity_expr},edges:{edge_expr},deletes:{delete_expr},invalidations:{invalid_expr}}})",
        f"RETURN v.version,{entity_expr} AS entities,{edge_expr} AS edges,{delete_expr} AS deletes,{invalid_expr} AS invalidations",
    ]
    return (
        " ".join(clauses),
        params,
        MutationResult(
            True,
            mutation_id,
            MutationSummary(
                counts["entities"],
                counts["edges"],
                counts["deletes"],
                counts["invalidations"],
            ),
            downgrades=list(plan.ontology_downgrades),
        ),
    )


def apply_atomic(
    graph,
    plan,
    *,
    expected_pot_id,
    expected_version,
    provenance_context,
    definition=DEFAULT_GRAPH_DEFINITION,
    reconciliation_config=None,
    embedder=None,
):
    plan = deepcopy(plan)
    validate_reconciliation_plan(
        plan, expected_pot_id, definition=definition, config=reconciliation_config
    )
    mutation_id = provenance_context.mutation_id
    fingerprint = mutation_batch_fingerprint(plan)
    now = datetime.now(timezone.utc)
    provenance = _build_provenance(
        plan,
        pot_id=expected_pot_id,
        mutation_id=mutation_id,
        context=provenance_context,
        graph_updated_at=now,
    )
    _ensure_state(graph, expected_pot_id)
    pipe = graph.client.pipeline()
    try:
        pipe.watch(graph.name)
        current, stored_fp, payload, entities, edges, deletes, invalidations = (
            _existing(graph, pipe, expected_pot_id, mutation_id)
        )
        if stored_fp:
            if stored_fp != fingerprint:
                raise MutationExecutionReuseError(
                    "mutation ID was already used with a different batch"
                )
            raw = json.loads(payload)
            summary = raw["mutation_summary"]
            summary.update(
                entity_upserts_applied=int(entities or 0),
                edge_upserts_applied=int(edges or 0),
                edge_deletes_applied=int(deletes or 0),
                invalidations_applied=int(invalidations or 0),
            )
            return MutationResult(
                True,
                mutation_id,
                MutationSummary(**summary),
                downgrades=raw.get("downgrades", []),
            )
        if int(current) != expected_version:
            raise GraphMutationVersionConflict(
                expected=expected_version, current=int(current)
            )
        query, params, result = _compile(
            plan,
            pot_id=expected_pot_id,
            expected=expected_version,
            mutation_id=mutation_id,
            fingerprint=fingerprint,
            provenance=provenance,
            definition=definition,
            embedder=embedder,
        )
        pipe.multi()
        pipe.execute_command(
            "GRAPH.QUERY",
            graph.name,
            graph._build_params_header(params) + query,
            "--compact",
        )
        responses = pipe.execute()
        row = _rows(graph, responses[0])[0]
        result.mutation_summary = MutationSummary(
            int(row[1]), int(row[2]), int(row[3]), int(row[4])
        )
        return result
    except WatchError:
        raise GraphMutationVersionConflict(
            expected=expected_version, current=current_version(graph, expected_pot_id)
        )
    finally:
        pipe.reset()
