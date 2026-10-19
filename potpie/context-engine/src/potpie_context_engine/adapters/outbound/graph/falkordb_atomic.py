"""Atomic FalkorDB mutation batches using one guarded GRAPH.QUERY."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone

from falkordb import QueryResult
from potpie_context_engine.core.definition import DEFAULT_GRAPH_DEFINITION
from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.mutation_execution import (
    MutationPersistenceError,
    batch_has_work,
)
from potpie_context_engine.core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_engine.core.reconciliation import MutationResult, MutationSummary
from potpie_context_engine.core.reconciliation_validation import (
    validate_reconciliation_plan,
)
from redis.exceptions import WatchError

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    _mutation_result_to_dict,
    complete_execution,
    mutation_batch_fingerprint,
    validate_execution_replay,
    validate_prepared_execution,
)
from potpie_context_engine.adapters.outbound.graph.apply_plan import _build_provenance
from potpie_context_engine.adapters.outbound.graph.cypher import (
    _REVIVE_CLAUSE,
    _clean_entity_text,
    _coerce_props_for_neo4j,
    _conditional_revive_clause,
    _embedding_props,
    _iso,
    _render_fact,
    _split_edge_properties,
    _stable_source_ref,
    coherent_entity_labels,
    compact_entity_summary,
    evidence_strength_for_truth,
)

from .falkordb_connection import transaction_client
from .falkordb_receipts import RECEIPT_COLUMNS, decode_receipt
from .falkordb_revision import read_revision

_STATE = "PotpieRevision"
_RECEIPT = "PotpieMutationReceipt"


def _rows(graph, response):
    return QueryResult(graph, response).result_set


def current_version(graph, pot_id: str) -> int:
    version = read_revision(graph, pot_id).version
    return version if version is not None else 0


def _ensure_state(graph, pot_id: str) -> None:
    # FalkorDB's MERGE lookup can run before the write lock is acquired.
    # Guard even bootstrap; otherwise two fresh-pot writers can create two
    # revision nodes before either begins its mutation transaction.
    while True:
        with transaction_client(graph).pipeline() as pipe:
            try:
                pipe.watch(graph.name)
                if read_revision(graph, pot_id, pipe).version is not None:
                    return
                pipe.multi()
                pipe.execute_command(
                    "GRAPH.QUERY",
                    graph.name,
                    graph._build_params_header({"pot": pot_id})
                    + f"CREATE (v:{_STATE} {{pot_id:$pot,version:0}})",
                    "--compact",
                )
                pipe.execute()
                return
            except WatchError:
                continue


def reset_pot(graph, pot_id: str, *, mutation_id: str | None = None) -> dict:
    """Internal writer compatibility entry into the canonical mutation boundary."""
    import uuid
    from potpie_context_engine.core.graph_mutations import ProvenanceContext
    from potpie_context_engine.core.reconciliation import MutationBatch

    result = apply_atomic(
        graph,
        MutationBatch(reset_pot=True),
        expected_pot_id=pot_id,
        expected_version=current_version(graph, pot_id),
        provenance_context=ProvenanceContext(
            mutation_id=mutation_id or uuid.uuid4().hex
        ),
    )
    return {
        "ok": result.ok,
        **result.administrative_summary,
        "version": result.execution.version_after,
        "mutation_id": result.mutation_id,
        "execution": result.execution.to_dict(),
    }


def lookup_execution(graph, plan, *, expected_pot_id, mutation_id):
    fingerprint = mutation_batch_fingerprint(plan)
    applied = read_revision(graph, expected_pot_id).execution(mutation_id, fingerprint)
    return MutationExecutionLookup(
        MutationExecutionState.completed.value
        if applied is not None
        else MutationExecutionState.absent.value,
        mutation_id,
        fingerprint,
        result=applied,
    )


def _compile(
    plan,
    *,
    pot_id,
    expected,
    mutation_id,
    fingerprint,
    provenance,
    provenance_context,
    receipt_batch,
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
    clauses = [
        f"MATCH (v:{_STATE} {{pot_id:$pot}}) "
        f"OPTIONAL MATCH (prior:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid}}) "
        "WITH v,prior WHERE v.version=$expected AND prior IS NULL"
    ]
    affected = []
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
            f"RETURN count(r) AS xc{i},collect(DISTINCT coalesce(r.record_id,r.uuid)) AS xr{i} }}"
        )
        affected.append(f"xr{i}")
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
                f"RETURN count(old) AS sc{i},collect(DISTINCT coalesce(old.record_id,old.uuid)) AS sr{i} }}"
            )
            affected.append(f"sr{i}")
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
            f"FOREACH (_ IN CASE WHEN r IS NULL THEN [] ELSE [1] END | SET r.invalid_at=$now,r.expired_at=$now,r.deleted_by=$mid) "
            f"RETURN count(r) AS dc{i},collect(DISTINCT coalesce(r.record_id,r.uuid)) AS dr{i} }}"
        )
        affected.append(f"dr{i}")
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
        aliases = [
            f"r{i}{suffix}"
            for suffix in ("a", "b", "c")
            if any(f"r{i}{suffix}" in match for match in matches)
        ]
        updates = " ".join(
            f"FOREACH (_ IN CASE WHEN {alias} IS NULL THEN [] ELSE [1] END | SET {alias} += $ip{i})"
            for alias in aliases
        )
        count_expr = "+".join(f"count(DISTINCT {alias})" for alias in aliases)
        claim_aliases = [alias for alias in aliases if not alias.endswith("b")]
        claim_ids = ",".join(
            f"collect(DISTINCT coalesce({alias}.record_id,{alias}.uuid)) AS ir{i}{j}"
            for j, alias in enumerate(claim_aliases)
        )
        clauses.append(
            f"CALL {{ WITH v {' '.join(matches)} {updates} RETURN {count_expr} AS ic{i}"
            + (f",{claim_ids}" if claim_ids else "")
            + " }"
        )
        affected.extend(f"ir{i}{j}" for j in range(len(claim_aliases)))
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
                    f"RETURN count(s) AS sic{i},collect(DISTINCT coalesce(s.record_id,s.uuid)) AS sir{i} }}"
                )
                affected.append(f"sir{i}")
        counts["invalidations"] += 1
    result = complete_execution(
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
        receipt_batch,
        pot_id=pot_id,
        version_before=expected,
        provenance=provenance_context,
        applied_at=datetime.fromisoformat(params["now"]),
    )
    params["payload"] = json.dumps(_mutation_result_to_dict(result), sort_keys=True)
    params["advance"] = int(batch_has_work(receipt_batch))
    entity_expr = "+".join(f"ec{i}" for i in range(len(plan.entity_upserts))) or "0"
    edge_expr = "+".join(f"xc{i}" for i in range(len(plan.edge_upserts))) or "0"
    delete_expr = "+".join(f"dc{i}" for i in range(len(plan.edge_deletes))) or "0"
    invalid_expr = "+".join(f"ic{i}" for i in range(len(plan.invalidations))) or "0"
    record_expr = "+".join(affected) or "[]"
    clauses += [
        "SET v.version=v.version+$advance",
        (
            f"CREATE (receipt:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid,fingerprint:$fingerprint,payload:$payload,"
            f"entities:{entity_expr},edges:{edge_expr},deletes:{delete_expr},invalidations:{invalid_expr},"
            f"affected_claim_record_ids:{record_expr}}})"
        ),
        f"RETURN v.version,{RECEIPT_COLUMNS.replace('r.', 'receipt.')}",
    ]
    return (
        " ".join(clauses),
        params,
        result,
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
    validate_prepared_execution(
        plan,
        pot_id=expected_pot_id,
        mutation_id=provenance_context.mutation_id,
        provenance=provenance_context,
    )
    while True:
        try:
            return _apply_watched(
                graph,
                plan,
                expected_pot_id=expected_pot_id,
                expected_version=expected_version,
                provenance_context=provenance_context,
                definition=definition,
                reconciliation_config=reconciliation_config,
                embedder=embedder,
            )
        except WatchError:
            # Re-read authoritative receipt before deciding whether this writer
            # lost a revision race or is replaying the concurrent winner.
            continue


def _apply_watched(
    graph,
    plan,
    *,
    expected_pot_id,
    expected_version,
    provenance_context,
    definition,
    reconciliation_config,
    embedder,
):
    fingerprint = mutation_batch_fingerprint(plan)
    receipt_batch = plan
    plan = deepcopy(plan)
    validate_reconciliation_plan(
        plan, expected_pot_id, definition=definition, config=reconciliation_config
    )
    from .falkordb_admin import validate_administration

    validate_administration(plan)
    mutation_id = provenance_context.mutation_id
    _ensure_state(graph, expected_pot_id)
    pipe = transaction_client(graph).pipeline()
    try:
        pipe.watch(graph.name)
        revision = read_revision(graph, expected_pot_id, pipe)
        current = revision.version if revision.version is not None else 0
        result = revision.execution(mutation_id, fingerprint)
        if result is not None:
            validate_execution_replay(result, provenance=provenance_context)
            return result
        if int(current) != expected_version:
            raise GraphMutationVersionConflict(
                expected=expected_version, current=int(current)
            )
        from .falkordb_journal import FalkorJournal, _state

        state = _state(graph, expected_pot_id, pipe)
        if plan.reset_pot or plan.snapshot_import is not None:
            from potpie_context_engine.core.graph_journal import JournalError
            from .falkordb_admin import compile_administration

            if state is not None:
                raise JournalError(
                    "reset/snapshot import is unsupported while journal capture is active"
                )
            query, params, result = compile_administration(
                graph,
                pipe,
                plan,
                pot_id=expected_pot_id,
                expected=expected_version,
                provenance_context=provenance_context,
            )
        else:
            provenance = _build_provenance(
                plan,
                pot_id=expected_pot_id,
                mutation_id=mutation_id,
                context=provenance_context,
                graph_updated_at=datetime.now(timezone.utc),
            )
            query, params, result = _compile(
                plan,
                pot_id=expected_pot_id,
                expected=expected_version,
                mutation_id=mutation_id,
                fingerprint=fingerprint,
                provenance=provenance,
                provenance_context=provenance_context,
                receipt_batch=receipt_batch,
                definition=definition,
                embedder=embedder,
            )
            params["journal_generated_event"] = (
                plan.event_ref is None and provenance_context.source_event_id is None
            )
            params["journal_generated_valid_from"] = (
                provenance_context.event_occurred_at is None
            )
            if state is not None:
                return FalkorJournal(graph, definition).apply(
                    plan,
                    pipe=pipe,
                    state=state,
                    params=params,
                    fingerprint=fingerprint,
                    actor=provenance_context.actor_user_id or "system",
                    provenance_context=provenance_context,
                    receipt_batch=receipt_batch,
                )
            from .falkordb_identity import validate_touched_claim_identities

            validate_touched_claim_identities(graph, pipe, plan, params, definition)
        pipe.multi()
        pipe.execute_command(
            "GRAPH.QUERY",
            graph.name,
            graph._build_params_header(params) + query,
            "--compact",
        )
        responses = pipe.execute()
        rows = _rows(graph, responses[0])
        if len(rows) != 1:
            raise MutationPersistenceError(
                "native execution guard did not publish exactly once"
            )
        return decode_receipt(
            rows[0][1:],
            pot_id=expected_pot_id,
            mutation_id=mutation_id,
            fingerprint=fingerprint,
        )
    finally:
        pipe.reset()
