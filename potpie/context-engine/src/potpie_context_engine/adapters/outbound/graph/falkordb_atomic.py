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
    validate_new_preparation,
    validate_prepared_execution,
)
from potpie_context_engine.adapters.outbound.graph.apply_plan import _build_provenance
from potpie_context_engine.adapters.outbound.graph.cypher import (
    _clean_entity_text,
    _coerce_props_for_neo4j,
    coherent_entity_labels,
    compact_entity_summary,
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
    counts = {"entities": 0, "edges": 0, "deletes": 0, "invalidations": 0}
    prov = provenance.to_properties()
    for i, item in enumerate(plan.entity_upserts):
        key, props = f"ek{i}", f"ep{i}"
        if item.preserve_labels:
            params[key] = item.entity_key
            params[props] = _coerce_props_for_neo4j({**item.properties, **prov})
            clauses.append(
                f"CALL {{ WITH v MATCH (e:Entity {{group_id:$pot,entity_key:${key}}}) "
                f"SET e+=${props} RETURN count(e) AS ec{i} }}"
            )
            counts["entities"] += 1
            continue
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
            f"ON CREATE SET e.created_at=timestamp() SET e+=${props} "
            f"SET e.name=CASE WHEN $ean{i}<>'' THEN $ean{i} WHEN coalesce(e.name,'')='' THEN ${key} ELSE e.name END, "
            f"e.summary=CASE WHEN $eas{i}<>'' THEN $eas{i} WHEN coalesce(e.summary,'')='' THEN ${key} ELSE e.summary END, "
            f"e.description=CASE WHEN $ead{i}<>'' THEN $ead{i} WHEN coalesce(e.description,'')='' THEN ${key} ELSE e.description END "
            f"{remove_text}"
            f"SET e:{label_text} RETURN count(e) AS ec{i} }}"
        )
        counts["entities"] += 1
    from .falkordb_claims import publication_clauses

    clauses.extend(publication_clauses())
    counts.update(
        edges=len(plan.edge_upserts),
        deletes=len(plan.edge_deletes),
        invalidations=len(plan.invalidations),
    )
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
    edge_expr = "$claim_edges"
    delete_expr = "$claim_deletes"
    invalid_expr = "$claim_invalidations"
    record_expr = "$claim_affected"
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
    from .falkordb_admin import validate_administration

    mutation_id = provenance_context.mutation_id
    pipe = transaction_client(graph).pipeline()
    try:
        pipe.watch(graph.name)
        revision = read_revision(graph, expected_pot_id, pipe)
        current = revision.version if revision.version is not None else 0
        result = revision.execution(mutation_id, fingerprint)
        if result is not None:
            validate_execution_replay(result, provenance=provenance_context)
            return result
        validate_new_preparation(provenance_context)
        prepared = provenance_context.prepared
        if prepared is not None and prepared.expected_version != current:
            raise GraphMutationVersionConflict(
                expected=prepared.expected_version, current=current
            )
        if int(current) != expected_version:
            raise GraphMutationVersionConflict(
                expected=expected_version, current=int(current)
            )
        graph_updated_at = datetime.now(timezone.utc)
        validate_reconciliation_plan(
            plan,
            expected_pot_id,
            definition=definition,
            config=reconciliation_config,
            anchor=prepared.prepared_at if prepared else graph_updated_at,
            provenance=provenance_context,
        )
        validate_administration(plan)
        if revision.version is None:
            _ensure_state(graph, expected_pot_id)
            # Bootstrap itself changed the watched key. Start a fresh bracket
            # before resolving candidates or publishing the first receipt.
            raise WatchError("native revision bootstrapped")
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
        elif plan.record_repairs:
            from .falkordb_legacy import compile_repairs

            query, params, result = compile_repairs(
                graph,
                pipe,
                plan,
                pot_id=expected_pot_id,
                expected=expected_version,
                provenance_context=provenance_context,
                state=state,
                definition=definition,
            )
        else:
            from .falkordb_legacy import prepare_entities, validate_final_aliases

            inventory = prepare_entities(
                graph, pipe, plan, pot_id=expected_pot_id, mutation_id=mutation_id
            )
            provenance = _build_provenance(
                plan,
                pot_id=expected_pot_id,
                mutation_id=mutation_id,
                context=provenance_context,
                graph_updated_at=graph_updated_at,
            )
            from .falkordb_claim_candidates import prepare_native_claims

            original_claims, final_rows = prepare_native_claims(
                graph,
                pipe,
                plan,
                pot_id=expected_pot_id,
                mutation_id=mutation_id,
                provenance=provenance,
                definition=definition,
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
            from .falkordb_claims import encode_native_claims

            encode_native_claims(
                graph,
                pipe,
                plan,
                params,
                provenance,
                embedder,
                original_claims,
                final_rows,
            )
            validate_final_aliases(
                inventory, plan, params["claim_writes"], pot_id=expected_pot_id
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
    except Exception:
        if pipe.watching:
            # A policy failure can precede EXEC after a concurrent writer
            # changed the reads. Surface it only for a still-current bracket;
            # otherwise EXEC raises WatchError and the caller resolves again.
            pipe.multi()
            pipe.ping()
            pipe.execute()
        raise
    finally:
        pipe.reset()
