"""Compile administrative effects inside FalkorDB's existing watched execution."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re

from potpie_context_engine.core.graph_snapshot import validate_snapshot_merge
from potpie_context_engine.core.graph_snapshot_mutation import (
    validate_snapshot_identities,
)
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.reconciliation import MutationResult, MutationSummary

from ._mutation_execution import _mutation_result_to_dict, complete_execution
from .cypher import _coerce_props_for_neo4j
from .falkordb_journal import _read
from .falkordb_receipts import RECEIPT_COLUMNS
from .native_snapshot import (
    stored_claim,
    stored_entity,
    validate_native_snapshot_properties,
)

_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def validate_administration(plan):
    incoming = plan.snapshot_import
    if incoming is None:
        return
    validate_native_snapshot_properties(incoming)
    for entity in incoming["entities"]:
        if any(not _LABEL.fullmatch(label) for label in entity["labels"]):
            raise ValueError(f"invalid snapshot entity label: {entity['labels']!r}")


def compile_administration(graph, pipe, plan, *, pot_id, expected, provenance_context):
    mutation_id = provenance_context.mutation_id
    params = {"pot": pot_id, "expected": expected, "mid": mutation_id}
    clauses = [
        "MATCH (v:PotpieRevision {pot_id:$pot}) "
        "OPTIONAL MATCH (prior:PotpieMutationReceipt {pot_id:$pot,mutation_id:$mid}) "
        "WITH v,prior WHERE v.version=$expected AND prior IS NULL AND v.journal IS NULL"
    ]
    if plan.reset_pot:
        result, record_ids = _reset_effects(graph, pipe, params, clauses)
    else:
        result, record_ids = _snapshot_effects(
            graph, pipe, plan.snapshot_import, params, clauses
        )
    completed = complete_execution(
        result,
        plan,
        pot_id=pot_id,
        version_before=expected,
        provenance=provenance_context,
        record_ids=record_ids,
        applied_at=datetime.now(timezone.utc),
    )
    params.update(
        fingerprint=completed.execution.batch_fingerprint,
        payload=json.dumps(_mutation_result_to_dict(completed), sort_keys=True),
        affected=list(completed.execution.affected_claim_record_ids),
        advance=completed.execution.version_after - completed.execution.version_before,
        entities=result.mutation_summary.entity_upserts_applied,
        edges=result.mutation_summary.edge_upserts_applied,
    )
    clauses.extend(
        [
            "SET v.version=v.version+$advance",
            "CREATE (receipt:PotpieMutationReceipt {pot_id:$pot,mutation_id:$mid,"
            "fingerprint:$fingerprint,payload:$payload,entities:$entities,edges:$edges,"
            "deletes:0,invalidations:0,affected_claim_record_ids:$affected})",
            f"RETURN v.version,{RECEIPT_COLUMNS.replace('r.', 'receipt.')}",
        ]
    )
    return " ".join(clauses), params, completed


def _reset_effects(graph, pipe, params, clauses):
    rows = _read(
        graph,
        pipe,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() RETURN coalesce(r.record_id,r.uuid)",
        params,
    )
    record_ids = tuple(row[0] for row in rows)
    if any(not isinstance(value, str) or not value for value in record_ids):
        raise MutationPersistenceError(
            "migration required: reset claim has no persistent record identity"
        )
    count = _read(graph, pipe, "MATCH (n {group_id:$pot}) RETURN count(n)", params)[0][
        0
    ]
    clauses.extend(
        [
            "OPTIONAL MATCH (n {group_id:$pot}) WITH v,collect(n) AS nodes",
            "FOREACH (item IN nodes | DETACH DELETE item)",
        ]
    )
    return MutationResult(
        True,
        params["mid"],
        MutationSummary(),
        administrative_summary={
            "group_id_nodes_before": count,
            "group_id_nodes_remaining": 0,
        },
    ), record_ids


def _snapshot_effects(graph, pipe, incoming, params, clauses):
    from .falkordb_snapshot import read_snapshot

    existing = read_snapshot(graph, pipe, params["pot"])
    validate_snapshot_merge(
        existing_entities=existing["entities"],
        existing_claims=existing["claims"],
        incoming=incoming,
    )
    entity_keys = {row["key"] for row in existing["entities"]}
    claim_keys = {row["claim_key"] for row in existing["claims"]}
    entities = [row for row in incoming["entities"] if row["key"] not in entity_keys]
    claims = [row for row in incoming["claims"] if row["claim_key"] not in claim_keys]
    validate_snapshot_identities(
        {
            "entities": existing["entities"] + entities,
            "claims": existing["claims"] + claims,
        }
    )
    for i, entity in enumerate(entities):
        key, labels, props = stored_entity(entity, params["pot"])
        params.update({f"ek{i}": key, f"ep{i}": _coerce_props_for_neo4j(props)})
        suffix = "".join(f":{label}" for label in labels)
        clauses.append(
            f"CALL {{ WITH v CREATE (e:Entity {{group_id:$pot,entity_key:$ek{i}}}) "
            f"SET e += $ep{i} "
            + (f"SET e{suffix} " if suffix else "")
            + f"RETURN count(e) AS ec{i} }}"
        )
    for i, claim in enumerate(claims):
        props = stored_claim(claim, params["pot"])
        embedding = props.pop("fact_embedding", None)
        params.update(
            {
                f"cp{i}": _coerce_props_for_neo4j(props),
                f"cs{i}": claim["subject_key"],
                f"co{i}": claim["object_key"],
            }
        )
        vector_set = ""
        if embedding is not None:
            params[f"ce{i}"] = embedding
            vector_set = f"SET r.fact_embedding=vecf32($ce{i}) "
        clauses.append(
            f"CALL {{ WITH v MATCH (a:Entity {{group_id:$pot,entity_key:$cs{i}}}),"
            f"(b:Entity {{group_id:$pot,entity_key:$co{i}}}) "
            f"CREATE (a)-[r:RELATES_TO]->(b) SET r += $cp{i} {vector_set}RETURN count(r) AS cc{i} }}"
        )
    return MutationResult(
        True,
        params["mid"],
        MutationSummary(
            entity_upserts_applied=len(entities), edge_upserts_applied=len(claims)
        ),
        administrative_summary={
            "imported_entities": len(entities),
            "imported_claims": len(claims),
        },
    ), tuple(row["record_id"] for row in claims)
