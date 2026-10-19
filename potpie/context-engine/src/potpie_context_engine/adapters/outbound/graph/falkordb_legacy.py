"""Lossless Falkor inventory and WATCH-scoped exact legacy repair encoding."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json

from redis.exceptions import WatchError

from potpie_context_engine.core.legacy_records import (
    LegacyInventory,
    LegacyRecord,
    MAX_REPAIR_RECORDS,
    match_record_repairs,
    resolve_entity_identity,
    validate_alias_ownership,
    validate_repair_limits,
)
from potpie_context_engine.core.reconciliation import MutationResult, MutationSummary

from ._mutation_execution import _mutation_result_to_dict, complete_execution
from .falkordb_connection import transaction_client
from .falkordb_journal import _read, _state
from .falkordb_receipts import RECEIPT_COLUMNS
from .falkordb_revision import read_revision
from .native_legacy import CLAIM_INVENTORY, ENTITY_INVENTORY, inventory_record

MAX_REPAIR_BYTES = 8_000_000


def raw_inventory(graph, pot_id, pipe=None):
    """Return raw records paired with locators used only in this read bracket."""
    captured = []
    for kind, template, locator, names in (
        ("entity", ENTITY_INVENTORY, "id(e)", ("locator", "key", "labels", "props")),
        (
            "claim",
            CLAIM_INVENTORY,
            "id(r)",
            (
                "locator",
                "props",
                "subject_key",
                "object_key",
                "subject_pot_id",
                "object_pot_id",
                "relationship_type",
            ),
        ),
    ):
        rows = _read(graph, pipe, template.format(locator=locator), {"pot": pot_id})
        for values in rows:
            raw = dict(zip(names, values))
            captured.append(
                (
                    inventory_record(raw, pot_id=pot_id, kind=kind),
                    (kind, raw["locator"]),
                )
            )
    captured.sort(key=lambda item: (item[0].kind, item[0].key, item[0].fingerprint))
    return tuple(item[0] for item in captured), tuple(item[1] for item in captured)


def repair_limits(state):
    return (
        (
            min(MAX_REPAIR_RECORDS, state.limits.max_records)
            if state
            else MAX_REPAIR_RECORDS
        ),
        min(MAX_REPAIR_BYTES, state.limits.max_bytes) if state else MAX_REPAIR_BYTES,
    )


def legacy_inventory(graph, pot_id):
    """Expose one coherent raw inventory without leaking database-local IDs."""
    while True:
        with transaction_client(graph).pipeline() as pipe:
            try:
                pipe.watch(graph.name)
                read_revision(graph, pot_id, pipe)
                records, _ = raw_inventory(graph, pot_id, pipe)
                state = _state(graph, pot_id, pipe)
                pipe.multi()
                pipe.execute()
                count_limit, byte_limit = repair_limits(state)
                return LegacyInventory(
                    records,
                    journal_active=state is not None,
                    max_repair_records=count_limit,
                    max_repair_bytes=byte_limit,
                )
            except WatchError:
                continue


def prepare_entities(graph, pipe, plan, *, pot_id, mutation_id):
    """Resolve ordinary entity identities from complete raw before-images."""
    records, _ = raw_inventory(graph, pot_id, pipe)
    validate_alias_ownership(records)
    staged = {record.key: record for record in records if record.kind == "entity"}
    counts = Counter(record.key for record in records if record.kind == "entity")
    for item in plan.entity_upserts:
        if counts[item.entity_key] > 1:
            raise ValueError("ambiguous native entity logical identity")
        old = staged.get(item.entity_key)
        if old is not None and "Entity" not in old.labels:
            raise ValueError(
                "migration_required: existing entity is missing its Entity label"
            )
        item.properties = resolve_entity_identity(
            old.properties if old else None,
            item.properties,
            pot_id=pot_id,
            mutation_id=mutation_id,
            entity_key=item.entity_key,
        )
        properties = {**(old.properties if old else {}), **item.properties}
        staged[item.entity_key] = (
            replace(old, properties=properties)
            if old
            else LegacyRecord(
                pot_id, "entity", item.entity_key, properties, labels=tuple(item.labels)
            )
        )
    return records


def validate_final_aliases(records, plan, claim_writes, *, pot_id):
    """Check final ownership across every stored kind and both physical aliases."""
    final = list(records)
    entities = {
        record.key: index
        for index, record in enumerate(final)
        if record.kind == "entity"
    }
    claims = {
        (record.properties.get("record_id") or record.properties.get("uuid")): index
        for index, record in enumerate(final)
        if record.kind == "claim"
    }
    for item in plan.entity_upserts:
        index = entities.get(item.entity_key)
        if index is not None:
            final[index] = replace(
                final[index], properties={**final[index].properties, **item.properties}
            )
        else:
            entities[item.entity_key] = len(final)
            final.append(
                LegacyRecord(pot_id, "entity", item.entity_key, dict(item.properties))
            )
    for write in claim_writes:
        index = claims.get(write["record_id"])
        if index is not None:
            if not final[index].properties.get("record_id"):
                raise ValueError("migration_required: touched claim has no record_id")
            final[index] = replace(final[index], properties=write["props"])
        else:
            claims[write["record_id"]] = len(final)
            final.append(
                LegacyRecord(
                    pot_id,
                    "claim",
                    write["props"].get("claim_key") or "",
                    write["props"],
                )
            )
    validate_alias_ownership(tuple(final))


def prepare_repairs(graph, pipe, plan, *, pot_id, state):
    records, locators = raw_inventory(graph, pot_id, pipe)
    validate_alias_ownership(records)
    effects = tuple(plan.record_repairs)
    selected = match_record_repairs(records, effects, pot_id=pot_id)
    payload = json.dumps(
        [effect.to_dict() for effect in effects], sort_keys=True, separators=(",", ":")
    )
    record_limit, byte_limit = repair_limits(state)
    validate_repair_limits(effects, max_records=record_limit, max_bytes=byte_limit)
    clauses, params, claim_ids = [], {"repair_entities": [], "repair_claims": []}, []
    for effect, target in zip(effects, selected):
        kind, locator = locators[target]
        params["repair_entities" if kind == "entity" else "repair_claims"].append(
            {"locator": locator, "properties": dict(effect.property_updates)}
        )
        if kind == "claim":
            claim_ids.append(effect.after().value("record_id"))
    for key, variable, match in (
        ("repair_entities", "e", "(e)"),
        ("repair_claims", "r", "()-[r]->()"),
    ):
        clauses.append(
            f"CALL {{ WITH v UNWIND ${key} AS repair MATCH {match} WHERE id({variable})=repair.locator "
            f"SET {variable}+=repair.properties RETURN count({variable}) AS {key} }}"
        )
    summary = {
        "repaired_entities": sum(effect.before.kind == "entity" for effect in effects),
        "repaired_claims": sum(effect.before.kind == "claim" for effect in effects),
    }
    return clauses, params, payload, tuple(claim_ids), summary


def compile_repairs(
    graph, pipe, plan, *, pot_id, expected, provenance_context, state, definition
):
    """Compile into the existing atomic executor; journal shares its publisher."""
    clauses, params, payload, claim_ids, summary = prepare_repairs(
        graph, pipe, plan, pot_id=pot_id, state=state
    )
    result = complete_execution(
        MutationResult(
            True,
            provenance_context.mutation_id,
            MutationSummary(),
            administrative_summary=summary,
        ),
        plan,
        pot_id=pot_id,
        version_before=expected,
        provenance=provenance_context,
        record_ids=claim_ids,
        applied_at=datetime.now(timezone.utc),
    )
    if state is not None:
        from .falkordb_journal import FalkorJournal

        return FalkorJournal(graph, definition).compile_record_repairs(
            state=state,
            plan=plan,
            clauses=clauses,
            params=params,
            payload=payload,
            result=result,
            provenance_context=provenance_context,
        )
    params.update(
        pot=pot_id,
        expected=expected,
        mid=result.mutation_id,
        fingerprint=result.execution.batch_fingerprint,
        payload=json.dumps(_mutation_result_to_dict(result), sort_keys=True),
        record_repairs=payload,
        affected=list(claim_ids),
        advance=result.execution.version_after - result.execution.version_before,
    )
    query = " ".join(
        [
            "MATCH (v:PotpieRevision {pot_id:$pot}) OPTIONAL MATCH (prior:PotpieMutationReceipt {pot_id:$pot,mutation_id:$mid}) "
            "WITH v,prior WHERE v.version=$expected AND prior IS NULL",
            *clauses,
            "SET v.version=v.version+$advance "
            "CREATE (receipt:PotpieMutationReceipt {pot_id:$pot,mutation_id:$mid,fingerprint:$fingerprint,payload:$payload,"
            "record_repairs:$record_repairs,entities:0,edges:0,deletes:0,invalidations:0,affected_claim_record_ids:$affected})",
            f"RETURN v.version,{RECEIPT_COLUMNS.replace('r.', 'receipt.')}",
        ]
    )
    check_query_size(graph, query, params, state)
    return query, params, result


def check_query_size(graph, query, params, state):
    if (
        len((graph._build_params_header(params) + query).encode())
        > repair_limits(state)[1]
    ):
        raise ValueError(
            "native legacy repair command exceeds its advertised byte limit"
        )
