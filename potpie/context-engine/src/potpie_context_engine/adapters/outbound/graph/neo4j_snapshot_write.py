"""Exact snapshot/reset effects within an already-owned Neo4j mutation transaction."""

from __future__ import annotations

import re
from typing import Any, Mapping

from potpie_context_engine.core.graph_snapshot import validate_snapshot_merge
from potpie_context_engine.core.graph_snapshot_mutation import normalize_snapshot_import
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.reconciliation import MutationResult, MutationSummary

from .cypher import _coerce_props_for_neo4j
from .native_snapshot import (
    claim_row,
    entity_row,
    payload_from_rows,
    stored_claim,
    stored_entity,
    validate_native_snapshot_properties,
)
from .neo4j_execution_state import _one
from .neo4j_snapshot import _SNAPSHOT

_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


async def merge_snapshot(
    transaction: Any,
    *,
    pot_id: str,
    mutation_id: str,
    incoming: Mapping[str, Any],
) -> tuple[MutationResult, tuple[str, ...]]:
    """Validate the complete merge before applying exact missing native records."""
    if normalize_snapshot_import(incoming, pot_id=pot_id) != incoming:
        raise ValueError("snapshot import requires normalized stable record identities")
    validate_native_snapshot_properties(incoming)
    for entity in incoming["entities"]:
        labels = tuple(entity["labels"])
        if any(not _LABEL.fullmatch(label) for label in labels):
            raise ValueError(f"invalid snapshot entity label: {labels!r}")
    row = await _one(transaction, _SNAPSHOT, pot=pot_id)
    existing = payload_from_rows(
        pot_id=pot_id,
        entity_rows=(
            entity_row(item["key"], item["labels"], item["props"])
            for item in (row["entities"] if row else ())
            if item is not None
        ),
        claim_rows=(
            claim_row(props)
            for props in (row["claims"] if row else ())
            if props is not None
        ),
    )
    validate_snapshot_merge(
        existing_entities=existing["entities"],
        existing_claims=existing["claims"],
        incoming=incoming,
    )
    entity_keys = {item["key"] for item in existing["entities"]}
    claim_keys = {item["claim_key"] for item in existing["claims"]}
    entities = [item for item in incoming["entities"] if item["key"] not in entity_keys]
    claims = [
        item for item in incoming["claims"] if item["claim_key"] not in claim_keys
    ]
    for entity in entities:
        key, labels, props = stored_entity(entity, pot_id)
        suffix = "".join(f":{label}" for label in labels)
        query = "MERGE (e:Entity {group_id:$pot,entity_key:$key}) SET e += $props"
        if suffix:
            query += f" SET e{suffix}"
        result = await transaction.run(
            query, pot=pot_id, key=key, props=_coerce_props_for_neo4j(props)
        )
        await result.consume()
    for claim in claims:
        result = await transaction.run(
            "MATCH (a:Entity {group_id:$pot,entity_key:$subject}),"
            "(b:Entity {group_id:$pot,entity_key:$object}) "
            "CREATE (a)-[r:RELATES_TO]->(b) SET r += $props",
            pot=pot_id,
            subject=claim["subject_key"],
            object=claim["object_key"],
            props=_coerce_props_for_neo4j(stored_claim(claim, pot_id)),
        )
        await result.consume()
    return MutationResult(
        ok=True,
        mutation_id=mutation_id,
        mutation_summary=MutationSummary(
            entity_upserts_applied=len(entities), edge_upserts_applied=len(claims)
        ),
        administrative_summary={
            "imported_entities": len(entities),
            "imported_claims": len(claims),
        },
    ), tuple(claim["record_id"] for claim in claims)


async def reset_graph(
    transaction: Any, *, pot_id: str, mutation_id: str
) -> tuple[MutationResult, tuple[str, ...]]:
    affected = await _one(
        transaction,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
        "RETURN collect(coalesce(r.record_id,r.uuid)) AS ids,count(r) AS count",
        pot=pot_id,
    )
    record_ids = affected["ids"] if affected is not None else []
    if (
        affected is None
        or len(record_ids) != affected["count"]
        or any(not isinstance(value, str) or not value for value in record_ids)
    ):
        raise MutationPersistenceError(
            "reset affected Neo4j claim has no record identity"
        )
    before = await _one(
        transaction,
        "MATCH (n {group_id:$pot}) WITH collect(n) AS nodes "
        "FOREACH (n IN nodes | DETACH DELETE n) RETURN size(nodes) AS count",
        pot=pot_id,
    )
    return MutationResult(
        ok=True,
        mutation_id=mutation_id,
        mutation_summary=MutationSummary(),
        administrative_summary={
            "group_id_nodes_before": before["count"],
            "group_id_nodes_remaining": 0,
        },
    ), tuple(record_ids)
