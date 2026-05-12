"""Canonical Potpie ontology mutations applied through Graphiti's driver.

Phase 1 of the rebuild lands a single write path: every mutation produced
by a validated ``ReconciliationPlan`` flows through these functions, which
share Graphiti's driver. The companion read port (``StructuralReadPort``)
is restricted to reads.

The Cypher MERGE patterns mirror what ``Neo4jStructuralAdapter`` previously
did; the only difference is the driver and the location. Identity stays on
``(group_id, entity_key)`` and edges stay typed by relation label, so no
schema change is forced by this consolidation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from domain.ontology import ENTITY_TYPES, is_canonical_entity_label


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_safe_cypher_identifier(value: str) -> bool:
    if not value:
        return False
    if not (value[0].isalpha() or value[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in value)


async def upsert_entities_async(
    driver: Any,
    pot_id: str,
    items: list[EntityUpsert],
    provenance: ProvenanceRef,
) -> int:
    if not items:
        return 0
    count = 0
    prov_props = provenance.to_properties()
    async with driver.session() as session:
        for item in items:
            props = dict(item.properties)
            props["group_id"] = pot_id
            props["provenance_source_event"] = provenance.source_event_id
            props.update(prov_props)
            # Graphiti expects every :Entity node to have a non-null name.
            props.setdefault("name", item.entity_key)
            await session.run(
                "MERGE (e:Entity {group_id: $gid, entity_key: $key}) "
                "ON CREATE SET e.uuid = randomUUID(), e.created_at = timestamp() "
                "SET e += $props",
                gid=pot_id,
                key=item.entity_key,
                props=props,
            )
            for lbl in item.labels:
                if lbl == "Entity":
                    continue
                if not is_canonical_entity_label(lbl) or lbl not in ENTITY_TYPES:
                    continue
                await session.run(
                    f"MATCH (e:Entity {{group_id: $gid, entity_key: $key}}) SET e:{lbl}",  # pyright: ignore[reportArgumentType]
                    gid=pot_id,
                    key=item.entity_key,
                )
            count += 1
    return count


async def upsert_edges_async(
    driver: Any,
    pot_id: str,
    items: list[EdgeUpsert],
    provenance: ProvenanceRef,
) -> int:
    if not items:
        return 0
    count = 0
    now_iso = _utc_now_iso()
    prov_props = provenance.to_properties()
    async with driver.session() as session:
        for item in items:
            if not _is_safe_cypher_identifier(item.edge_type):
                continue
            props = dict(item.properties)
            props.setdefault("valid_from", now_iso)
            props["provenance_source_event"] = provenance.source_event_id
            props.update(prov_props)
            res = await session.run(
                f"MATCH (a:Entity {{group_id: $gid, entity_key: $from_key}}) "  # pyright: ignore[reportArgumentType]
                f"MATCH (b:Entity {{group_id: $gid, entity_key: $to_key}}) "
                f"MERGE (a)-[r:{item.edge_type}]->(b) "
                "SET r += $props "
                "RETURN count(r) AS cnt",
                gid=pot_id,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                props=props,
            )
            rec = await res.single()
            await res.consume()
            count += int(rec["cnt"]) if rec is not None else 0
    return count


async def delete_edges_async(
    driver: Any,
    pot_id: str,
    items: list[EdgeDelete],
    provenance: ProvenanceRef,
) -> int:
    if not items:
        return 0
    count = 0
    deleted_by = provenance.source_event_id
    deleted_at = _utc_now_iso()
    async with driver.session() as session:
        for item in items:
            if not _is_safe_cypher_identifier(item.edge_type):
                continue
            res = await session.run(
                f"MATCH (a:Entity {{group_id: $gid, entity_key: $from_key}})"  # pyright: ignore[reportArgumentType]
                f"-[r:{item.edge_type}]->"
                f"(b:Entity {{group_id: $gid, entity_key: $to_key}}) "
                "SET r.prov_deleted_by = $deleted_by, "
                "    r.prov_deleted_at = $deleted_at "
                "WITH r DELETE r RETURN count(r) AS cnt",
                gid=pot_id,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                deleted_by=deleted_by,
                deleted_at=deleted_at,
            )
            rec = await res.single()
            await res.consume()
            count += int(rec["cnt"]) if rec is not None else 0
    return count


async def apply_invalidations_async(
    driver: Any,
    pot_id: str,
    items: list[InvalidationOp],
    provenance: ProvenanceRef,
) -> int:
    if not items:
        return 0
    count = 0
    now_iso = _utc_now_iso()
    prov_props = provenance.to_properties()
    async with driver.session() as session:
        for item in items:
            valid_to = item.valid_to or now_iso
            inv_props: dict[str, Any] = dict(prov_props)
            inv_props["valid_to"] = valid_to
            inv_props["invalidation_reason"] = item.reason
            inv_props["invalidated_by"] = provenance.source_event_id
            inv_props["prov_valid_to"] = valid_to

            if item.target_entity_key:
                res = await session.run(
                    "MATCH (e:Entity {group_id: $gid, entity_key: $key}) "
                    "SET e += $props "
                    "RETURN count(e) AS cnt",
                    gid=pot_id,
                    key=item.target_entity_key,
                    props=inv_props,
                )
                rec = await res.single()
                await res.consume()
                matched = int(rec["cnt"]) if rec is not None else 0
                if matched and item.superseded_by_key:
                    await session.run(
                        "MATCH (new:Entity {group_id: $gid, entity_key: $new_key}) "
                        "MATCH (old:Entity {group_id: $gid, entity_key: $old_key}) "
                        "MERGE (new)-[r:SUPERSEDES]->(old) "
                        "SET r.reason = $reason, r.superseded_at = $valid_to",
                        gid=pot_id,
                        new_key=item.superseded_by_key,
                        old_key=item.target_entity_key,
                        reason=item.reason,
                        valid_to=valid_to,
                    )
                count += matched
            elif item.target_edge:
                edge_type, from_key, to_key = item.target_edge
                if not _is_safe_cypher_identifier(edge_type):
                    continue
                res = await session.run(
                    f"MATCH (a:Entity {{group_id: $gid, entity_key: $from_key}})"  # pyright: ignore[reportArgumentType]
                    f"-[r:{edge_type}]->"
                    f"(b:Entity {{group_id: $gid, entity_key: $to_key}}) "
                    "SET r += $props "
                    "RETURN count(r) AS cnt",
                    gid=pot_id,
                    from_key=from_key,
                    to_key=to_key,
                    props=inv_props,
                )
                rec = await res.single()
                await res.consume()
                matched = int(rec["cnt"]) if rec is not None else 0
                if matched and item.superseded_by_key:
                    await session.run(
                        "MATCH (new:Entity {group_id: $gid, entity_key: $new_key}) "
                        "MATCH (old:Entity {group_id: $gid, entity_key: $to_key}) "
                        "MERGE (new)-[r:SUPERSEDES]->(old) "
                        "SET r.reason = $reason, r.superseded_at = $valid_to",
                        gid=pot_id,
                        new_key=item.superseded_by_key,
                        to_key=to_key,
                        reason=item.reason,
                        valid_to=valid_to,
                    )
                count += matched
    return count
