"""Repair helpers for entity ``summary`` / ``description`` metadata."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from domain.graph_entity_summary import normalize_entity_properties

ENTITY_SUMMARY_TARGET = "entity_summaries"
ENTITY_SUMMARY_TARGETS = frozenset(
    {ENTITY_SUMMARY_TARGET, "entity_summary", "summaries", "summary"}
)

ENTITY_SUMMARY_SCAN_CYPHER = """
MATCH (e:Entity {group_id: $gid})
RETURN e.entity_key AS key, properties(e) AS props
LIMIT $limit
"""

ENTITY_SUMMARY_UPDATE_CYPHER = """
MATCH (e:Entity {group_id: $gid, entity_key: $key})
SET e += $props
RETURN count(e) AS cnt
"""

ENTITY_SUMMARY_REPAIR_LIMIT = 100_000


def wants_entity_summary_repair(targets: Sequence[str] = ()) -> bool:
    """Return true when a repair invocation should backfill entity summaries."""
    if not targets:
        return True
    return any(t.strip().lower() in ENTITY_SUMMARY_TARGETS for t in targets)


def repaired_entity_properties(
    entity_key: str, raw_props: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Return normalized properties when an entity needs summary repair.

    The repair derives only from existing node metadata and the stable
    ``entity_key``. It does not inspect repository files or invent repo facts.
    """
    props = dict(raw_props or {})
    normalized = normalize_entity_properties(props, entity_key=entity_key)
    for key in ("name", "summary", "description"):
        if normalized.get(key) != props.get(key):
            return normalized
    return None


__all__ = [
    "ENTITY_SUMMARY_REPAIR_LIMIT",
    "ENTITY_SUMMARY_SCAN_CYPHER",
    "ENTITY_SUMMARY_TARGET",
    "ENTITY_SUMMARY_UPDATE_CYPHER",
    "repaired_entity_properties",
    "wants_entity_summary_repair",
]
