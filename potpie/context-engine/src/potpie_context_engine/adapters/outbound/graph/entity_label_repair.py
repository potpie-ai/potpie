"""Detect and repair canonical entity labels that conflict with entity keys."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from potpie_context_engine.core.entity_canonicalization import coherent_entity_labels
from potpie_context_engine.core.ontology import ENTITY_TYPES, EntityTypeSpec

ENTITY_LABEL_TARGET = "entity_labels"
ENTITY_LABEL_TARGETS = frozenset(
    {ENTITY_LABEL_TARGET, "entity-labels", "labels", "label"}
)
ENTITY_LABEL_REPAIR_LIMIT = 100_000

ENTITY_LABEL_SCAN_CYPHER = """
MATCH (e:Entity {group_id: $gid})
WHERE e.entity_key > $after
RETURN e.entity_key AS key, labels(e) AS labels
ORDER BY e.entity_key
LIMIT $limit
"""


def wants_entity_label_repair(targets: Sequence[str] = ()) -> bool:
    """Return true when a repair invocation should normalize entity labels."""
    if not targets:
        return True
    return any(t.strip().lower() in ENTITY_LABEL_TARGETS for t in targets)


def repaired_entity_labels(
    entity_key: str,
    raw_labels: Iterable[str],
    *,
    entity_types: Mapping[str, EntityTypeSpec] = ENTITY_TYPES,
) -> tuple[str, ...] | None:
    """Return coherent canonical labels, or ``None`` when no repair is needed."""
    labels = tuple(dict.fromkeys(str(label) for label in raw_labels))
    fixed = coherent_entity_labels(
        entity_key,
        labels,
        entity_types=entity_types,
    )
    if set(fixed) == set(labels):
        return None
    return fixed


def canonical_label_changes(
    current: Iterable[str],
    fixed: Iterable[str],
    *,
    entity_types: Mapping[str, EntityTypeSpec] = ENTITY_TYPES,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return safe canonical labels to remove and add."""
    current_set = set(current)
    fixed_set = set(fixed)
    canonical_labels = set(entity_types)
    remove = tuple(sorted((current_set - fixed_set) & canonical_labels))
    add = tuple(sorted((fixed_set - current_set) & canonical_labels))
    return remove, add


__all__ = [
    "ENTITY_LABEL_REPAIR_LIMIT",
    "ENTITY_LABEL_SCAN_CYPHER",
    "ENTITY_LABEL_TARGET",
    "canonical_label_changes",
    "repaired_entity_labels",
    "wants_entity_label_repair",
]
