"""Shared raw inventory queries and decoding for native graph transactions.

Adapters choose an ephemeral locator expression for their active transaction.
Only the decoded raw records cross the preparation/persistence boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from potpie_context_engine.core.record_identity import StoredRecord, raw_properties

ENTITY_INVENTORY = (
    "MATCH (e) WHERE e.group_id = $pot "
    "RETURN {locator} AS locator, e.entity_key AS key, labels(e) AS labels, "
    "properties(e) AS props"
)
CLAIM_INVENTORY = (
    "MATCH (a)-[r]->(b) "
    "WHERE r.group_id = $pot OR a.group_id = $pot OR b.group_id = $pot "
    "RETURN {locator} AS locator, properties(r) AS props, "
    "a.entity_key AS subject_key, b.entity_key AS object_key, "
    "a.group_id AS subject_pot_id, b.group_id AS object_pot_id, "
    "type(r) AS relationship_type"
)


def inventory_record(raw: Mapping[str, Any], *, pot_id: str, kind: str) -> StoredRecord:
    props, types, issues = raw_properties(raw["props"])
    if props.get("group_id") != pot_id:
        issues += ("record has a different or missing pot scope",)
    return StoredRecord(
        pot_id=pot_id,
        kind=kind,
        key=(raw.get("key") if kind == "entity" else props.get("claim_key")) or "",
        properties=props,
        property_types=types,
        labels=tuple(sorted(raw.get("labels") or ())),
        subject_key=raw.get("subject_key"),
        object_key=raw.get("object_key"),
        relationship_type=raw.get("relationship_type"),
        subject_pot_id=raw.get("subject_pot_id"),
        object_pot_id=raw.get("object_pot_id"),
        issues=issues,
    )
