"""Current in-memory record inventory and physical-identity ownership checks."""

from __future__ import annotations


from potpie_context_engine.core.record_identity import (
    StoredRecord,
    raw_properties,
    validate_alias_ownership,
)


def _inventory_entries(store, pot_id: str):
    from .backends.in_memory_backend import _stored_row_to_dict

    keys = {key for pid, key in store.entity_label_index if pid == pot_id}
    keys.update(key for pid, key in store.entity_property_index if pid == pot_id)
    entries = []
    for key in sorted(keys):
        properties, types, issues = raw_properties(
            store.entity_property_index.get((pot_id, key), {})
        )
        entries.append(
            (
                StoredRecord(
                    pot_id=pot_id,
                    kind="entity",
                    key=key,
                    properties=properties,
                    property_types=types,
                    labels=tuple(
                        sorted(store.entity_label_index.get((pot_id, key), ()))
                    ),
                    issues=issues,
                ),
                key,
            )
        )
    for index, row in enumerate(store.rows):
        if row.pot_id != pot_id:
            continue
        raw = _stored_row_to_dict(row)
        properties, property_types, property_issues = raw_properties(
            raw.pop("properties")
        )
        fields, field_types, field_issues = raw_properties(raw)
        entries.append(
            (
                StoredRecord(
                    pot_id=pot_id,
                    kind="claim",
                    key=row.claim_key or "",
                    properties=properties,
                    fields=fields,
                    property_types=property_types,
                    field_types=field_types,
                    subject_key=row.subject_key,
                    object_key=row.object_key,
                    subject_pot_id=pot_id if row.subject_key in keys else None,
                    object_pot_id=pot_id if row.object_key in keys else None,
                    relationship_type="RELATES_TO",
                    issues=property_issues + field_issues,
                ),
                index,
            )
        )
    return entries


def validate_store_aliases(store, pot_id: str) -> None:
    validate_alias_ownership(
        tuple(record for record, _ in _inventory_entries(store, pot_id))
    )
