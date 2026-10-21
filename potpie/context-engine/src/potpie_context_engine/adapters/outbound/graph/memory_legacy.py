"""Lossless memory inventory and exact repair effects in the staged writer."""

from __future__ import annotations

from copy import deepcopy

from potpie_context_engine.core.legacy_records import (
    LegacyInventory,
    LegacyRecord,
    MAX_REPAIR_RECORDS,
    match_record_repairs,
    raw_properties,
    validate_alias_ownership,
    validate_repair_limits,
)
from potpie_context_engine.core.reconciliation import MutationResult, MutationSummary


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
                LegacyRecord(
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
                LegacyRecord(
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


def legacy_inventory(writer, pot_id: str) -> LegacyInventory:
    state = writer.journal_data.get("states", {}).get(pot_id)
    return LegacyInventory(
        records=tuple(record for record, _ in _inventory_entries(writer.store, pot_id)),
        journal_active=state is not None,
        max_repair_records=state.limits.max_records if state else MAX_REPAIR_RECORDS,
        max_repair_bytes=state.limits.max_bytes if state else 8_000_000,
    )


def validate_store_aliases(store, pot_id: str) -> None:
    validate_alias_ownership(
        tuple(record for record, _ in _inventory_entries(store, pot_id))
    )


def apply_record_repairs(writer, effects, *, pot_id: str, mutation_id: str):
    """Validate all raw before-images before updating any staged record."""
    from .backends.in_memory_backend import _row_from_dict, _stored_row_to_dict

    entries = _inventory_entries(writer.store, pot_id)
    state = writer.journal_data.get("states", {}).get(pot_id)
    validate_repair_limits(
        effects,
        max_records=state.limits.max_records if state else MAX_REPAIR_RECORDS,
        max_bytes=state.limits.max_bytes if state else 8_000_000,
    )
    indexes = match_record_repairs(
        tuple(record for record, _ in entries), effects, pot_id=pot_id
    )
    replacements = []
    for index, effect in zip(indexes, effects, strict=True):
        before, locator = entries[index]
        if before.kind == "entity":
            properties = deepcopy(
                writer.store.entity_property_index.get((pot_id, locator), {})
            )
            properties.update(deepcopy(dict(effect.property_updates)))
            replacements.append((before.kind, locator, properties))
        else:
            raw = deepcopy(_stored_row_to_dict(writer.store.rows[locator]))
            raw.update(deepcopy(dict(effect.field_updates)))
            raw["properties"].update(deepcopy(dict(effect.property_updates)))
            replacements.append((before.kind, locator, _row_from_dict(pot_id, raw)))
    for kind, locator, value in replacements:
        if kind == "entity":
            writer.store.entity_property_index[(pot_id, locator)] = value
        else:
            writer.store.rows[locator] = value
    entities = sum(kind == "entity" for kind, _, _ in replacements)
    return MutationResult(
        ok=True,
        mutation_id=mutation_id,
        mutation_summary=MutationSummary(),
        administrative_summary={
            "repaired_records": len(replacements),
            "repaired_entities": entities,
            "repaired_claims": len(replacements) - entities,
        },
    )
