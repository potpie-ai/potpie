"""WATCH-scoped inventory and immutable identities for current FalkorDB writes."""

from __future__ import annotations


from collections import Counter


from dataclasses import replace


from potpie_context_engine.core.record_identity import (
    StoredRecord,
    resolve_entity_identity,
    validate_alias_ownership,
)


from .falkordb_journal import _read


from .native_inventory import CLAIM_INVENTORY, ENTITY_INVENTORY, inventory_record


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
                "invalid stored entity: existing entity is missing its Entity label"
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
            else StoredRecord(
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
                StoredRecord(pot_id, "entity", item.entity_key, dict(item.properties))
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
                StoredRecord(
                    pot_id,
                    "claim",
                    write["props"].get("claim_key") or "",
                    write["props"],
                )
            )
    validate_alias_ownership(tuple(final))
