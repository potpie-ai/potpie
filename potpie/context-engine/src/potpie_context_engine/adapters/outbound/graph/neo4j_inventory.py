"""Transaction-scoped inventory and immutable identities for current Neo4j writes."""

from __future__ import annotations


from dataclasses import dataclass, replace


from typing import Any


from potpie_context_engine.core.record_identity import (
    StoredRecord,
    resolve_entity_identity,
    validate_alias_ownership,
)


from potpie_context_engine.core.mutation_execution import MutationPersistenceError


from .native_inventory import CLAIM_INVENTORY, ENTITY_INVENTORY, inventory_record


@dataclass(frozen=True)
class LocatedRecord:
    locator: str
    record: StoredRecord


async def read_records(transaction: Any, pot_id: str) -> tuple[LocatedRecord, ...]:
    """Keep database locators local to the transaction that reads them."""
    records = []
    for kind, template, locator in (
        ("entity", ENTITY_INVENTORY, "elementId(e)"),
        ("claim", CLAIM_INVENTORY, "elementId(r)"),
    ):
        result = await transaction.run(template.format(locator=locator), pot=pot_id)
        rows = await result.data()
        await result.consume()
        for row in rows:
            if not isinstance(row["locator"], str) or not row["locator"]:
                raise MutationPersistenceError("invalid native record locator")
            records.append(
                LocatedRecord(
                    row["locator"], inventory_record(row, pot_id=pot_id, kind=kind)
                )
            )
    return tuple(records)


async def prepare_entities(
    transaction: Any, pot_id: str, items: list[Any], provenance: Any
) -> list[Any]:
    """Resolve immutable entity IDs against the same inventory used by publication."""
    located = await read_records(transaction, pot_id)
    staged = [item.record for item in located]
    resolved = []
    for item in items:
        matches = [
            index
            for index, record in enumerate(staged)
            if record.kind == "entity" and record.key == item.entity_key
        ]
        if len(matches) > 1:
            raise ValueError("entity target is ambiguous")
        index = matches[0] if matches else None
        previous = staged[index] if index is not None else None
        if previous is not None and "Entity" not in previous.labels:
            raise ValueError(
                "invalid stored entity: existing entity is missing its Entity label"
            )
        properties = resolve_entity_identity(
            previous.properties if previous is not None else None,
            item.properties,
            pot_id=pot_id,
            mutation_id=provenance.mutation_id,
            entity_key=item.entity_key,
        )
        resolved.append(replace(item, properties=properties))
        if previous is None:
            staged.append(
                StoredRecord(
                    pot_id, "entity", item.entity_key, properties, labels=item.labels
                )
            )
        else:
            staged[index] = replace(
                previous, properties={**previous.properties, **properties}
            )
    validate_alias_ownership(tuple(staged))
    return resolved


async def validate_aliases(transaction: Any, pot_id: str) -> None:
    validate_alias_ownership(
        tuple(item.record for item in await read_records(transaction, pot_id))
    )
