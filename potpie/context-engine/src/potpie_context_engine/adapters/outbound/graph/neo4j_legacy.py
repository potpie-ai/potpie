"""Lossless Neo4j inventory and exact repairs inside the mutation transaction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.legacy_records import (
    LegacyInventory,
    LegacyRecord,
    RecordRepair,
    match_record_repairs,
    resolve_entity_identity,
    validate_alias_ownership,
    validate_repair_limits,
)
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.reconciliation import MutationResult, MutationSummary

from .native_legacy import CLAIM_INVENTORY, ENTITY_INVENTORY, inventory_record
from .neo4j_execution_state import read_execution_state


@dataclass(frozen=True)
class LocatedRecord:
    locator: str
    record: LegacyRecord


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
                raise MutationPersistenceError("invalid native legacy record locator")
            records.append(
                LocatedRecord(
                    row["locator"], inventory_record(row, pot_id=pot_id, kind=kind)
                )
            )
    return tuple(records)


async def legacy_inventory(writer: Any, pot_id: str) -> LegacyInventory:
    from .cypher import _require_valid_pot_id

    _require_valid_pot_id(pot_id)
    driver = writer._new_driver()
    if driver is None:
        raise RuntimeError("neo4j_unavailable")
    try:
        async with driver.session() as session:

            async def read(transaction):
                before = await read_execution_state(transaction, pot_id)
                located = await read_records(transaction, pot_id)
                after = await read_execution_state(transaction, pot_id)
                if before.version != after.version:
                    raise GraphMutationVersionConflict(
                        expected=before.version, current=after.version
                    )
                return LegacyInventory(
                    records=tuple(
                        sorted(
                            (item.record for item in located),
                            key=lambda record: (
                                record.kind,
                                record.fingerprint,
                            ),
                        )
                    )
                )

            return await session.execute_write(read)
    finally:
        await driver.close()


async def apply_record_repairs(
    transaction: Any,
    effects: tuple[RecordRepair, ...],
    *,
    pot_id: str,
    mutation_id: str,
) -> tuple[MutationResult, tuple[str, ...]]:
    located = await read_records(transaction, pot_id)
    inventory = LegacyInventory(records=tuple(item.record for item in located))
    validate_repair_limits(
        effects,
        max_records=inventory.max_repair_records,
        max_bytes=inventory.max_repair_bytes,
    )
    selected = match_record_repairs(
        tuple(item.record for item in located), effects, pot_id=pot_id
    )
    for kind in ("entity", "claim"):
        patches = [
            {
                "locator": located[index].locator,
                "updates": {**effect.field_updates, **effect.property_updates},
            }
            for index, effect in zip(selected, effects, strict=True)
            if effect.before.kind == kind
        ]
        if patches:
            await patch_records(transaction, kind, patches)
    entities = sum(effect.before.kind == "entity" for effect in effects)
    claims = len(effects) - entities
    return MutationResult(
        ok=True,
        mutation_id=mutation_id,
        mutation_summary=MutationSummary(),
        administrative_summary={
            "repaired_entities": entities,
            "repaired_claims": claims,
        },
    ), tuple(
        effect.after().value("record_id")
        for effect in effects
        if effect.before.kind == "claim"
    )


async def patch_records(
    transaction: Any, kind: str, patches: list[dict[str, Any]]
) -> None:
    variable, pattern = ("e", "(e)") if kind == "entity" else ("r", "()-[r]->()")
    result = await transaction.run(
        f"UNWIND $patches AS patch MATCH {pattern} WHERE elementId({variable})=patch.locator "
        f"SET {variable} += patch.updates RETURN count({variable}) AS count",
        patches=patches,
    )
    row = await result.single()
    await result.consume()
    if row is None or row["count"] != len(patches):
        raise MutationPersistenceError(
            "legacy repair target changed inside its transaction"
        )


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
                "migration_required: existing entity is missing its Entity label"
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
                LegacyRecord(
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
