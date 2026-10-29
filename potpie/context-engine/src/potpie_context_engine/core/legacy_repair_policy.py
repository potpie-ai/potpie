"""Pure, conservative legacy repair policy shared by preview and execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .claim_codec import normalize_timestamp, stored_temporal_diagnostics
from .legacy_records import (
    LegacyInventory,
    LegacyRecord,
    RecordRepair,
    REPAIR_TIME_FIELDS,
)


@dataclass(frozen=True, slots=True)
class LegacyFinding:
    code: str
    detail: str
    record_fingerprint: str | None = None
    blocking: bool = True


def _identity(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _record_issues(record: LegacyRecord) -> tuple[str, ...]:
    issues = list(record.issues)
    for namespace in (record.fields, record.properties):
        for name in ("pot_id", "group_id"):
            if name in namespace and namespace[name] != record.pot_id:
                issues.append(f"stored {name} contradicts the record pot")
    if record.kind == "entity":
        if not isinstance(record.key, str) or not record.key.strip():
            issues.append("record has no valid string logical key")
        if "Entity" not in record.labels:
            issues.append("entity is missing its Entity label")
    else:
        key = record.value("claim_key")
        if key not in (None, "") and not isinstance(key, str):
            issues.append("claim has a malformed logical key")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                record.subject_key,
                record.object_key,
                record.relationship_type,
            )
        ):
            issues.append("claim has incomplete physical endpoints/type")
        if record.relationship_type != "RELATES_TO":
            issues.append("claim has an unsupported physical relationship type")
        if (record.subject_pot_id, record.object_pot_id) != (
            record.pot_id,
            record.pot_id,
        ):
            issues.append("claim endpoints belong to a different or missing pot")
        predicate = record.value("predicate") or record.value("name")
        if not isinstance(predicate, str) or not predicate.strip():
            issues.append("claim has no predicate")
        for name, actual in (
            ("subject_key", record.subject_key),
            ("object_key", record.object_key),
        ):
            for namespace in (record.fields, record.properties):
                if name in namespace and namespace[name] != actual:
                    issues.append(
                        f"stored {name} differs from the physical relationship"
                    )
        for aliases in (
            ("predicate", "name"),
            ("claim_key",),
            ("subgraph",),
            ("source_system",),
            ("source_ref",),
            ("environment",),
            ("truth",),
        ):
            values = [
                namespace[name]
                for namespace in (record.fields, record.properties)
                for name in aliases
                if name in namespace
            ]
            if values and any(
                type(value) is not type(values[0]) or value != values[0]
                for value in values
            ):
                issues.append(f"stored {'/'.join(aliases)} aliases disagree")
    for name in ("record_id", "uuid"):
        for namespace in (record.fields, record.properties):
            value = namespace.get(name)
            if value is not None and value != "" and not _identity(value):
                issues.append(f"{name} is not a valid identity")
        left, right = record.fields.get(name), record.properties.get(name)
        if left not in (None, "") and right not in (None, "") and left != right:
            issues.append(f"stored {name} aliases disagree")
    for types in (record.field_types, record.property_types):
        if any(kind not in {"datetime", "DateTime"} for kind in types.values()):
            issues.append("unsupported stored native property type")
    temporal = []
    for namespace in (record.properties, record.fields):
        values = {
            name: value
            for name, value in namespace.items()
            if name in REPAIR_TIME_FIELDS
            and value is not None
            and not (record.kind == "entity" and value == "")
        }
        temporal.append(values)
        for name, value in values.items():
            try:
                normalize_timestamp(value, field=name)
            except ValueError as exc:
                issues.append(str(exc))
            # The shared reader diagnostics also detects unsupported precision.
            issues.extend(stored_temporal_diagnostics({"observed_at": value}).values())
    issues.extend(
        stored_temporal_diagnostics(temporal[0], canonical_fields=temporal[1]).values()
    )
    return tuple(dict.fromkeys(issues))


def record_patch(record: LegacyRecord, *, new_record_id: str) -> RecordRepair | None:
    """Compute the only permitted patch. An ID is supplied, never generated here."""
    issues = _record_issues(record)
    if issues:
        raise ValueError("; ".join(issues))
    fields, properties = {}, {}
    for values, types, updates in (
        (record.fields, record.field_types, fields),
        (record.properties, record.property_types, properties),
    ):
        for name, value in values.items():
            if (
                name not in REPAIR_TIME_FIELDS
                or value is None
                or (record.kind == "entity" and value == "")
            ):
                continue
            normalized = normalize_timestamp(value, field=name)
            if normalized != value or name in types:
                updates[name] = normalized
    primary_fields = record.kind == "claim" and bool(record.fields)
    original = record.fields if primary_fields else record.properties
    if not _identity(original.get("record_id")):
        identity = record.value("record_id") or record.value("uuid") or new_record_id
        if not _identity(identity):
            raise ValueError("repair requires a frozen replacement record_id")
        destination = fields if primary_fields else properties
        destination["record_id"] = identity
    if not fields and not properties:
        return None
    return RecordRepair(record, fields, properties)


def validate_record_repair(effect: RecordRepair, *, pot_id: str) -> None:
    if effect.before.pot_id != pot_id:
        raise ValueError("repair target belongs to another pot")
    expected = record_patch(
        effect.before, new_record_id=effect.after().value("record_id")
    )
    if expected != effect:
        raise ValueError("repair differs from exact identity/time normalization policy")


def plan_legacy_repairs(
    inventory: LegacyInventory,
) -> tuple[tuple[RecordRepair, ...], tuple[LegacyFinding, ...]]:
    findings = [
        LegacyFinding("inventory_unavailable", issue) for issue in inventory.issues
    ]
    effects = []
    aliases: dict[str, set[int]] = {}
    entity_keys = Counter(
        (record.pot_id, record.key)
        for record in inventory.records
        if record.kind == "entity" and isinstance(record.key, str)
    )
    fingerprints = Counter(record.fingerprint for record in inventory.records)
    for index, record in enumerate(inventory.records):
        for namespace in (record.fields, record.properties):
            for name in ("record_id", "uuid"):
                value = namespace.get(name)
                if _identity(value):
                    aliases.setdefault(value, set()).add(index)
        issues = list(_record_issues(record))
        if fingerprints[record.fingerprint] > 1:
            issues.append("duplicate indistinguishable physical records")
        if (
            record.kind == "entity"
            and isinstance(record.key, str)
            and entity_keys[(record.pot_id, record.key)] > 1
        ):
            issues.append("logical entity key is shared by multiple records")
        if inventory.journal_active and not _identity(record.value("record_id")):
            issues.append(
                "active journal has an identity-less record; historical identity requires explicit resolution"
            )
        if issues:
            findings.extend(
                LegacyFinding("ambiguous_record", issue, record.fingerprint)
                for issue in dict.fromkeys(issues)
            )
            continue
        effect = record_patch(record, new_record_id=uuid4().hex)
        if effect is not None:
            effects.append(effect)
    for identity, indices in aliases.items():
        if len(indices) > 1:
            findings.append(
                LegacyFinding(
                    "duplicate_identity",
                    f"identity {identity!r} is shared by {len(indices)} records",
                )
            )
    findings.append(
        LegacyFinding(
            "history_unavailable",
            "Previously overwritten or erased claim intervals are unavailable; repair cannot reconstruct them.",
            blocking=False,
        )
    )
    return tuple(effects), tuple(findings)
