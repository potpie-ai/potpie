"""Lossless record inventory and narrowly scoped legacy repair effects.

Inventories retain storage namespaces and temporal types. They never infer a
missing identity or interval. A repair matches the entire before-image inside
the backend transaction; database-local locators never enter a prepared plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import hashlib
import json
import math
from typing import Any

from .claim_codec import ENTITY_TEMPORAL_FIELDS

REPAIR_SCHEMA_VERSION = 1
MAX_REPAIR_RECORDS = 5000
REPAIR_ENVELOPE_BYTES = 16384
REPAIR_TIME_FIELDS = ENTITY_TEMPORAL_FIELDS
REPAIR_PROPERTIES = REPAIR_TIME_FIELDS | {"record_id"}


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported stored value type: {type(value).__name__}")


def raw_properties(
    properties: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str], tuple[str, ...]]:
    """Encode exact native temporal values, reporting every unsupported value.

    Type markers live outside user properties so a literal user mapping cannot
    impersonate a typed timestamp. Unsupported values make the inventory unsafe
    to apply; their diagnostic placeholder is never a repair target.
    """
    values, types, issues = {}, {}, []
    for key, value in properties.items():
        try:
            if isinstance(value, datetime):
                values[key], types[key] = value.isoformat(), "datetime"
            elif type(value).__module__ == "neo4j.time" and callable(
                getattr(value, "iso_format", None)
            ):
                values[key], types[key] = value.iso_format(), type(value).__name__
            else:
                values[key] = _json_value(value)
        except ValueError as exc:
            values[key] = None
            issues.append(f"{key}: {exc}")
    return values, types, tuple(issues)


@dataclass(frozen=True, slots=True)
class LegacyRecord:
    pot_id: str
    kind: str
    key: str
    properties: Mapping[str, Any]
    fields: Mapping[str, Any] = field(default_factory=dict)
    labels: tuple[str, ...] = ()
    subject_key: str | None = None
    object_key: str | None = None
    relationship_type: str | None = None
    subject_pot_id: str | None = None
    object_pot_id: str | None = None
    property_types: Mapping[str, str] = field(default_factory=dict)
    field_types: Mapping[str, str] = field(default_factory=dict)
    issues: tuple[str, ...] = ()
    schema_version: int = REPAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != REPAIR_SCHEMA_VERSION
        ):
            raise ValueError("unsupported legacy record schema")
        if self.kind not in {"entity", "claim"} or not self.pot_id:
            raise ValueError("invalid legacy record scope")
        for namespace in (self.fields, self.properties):
            _json_value(namespace)
        for types, values in (
            (self.field_types, self.fields),
            (self.property_types, self.properties),
        ):
            if not types.keys() <= values.keys() or any(
                not isinstance(kind, str) or not kind for kind in types.values()
            ):
                raise ValueError("invalid raw property type metadata")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LegacyRecord:
        data = dict(raw)
        data["labels"] = tuple(data.get("labels", ()))
        data["issues"] = tuple(data.get("issues", ()))
        return cls(**data)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(encoded.encode(), digest_size=20).hexdigest()

    def value(self, name: str) -> Any:
        if name in {"record_id", "uuid"}:
            for values in (self.fields, self.properties):
                value = values.get(name)
                if isinstance(value, str) and value.strip():
                    return value
        return self.fields.get(name, self.properties.get(name))


@dataclass(frozen=True, slots=True)
class RecordRepair:
    before: LegacyRecord
    field_updates: Mapping[str, Any] = field(default_factory=dict)
    property_updates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.before.issues or not (self.field_updates or self.property_updates):
            raise ValueError("repair requires a supported before-image and a patch")
        for updates in (self.field_updates, self.property_updates):
            if not updates.keys() <= REPAIR_PROPERTIES:
                raise ValueError(
                    "legacy repair may only backfill IDs or normalize times"
                )
            _json_value(updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "field_updates": _json_value(self.field_updates),
            "property_updates": _json_value(self.property_updates),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RecordRepair:
        if set(raw) != {"before", "field_updates", "property_updates"}:
            raise ValueError("invalid record repair fields")
        return cls(
            LegacyRecord.from_dict(raw["before"]),
            raw["field_updates"],
            raw["property_updates"],
        )

    def after(self) -> LegacyRecord:
        return replace(
            self.before,
            fields={**self.before.fields, **self.field_updates},
            properties={**self.before.properties, **self.property_updates},
            field_types={
                key: value
                for key, value in self.before.field_types.items()
                if key not in self.field_updates
            },
            property_types={
                key: value
                for key, value in self.before.property_types.items()
                if key not in self.property_updates
            },
        )


@dataclass(frozen=True, slots=True)
class LegacyInventory:
    records: tuple[LegacyRecord, ...]
    issues: tuple[str, ...] = ()
    journal_active: bool = False
    max_repair_records: int = MAX_REPAIR_RECORDS
    max_repair_bytes: int = 8_000_000


def validate_entity_identity(
    existing: Mapping[str, Any], incoming: Mapping[str, Any]
) -> None:
    """Ordinary updates cannot assign or rewrite an existing physical identity."""
    if not isinstance(existing.get("record_id"), str) or not existing["record_id"]:
        raise ValueError("migration_required: existing entity has no record_id")
    for name in ("record_id", "uuid"):
        if name in incoming and incoming[name] != existing.get(name):
            raise ValueError(f"immutable entity identity: {name}")


def resolve_entity_identity(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
    *,
    pot_id: str,
    mutation_id: str,
    entity_key: str,
) -> dict[str, Any]:
    """Choose IDs once for new entities; preserve both aliases on existing ones.

    The caller must also validate alias ownership over its complete transaction
    inventory before publication. Existing identity-less entities require repair.
    """
    from uuid import NAMESPACE_URL, uuid5

    props = dict(incoming)
    if existing is not None:
        validate_entity_identity(existing, incoming)
        return props
    for name in ("record_id", "uuid"):
        if name in props and (
            not isinstance(props[name], str) or not props[name].strip()
        ):
            raise ValueError(f"invalid new entity identity: {name}")
    props.setdefault(
        "record_id",
        props.get("uuid")
        or str(
            uuid5(
                NAMESPACE_URL,
                json.dumps(["potpie:entity", pot_id, mutation_id, entity_key]),
            )
        ),
    )
    props.setdefault("uuid", props["record_id"])
    return props


def validate_alias_ownership(records: tuple[LegacyRecord, ...]) -> None:
    owners: dict[str, int] = {}
    entity_keys = set()
    for index, record in enumerate(records):
        if record.kind == "entity" and isinstance(record.key, str) and record.key:
            key = (record.pot_id, record.key)
            if key in entity_keys:
                raise ValueError("logical entity key is shared by multiple records")
            entity_keys.add(key)
        for values in (record.fields, record.properties):
            for name in ("record_id", "uuid"):
                identity = values.get(name)
                if identity in (None, ""):
                    continue
                if not isinstance(identity, str) or not identity.strip():
                    raise ValueError("invalid physical record identity")
                if identity in owners and owners[identity] != index:
                    raise ValueError("physical identity is shared by multiple records")
                owners[identity] = index


def match_record_repairs(
    records: tuple[LegacyRecord, ...],
    effects: tuple[RecordRepair, ...],
    *,
    pot_id: str,
) -> tuple[int, ...]:
    """Resolve exact before-images and validate the final global alias ownership."""
    from .legacy_repair_policy import validate_record_repair

    selected, after = [], list(records)
    by_fingerprint: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_fingerprint.setdefault(record.fingerprint, []).append(index)
    for effect in effects:
        validate_record_repair(effect, pot_id=pot_id)
        matches = by_fingerprint.get(effect.before.fingerprint, [])
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError("repair target is missing, changed, or ambiguous")
        index = matches[0]
        selected.append(index)
        after[index] = effect.after()
    validate_alias_ownership(tuple(after))
    return tuple(selected)


def repair_effect_bytes(effect: RecordRepair) -> int:
    """Conservative before/after, JSON escaping and journal envelope allowance."""
    return 4 * len(json.dumps(effect.to_dict(), ensure_ascii=True).encode()) + 4096


def validate_repair_limits(
    effects: tuple[RecordRepair, ...], *, max_records: int, max_bytes: int
) -> None:
    if (
        type(max_records) is not int
        or type(max_bytes) is not int
        or max_records < 1
        or max_bytes < REPAIR_ENVELOPE_BYTES
    ):
        raise ValueError("backend has unsupported repair limits")
    if len(effects) > min(max_records, MAX_REPAIR_RECORDS) or (
        REPAIR_ENVELOPE_BYTES + sum(repair_effect_bytes(effect) for effect in effects)
        > max_bytes
    ):
        raise ValueError("legacy repair batch exceeds backend transaction limits")
