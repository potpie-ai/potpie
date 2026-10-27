"""Current physical-record identity and transaction inventory invariants."""

from __future__ import annotations


from collections.abc import Mapping


from dataclasses import asdict, dataclass, field


from datetime import datetime


import hashlib


import json


import math


from typing import Any


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
class StoredRecord:
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

    def __post_init__(self) -> None:
        if self.kind not in {"entity", "claim"} or not self.pot_id:
            raise ValueError("invalid stored record scope")
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


def validate_entity_identity(
    existing: Mapping[str, Any], incoming: Mapping[str, Any]
) -> None:
    """Ordinary updates cannot assign or rewrite an existing physical identity."""
    if not isinstance(existing.get("record_id"), str) or not existing["record_id"]:
        raise ValueError("invalid stored entity: existing entity has no record_id")
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


def validate_alias_ownership(records: tuple[StoredRecord, ...]) -> None:
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


def validate_current_record(record: StoredRecord) -> None:
    """Refuse incomplete or malformed current records without upgrading them."""
    from .claim_codec import stored_temporal_diagnostics

    if record.issues:
        raise ValueError("; ".join(record.issues))
    if not record.value("record_id"):
        raise ValueError("stored record has no record_id")
    diagnostics = stored_temporal_diagnostics(
        record.properties, canonical_fields=record.fields or None
    )
    if diagnostics:
        raise ValueError(f"invalid stored timestamps: {diagnostics}")
    if record.kind == "claim" and (
        record.subject_pot_id != record.pot_id or record.object_pot_id != record.pot_id
    ):
        raise ValueError("stored claim has missing or foreign endpoints")
