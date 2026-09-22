"""Shared conversion and file helpers for graph-native snapshot adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Iterable, Mapping

from potpie_context_core.graph_snapshot import build_snapshot_payload
from potpie_context_core.ports.graph.snapshot import SnapshotManifest

_SNAPSHOT_PROPERTIES = "__potpie_snapshot_properties_v2"
_SNAPSHOT_CLAIM_FIELDS = "__potpie_snapshot_claim_fields_v2"


def validate_native_snapshot_properties(payload: Mapping[str, Any]) -> None:
    reserved = {_SNAPSHOT_PROPERTIES, _SNAPSHOT_CLAIM_FIELDS}
    for collection in ("entities", "claims"):
        for row in payload.get(collection, ()):
            properties = row.get("properties") or {}
            collision = reserved & properties.keys()
            if collision:
                raise ValueError(
                    f"snapshot properties may not use reserved key {sorted(collision)[0]!r}"
                )


def _restore_snapshot_properties(props: dict[str, Any]) -> dict[str, Any]:
    encoded = props.pop(_SNAPSHOT_PROPERTIES, None)
    if not isinstance(encoded, str):
        return props
    try:
        markers = json.loads(encoded)
    except json.JSONDecodeError:
        return props
    if not isinstance(markers, dict):
        return props
    for key, stored_encoding in markers.items():
        if stored_encoding is None and key not in props:
            props[key] = None
            continue
        # A later ordinary graph mutation wins. Decode only while the native
        # value still equals the exact representation written by snapshot
        # import; this preserves JSON-looking strings and reflects patches.
        if key not in props or props[key] != stored_encoding:
            continue
        try:
            props[key] = json.loads(stored_encoding)
        except (TypeError, json.JSONDecodeError):
            continue
    return props


def _encoded_property_markers(properties: Mapping[str, Any]) -> dict[str, Any]:
    markers: dict[str, Any] = {}
    for key, value in properties.items():
        if value is None:
            markers[key] = None
            continue
        if isinstance(value, dict):
            markers[key] = json.dumps(value, default=str, sort_keys=True)
            continue
        if isinstance(value, list):
            kinds = {bool if isinstance(item, bool) else type(item) for item in value}
            native = all(isinstance(item, (bool, int, float, str)) for item in value)
            if not native or len(kinds) > 1:
                markers[key] = json.dumps(value, default=str, sort_keys=True)
    return markers


def _stamp_property_markers(props: dict[str, Any], original: Mapping[str, Any]) -> None:
    if _SNAPSHOT_PROPERTIES in original:
        raise ValueError(
            f"snapshot properties may not use reserved key {_SNAPSHOT_PROPERTIES!r}"
        )
    markers = _encoded_property_markers(original)
    if markers:
        props[_SNAPSHOT_PROPERTIES] = json.dumps(
            markers, allow_nan=False, separators=(",", ":"), sort_keys=True
        )


def entity_row(key: Any, labels: Iterable[Any], props: Mapping[str, Any]) -> dict[str, Any]:
    properties = dict(props)
    properties.pop("entity_key", None)
    properties.pop("group_id", None)
    properties = _restore_snapshot_properties(properties)
    return {
        "key": str(key),
        "labels": sorted(str(label) for label in labels),
        "properties": properties,
    }


def claim_row(props: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(props)
    raw.pop("group_id", None)
    predicate = raw.pop("name", None)
    subject = raw.pop("subject_key", None)
    object_ = raw.pop("object_key", None)
    encoded_fields = raw.pop(_SNAPSHOT_CLAIM_FIELDS, None)
    if isinstance(encoded_fields, str):
        try:
            original_fields = json.loads(encoded_fields)
        except json.JSONDecodeError:
            original_fields = {}
        original_embedding = original_fields.get("fact_embedding")
        stored_embedding = raw.get("fact_embedding")
        if _same_numeric_vector(stored_embedding, original_embedding):
            raw["fact_embedding"] = original_embedding
    raw = _restore_snapshot_properties(raw)
    fields = {
        name: raw.pop(name, None)
        for name in (
            "valid_at", "invalid_at", "evidence_strength", "source_system",
            "source_ref", "fact", "fact_embedding", "claim_key", "subgraph",
            "truth", "confidence", "description", "environment", "observed_at",
            "valid_until", "mutation_id", "source_refs", "evidence",
            "graph_contract_version", "ontology_version",
        )
    }
    for name in ("source_refs", "evidence"):
        value = fields[name]
        if value is None:
            fields[name] = []
            continue
        if isinstance(value, str):
            try:
                fields[name] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return {
        "predicate": predicate,
        "subject_key": subject,
        "object_key": object_,
        **fields,
        "properties": raw,
    }


def payload_from_rows(*, pot_id: str, entity_rows: Iterable[Mapping[str, Any]], claim_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return build_snapshot_payload(pot_id=pot_id, entities=entity_rows, claims=claim_rows)


def write_payload(destination: str, payload: Mapping[str, Any]) -> SnapshotManifest:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return manifest(str(path), payload)


def read_payload(source: str) -> Mapping[str, Any]:
    with Path(source).expanduser().open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("snapshot document must be a JSON object")
    return value


def manifest(location: str, payload: Mapping[str, Any]) -> SnapshotManifest:
    return SnapshotManifest(
        pot_id=str(payload["pot_id"]),
        location=location,
        format_version=str(payload["format_version"]),
        entity_count=len(payload.get("entities", ())),
        claim_count=len(payload.get("claims", ())),
    )


def imported_manifest(location: str, payload: Mapping[str, Any], *, pot_id: str) -> SnapshotManifest:
    return SnapshotManifest(
        pot_id=pot_id,
        location=location,
        format_version=str(payload["format_version"]),
        entity_count=len(payload.get("entities", ())),
        claim_count=len(payload.get("claims", ())),
    )


def stored_entity(entity: Mapping[str, Any], pot_id: str) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    props = dict(entity.get("properties") or {})
    _stamp_property_markers(props, dict(entity.get("properties") or {}))
    props.update(group_id=pot_id, entity_key=str(entity["key"]))
    return str(entity["key"]), tuple(entity.get("labels") or ()), props


def stored_claim(claim: Mapping[str, Any], pot_id: str) -> dict[str, Any]:
    props = dict(claim.get("properties") or {})
    _stamp_property_markers(props, dict(claim.get("properties") or {}))
    for name, value in claim.items():
        if name not in {"properties", "predicate"} and value is not None:
            props[name] = value
    if claim.get("fact_embedding") is not None:
        props[_SNAPSHOT_CLAIM_FIELDS] = json.dumps(
            {"fact_embedding": claim["fact_embedding"]},
            allow_nan=False,
            separators=(",", ":"),
        )
    props.update(group_id=pot_id, name=str(claim["predicate"]))
    return props


def _same_numeric_vector(left: Any, right: Any) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, list):
        return False
    return len(left) == len(right) and all(
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and (
            float(a) == float(b)
            or float(a) == struct.unpack("!f", struct.pack("!f", float(b)))[0]
        )
        for a, b in zip(left, right)
    )
