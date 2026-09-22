"""Validation and normalization for portable graph snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from potpie_context_core.graph_contract import evidence_strength_for_truth

SNAPSHOT_FORMAT_VERSION = "2"
RESERVED_SNAPSHOT_PROPERTIES = frozenset({
    "__potpie_snapshot_properties_v2", "__potpie_snapshot_claim_fields_v2",
})

_REQUIRED_CLAIM_STRINGS = ("predicate", "subject_key", "object_key")
_OPTIONAL_CLAIM_STRINGS = (
    "claim_key", "source_system", "source_ref", "fact", "subgraph", "truth",
    "description", "environment", "mutation_id", "graph_contract_version",
    "ontology_version", "valid_at", "invalid_at", "observed_at", "valid_until",
)
_CLAIM_FIELDS = set(_REQUIRED_CLAIM_STRINGS) | set(_OPTIONAL_CLAIM_STRINGS) | {
    "properties", "fact_embedding", "confidence", "source_refs", "evidence",
    "evidence_strength",
}


def _json_copy(value: Any, *, path: str) -> Any:
    def check_keys(item: Any, item_path: str) -> None:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{item_path} object keys must be strings")
            for key, child in item.items():
                check_keys(child, f"{item_path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                check_keys(child, f"{item_path}[{index}]")

    check_keys(value, path)
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain only JSON-safe values") from exc


def snapshot_claim_identity(claim: Mapping[str, Any]) -> str:
    key = claim.get("claim_key")
    if not isinstance(key, str) or not key:
        raise ValueError("claim.claim_key must be a non-empty string")
    return key


def _normalize_entity(raw: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"entities[{index}] must be an object")
    unknown = set(raw) - {"key", "labels", "properties"}
    if unknown:
        raise ValueError(f"entities[{index}] has unknown fields: {sorted(unknown)!r}")
    key = raw.get("key")
    labels = raw.get("labels")
    properties = raw.get("properties")
    if not isinstance(key, str) or not key:
        raise ValueError(f"entities[{index}].key must be a non-empty string")
    if not isinstance(labels, list) or any(not isinstance(x, str) or not x for x in labels):
        raise ValueError(f"entities[{index}].labels must be a list of non-empty strings")
    if len(labels) != len(set(labels)):
        raise ValueError(f"entities[{index}].labels contains duplicates")
    if not isinstance(properties, Mapping):
        raise ValueError(f"entities[{index}].properties must be an object")
    if reserved := RESERVED_SNAPSHOT_PROPERTIES.intersection(properties):
        raise ValueError(
            f"entities[{index}].properties uses reserved key {sorted(reserved)[0]!r}"
        )
    return {
        "key": key,
        "labels": sorted(labels),
        "properties": _json_copy(dict(properties), path=f"entities[{index}].properties"),
    }


def _generated_claim_key(claim: Mapping[str, Any], target_pot_id: str) -> str:
    material = {k: v for k, v in claim.items() if k != "claim_key"}
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
    return f"claim:{target_pot_id}:{digest}"


def _normalize_claim(
    raw: Any, *, index: int, source_pot_id: str, target_pot_id: str, legacy: bool
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"claims[{index}] must be an object")
    unknown = set(raw) - _CLAIM_FIELDS
    if unknown:
        raise ValueError(f"claims[{index}] has unknown fields: {sorted(unknown)!r}")
    claim = _json_copy(dict(raw), path=f"claims[{index}]")
    for field in _REQUIRED_CLAIM_STRINGS:
        if not isinstance(claim.get(field), str) or not claim[field]:
            raise ValueError(f"claims[{index}].{field} must be a non-empty string")
    for field in _OPTIONAL_CLAIM_STRINGS:
        value = claim.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"claims[{index}].{field} must be null or a non-empty string")
    if not isinstance(claim.get("properties", {}), Mapping):
        raise ValueError(f"claims[{index}].properties must be an object")
    if reserved := RESERVED_SNAPSHOT_PROPERTIES.intersection(claim.get("properties", {})):
        raise ValueError(
            f"claims[{index}].properties uses reserved key {sorted(reserved)[0]!r}"
        )
    for field in ("source_refs", "evidence"):
        value = claim.get(field, [])
        if not isinstance(value, list):
            raise ValueError(f"claims[{index}].{field} must be a list")
    if any(not isinstance(x, str) or not x for x in claim.get("source_refs", [])):
        raise ValueError(f"claims[{index}].source_refs must contain non-empty strings")
    if any(not isinstance(x, Mapping) for x in claim.get("evidence", [])):
        raise ValueError(f"claims[{index}].evidence must contain objects")
    embedding = claim.get("fact_embedding")
    if embedding is not None and (
        not isinstance(embedding, list)
        or any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in embedding)
    ):
        raise ValueError(f"claims[{index}].fact_embedding must be null or a numeric list")
    confidence = claim.get("confidence")
    if confidence is not None and (
        not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
    ):
        raise ValueError(f"claims[{index}].confidence must be null or numeric")
    if confidence is not None and not 0 <= confidence <= 1:
        raise ValueError(f"claims[{index}].confidence must be between 0 and 1")
    for field in ("valid_at", "invalid_at", "observed_at", "valid_until"):
        value = claim.get(field)
        if value is not None:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"claims[{index}].{field} must be an ISO timestamp") from exc

    key = claim.get("claim_key")
    if key is None:
        if not legacy:
            raise ValueError(f"claims[{index}].claim_key must be a non-empty string")
        key = _generated_claim_key(claim, target_pot_id)
    prefix = f"claim:{source_pot_id}:"
    if source_pot_id != target_pot_id and key.startswith(prefix):
        key = f"claim:{target_pot_id}:{key[len(prefix):]}"
    claim["claim_key"] = key
    claim.setdefault("properties", {})
    claim.setdefault("source_refs", [])
    claim.setdefault("evidence", [])
    claim.setdefault("fact_embedding", None)
    claim.setdefault("confidence", None)
    for field in _OPTIONAL_CLAIM_STRINGS:
        claim.setdefault(field, None)
    truth = claim.get("truth")
    expected_strength = evidence_strength_for_truth(truth if isinstance(truth, str) else None)
    strength = claim.get("evidence_strength")
    if strength is None:
        claim["evidence_strength"] = expected_strength
    elif not isinstance(strength, str) or not strength:
        raise ValueError(f"claims[{index}].evidence_strength must be a non-empty string")
    return claim


def normalize_snapshot_payload(
    payload: Mapping[str, Any], *, target_pot_id: str
) -> dict[str, Any]:
    """Validate all input and return a deterministic version-2 payload."""
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("snapshot payload must be a non-empty object")
    version = payload.get("format_version")
    if not isinstance(version, str) or version not in {"1", SNAPSHOT_FORMAT_VERSION}:
        raise ValueError(f"unsupported snapshot format_version: {version!r}")
    source_pot_id = payload.get("pot_id")
    if not isinstance(source_pot_id, str) or not source_pot_id:
        raise ValueError("snapshot pot_id must be a non-empty string")
    if not isinstance(target_pot_id, str) or not target_pot_id:
        raise ValueError("target pot_id must be a non-empty string")
    claims_raw = payload.get("claims")
    if not isinstance(claims_raw, list):
        raise ValueError("snapshot claims must be a list")

    if version == "1":
        unknown = set(payload) - {"format_version", "pot_id", "claims", "labels"}
        if unknown:
            raise ValueError(f"version 1 snapshot has unknown fields: {sorted(unknown)!r}")
        labels = payload.get("labels")
        if not isinstance(labels, Mapping):
            raise ValueError("version 1 snapshot labels must be an object")
        entities_raw = [
            {"key": key, "labels": value, "properties": {}}
            for key, value in labels.items()
        ]
        labelled = {raw["key"] for raw in entities_raw}
        endpoints = {
            raw.get(field)
            for raw in claims_raw
            if isinstance(raw, Mapping)
            for field in ("subject_key", "object_key")
            if isinstance(raw.get(field), str) and raw.get(field)
        }
        entities_raw.extend(
            {"key": key, "labels": ["Entity"], "properties": {}}
            for key in endpoints - labelled
        )
    else:
        unknown = set(payload) - {"format_version", "pot_id", "claims", "entities"}
        if unknown:
            raise ValueError(f"version 2 snapshot has unknown fields: {sorted(unknown)!r}")
        entities_raw = payload.get("entities")
        if not isinstance(entities_raw, list):
            raise ValueError("version 2 snapshot entities must be a list")

    entities = [_normalize_entity(raw, index=i) for i, raw in enumerate(entities_raw)]
    claims = [
        _normalize_claim(
            raw, index=i, source_pot_id=source_pot_id,
            target_pot_id=target_pot_id, legacy=version == "1",
        )
        for i, raw in enumerate(claims_raw)
    ]
    _reject_conflicting_duplicates(entities, "key", "entity")
    _reject_conflicting_duplicates(claims, "claim_key", "claim")
    entity_keys = {row["key"] for row in entities}
    missing = sorted(
        {claim[field] for claim in claims for field in ("subject_key", "object_key")}
        - entity_keys
    )
    if missing:
        raise ValueError(f"snapshot claims reference missing entities: {missing!r}")
    return {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "pot_id": target_pot_id,
        "entities": sorted(entities, key=lambda item: item["key"]),
        "claims": sorted(claims, key=lambda item: item["claim_key"]),
    }


def _reject_conflicting_duplicates(rows: list[dict[str, Any]], key: str, kind: str) -> None:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row[key]
        prior = seen.get(identity)
        if prior is not None:
            detail = "conflicting " if prior != row else ""
            raise ValueError(f"duplicate {kind} identity ({detail}{identity})")
        seen[identity] = row


def build_snapshot_payload(
    *, pot_id: str, entities: Iterable[Mapping[str, Any]], claims: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    claim_rows = [dict(row) for row in claims]
    for row in claim_rows:
        if not row.get("claim_key"):
            row["claim_key"] = _generated_claim_key(row, pot_id)
    return normalize_snapshot_payload(
        {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "pot_id": pot_id,
            "entities": list(entities),
            "claims": claim_rows,
        },
        target_pot_id=pot_id,
    )


def validate_snapshot_merge(
    *, existing_entities: Iterable[Mapping[str, Any]],
    existing_claims: Iterable[Mapping[str, Any]], incoming: Mapping[str, Any]
) -> None:
    """Reject incoming identities whose target representation differs."""
    entity_index = {row["key"]: dict(row) for row in existing_entities}
    claim_index = {snapshot_claim_identity(row): dict(row) for row in existing_claims}
    for row in incoming["entities"]:
        prior = entity_index.get(row["key"])
        if prior is not None and prior != row:
            raise ValueError(f"snapshot entity conflicts with target: {row['key']}")
    for row in incoming["claims"]:
        identity = snapshot_claim_identity(row)
        prior = claim_index.get(identity)
        if prior is not None and prior != row:
            raise ValueError(f"snapshot claim conflicts with target: {identity}")


__all__ = [
    "RESERVED_SNAPSHOT_PROPERTIES",
    "SNAPSHOT_FORMAT_VERSION",
    "build_snapshot_payload",
    "normalize_snapshot_payload",
    "snapshot_claim_identity", "validate_snapshot_merge",
]
