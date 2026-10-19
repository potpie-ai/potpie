"""Pure canonical persisted-claim hydration shared by preparation and adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .graph_contract import evidence_strength_for_truth
from .ports.claim_query import ClaimRow

# Edge properties that are part of the canonical V1.5 contract or backend system
# frame. These are hydrated into first-class ``ClaimRow`` fields or intentionally
# hidden from the reader extras bag. ``ClaimRow.properties`` is for non-contract
# annotations only.
CONTRACT_EDGE_KEYS = frozenset(
    {
        "created_at",
        "created_by",
        "claim_key",
        "confidence",
        "description",
        "embedding_dim",
        "embedding_model",
        "environment",
        "evidence",
        "evidence_strength",
        "expired_at",
        "fact",
        "fact_embedding",
        "group_id",
        "graph_contract_version",
        "identity_key",
        "idempotency_key",
        "invalid_at",
        "mutation_id",
        "name",
        "object_key",
        "observed_at",
        "ontology_version",
        "record_id",
        "retired",
        "__potpie_snapshot_properties_v2",
        "__potpie_snapshot_claim_fields_v2",
        "source_ref",
        "source_refs",
        "source_system",
        "subgraph",
        "subject_key",
        "truth",
        "uuid",
        "valid_at",
        "valid_from",
        "valid_until",
    }
)
RESERVED_EDGE_KEYS = CONTRACT_EDGE_KEYS


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    # neo4j.time.DateTime -> native
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        try:
            native = to_native()
        except Exception:  # noqa: BLE001
            return None
        return native if isinstance(native, datetime) else None
    return None


def iso(value: datetime | None) -> str | None:
    return _as_utc(value).isoformat() if value is not None else None


def normalize_timestamp(value: Any, *, field: str) -> str:
    """Encode an authored new timestamp; legacy diagnostic parsing stays separate."""
    parsed = parse_dt(value)
    if parsed is None:
        raise ValueError(
            f"{field} must be a valid ISO 8601 timestamp with an explicit UTC offset"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} timestamp requires an explicit UTC offset")
    return _as_utc(parsed).isoformat()


def normalize_claim_times(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the temporal contract without changing unrelated annotations."""
    props = dict(properties)
    for field in (
        "valid_at",
        "valid_from",
        "invalid_at",
        "valid_until",
        "observed_at",
        "occurred_at",
    ):
        if props.get(field) is not None:
            props[field] = normalize_timestamp(props[field], field=field)
    starts = [
        props[field]
        for field in ("valid_at", "valid_from")
        if props.get(field) is not None
    ]
    if len(set(starts)) > 1:
        raise ValueError("valid_at and valid_from timestamps disagree")
    if starts:
        props["valid_at"] = props["valid_from"] = starts[0]
        start = parse_dt(starts[0])
        for field in ("invalid_at", "valid_until"):
            if props.get(field) is not None and parse_dt(props[field]) <= start:
                raise ValueError(f"{field} timestamp must follow the validity start")
    return props


ENTITY_TEMPORAL_FIELDS = frozenset(
    {
        "valid_at",
        "valid_from",
        "valid_to",
        "invalid_at",
        "valid_until",
        "observed_at",
        "occurred_at",
        "deployed_at",
        "merge_recorded_at",
        "state_transition_at",
    }
)


def normalize_entity_times(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Standard entity time fields share the claim timestamp syntax and aliases."""
    props = dict(properties)
    for field in ENTITY_TEMPORAL_FIELDS:
        value = props.get(field)
        if value is not None and value != "":
            props[field] = normalize_timestamp(value, field=field)
    # Empty required-property defaults mean unknown, not an authored instant.
    temporal = {
        key: value
        for key, value in props.items()
        if key in ENTITY_TEMPORAL_FIELDS and value != ""
    }
    props.update(normalize_claim_times(temporal))
    return props


def normalize_snapshot_claim_times(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize one imported/comparable claim without regenerating identity."""
    canonical = dict(claim)
    props = dict(claim.get("properties") or {})
    combined = dict(props)
    fields = ("valid_at", "invalid_at", "valid_until", "observed_at")
    for field in fields:
        value = claim.get(field)
        if value is not None:
            encoded = normalize_timestamp(value, field=field)
            if (
                props.get(field) is not None
                and normalize_timestamp(props[field], field=field) != encoded
            ):
                raise ValueError(f"snapshot {field} timestamp aliases disagree")
            combined[field] = encoded
    normalized = normalize_claim_times(combined)
    for field in fields:
        if normalized.get(field) is not None:
            canonical[field] = normalized[field]
        props.pop(field, None)
    # Validity aliases are canonical fields, not independent annotations.
    props.pop("valid_from", None)
    if normalized.get("occurred_at") is not None:
        props["occurred_at"] = normalized["occurred_at"]
    canonical["properties"] = props
    return canonical


def vector_property(value: Any) -> tuple[float, ...] | None:
    """Coerce a stored vector property into the ClaimRow tuple shape."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return tuple(float(x) for x in value)
        except (TypeError, ValueError):
            return None
    try:
        return tuple(float(x) for x in list(value))
    except Exception:  # noqa: BLE001
        return None


def _decode_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _coerce_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_tuple(value: Any) -> tuple[str, ...]:
    value = _decode_json_value(value)
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item is not None and str(item))
    return ()


def _evidence_tuple(value: Any) -> tuple[Mapping[str, Any], ...]:
    value = _decode_json_value(value)
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[Mapping[str, Any]] = []
    for item in value:
        item = _decode_json_value(item)
        if isinstance(item, Mapping):
            out.append(dict(item))
    return tuple(out)


def _reader_extras(props: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: _decode_json_value(v)
        for k, v in props.items()
        if k not in CONTRACT_EDGE_KEYS
    }


def row_from_record(rec: Mapping[str, Any]) -> ClaimRow:
    """Build a ``ClaimRow`` from one Cypher record containing ``props``."""
    props = dict(rec.get("props") or {})
    truth = _coerce_str(props.get("truth"))
    source_refs = _str_tuple(props.get("source_refs"))
    return ClaimRow(
        pot_id=str(props.get("group_id") or rec.get("gid") or ""),
        predicate=str(props.get("name") or ""),
        subject_key=str(props.get("subject_key") or ""),
        object_key=str(props.get("object_key") or ""),
        valid_at=parse_dt(props.get("valid_from") or props.get("valid_at")),
        invalid_at=parse_dt(props.get("invalid_at")),
        evidence_strength=evidence_strength_for_truth(truth),
        source_system=_coerce_str(props.get("source_system")),
        source_ref=_coerce_str(props.get("source_ref")),
        fact=_coerce_str(props.get("fact")),
        properties=_reader_extras(props),
        fact_embedding=vector_property(props.get("fact_embedding")),
        claim_key=_coerce_str(props.get("claim_key")),
        subgraph=_coerce_str(props.get("subgraph")),
        truth=truth,
        confidence=_coerce_float(props.get("confidence")),
        description=_coerce_str(props.get("description")),
        environment=_coerce_str(props.get("environment")),
        observed_at=parse_dt(props.get("observed_at")),
        valid_until=parse_dt(props.get("valid_until")),
        mutation_id=_coerce_str(props.get("mutation_id")),
        source_refs=source_refs,
        evidence=_evidence_tuple(props.get("evidence")),
        graph_contract_version=_coerce_str(props.get("graph_contract_version")),
        ontology_version=_coerce_str(props.get("ontology_version")),
        record_id=str(props.get("record_id") or props.get("uuid") or ""),
        retired=bool(props.get("retired", False)),
    )


def claim_properties(row: ClaimRow) -> dict[str, Any]:
    """Canonical property representation, excluding backend creation identity."""
    props = dict(row.properties)
    props.update(
        {
            "group_id": row.pot_id,
            "name": row.predicate,
            "subject_key": row.subject_key,
            "object_key": row.object_key,
            "record_id": row.record_id,
            "claim_key": row.claim_key,
            "subgraph": row.subgraph,
            "truth": row.truth,
            "confidence": row.confidence,
            "fact": row.fact,
            "description": row.description,
            "environment": row.environment,
            "source_system": row.source_system,
            "source_ref": row.source_ref,
            "source_refs": list(row.source_refs),
            "evidence": [dict(item) for item in row.evidence],
            "evidence_strength": row.evidence_strength,
            "mutation_id": row.mutation_id,
            "graph_contract_version": row.graph_contract_version,
            "ontology_version": row.ontology_version,
            "valid_at": iso(row.valid_at),
            "valid_from": iso(row.valid_at),
            "invalid_at": iso(row.invalid_at),
            "valid_until": iso(row.valid_until),
            "observed_at": iso(row.observed_at),
        }
    )
    if row.fact_embedding is not None:
        props["fact_embedding"] = list(row.fact_embedding)
    return props


def claim_preservation_snapshot(row: ClaimRow) -> dict[str, Any]:
    def timestamp(value):
        return value.isoformat() if value is not None else None

    return {
        "record_id": row.record_id,
        "subgraph": row.subgraph,
        "source_ref": row.source_ref,
        "claim_key": row.claim_key,
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "fact": row.fact,
        "description": row.description,
        "source_system": row.source_system,
        "source_refs": list(row.source_refs),
        "evidence": [dict(item) for item in row.evidence],
        "properties": dict(row.properties),
        "valid_at": timestamp(row.valid_at),
        "invalid_at": timestamp(row.invalid_at),
        "valid_until": timestamp(row.valid_until),
        "observed_at": timestamp(row.observed_at),
        "truth": row.truth,
        "confidence": row.confidence,
        "environment": row.environment,
    }
