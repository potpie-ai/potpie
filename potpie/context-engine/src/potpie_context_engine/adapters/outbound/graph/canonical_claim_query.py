"""Shared canonical claim-query helpers for Cypher graph adapters."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime
from typing import Any, Iterable, Mapping

from potpie_context_engine.domain.graph_contract import evidence_strength_for_truth
from potpie_context_engine.domain.ports.claim_query import ClaimRow

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
    return value.isoformat() if value is not None else None


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
        valid_at=parse_dt(props.get("valid_at")),
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
    )


def embedding_score(fact: str | None, query: str) -> float:
    """Token-overlap stand-in for native vector similarity."""
    if not fact:
        return 0.0
    a = set(query.lower().split())
    b = set(fact.lower().split())
    if not a or not b:
        return 0.0
    base = len(a & b) / len(a | b)
    if a & b:
        base = math.sqrt(base)
    return max(0.0, min(1.0, base))


def stamp_similarity(rows: Iterable[ClaimRow], query: str) -> list[ClaimRow]:
    """Sort rows by lexical similarity and stamp ``semantic_similarity``."""
    scored = sorted(
        ((embedding_score(row.fact, query), row) for row in rows),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return stamp_scored_rows(scored)


def stamp_scored_rows(scored: Iterable[tuple[float, ClaimRow]]) -> list[ClaimRow]:
    """Stamp precomputed scores onto rows, preserving score order."""
    out: list[ClaimRow] = []
    for score, row in scored:
        props = dict(row.properties)
        props["semantic_similarity"] = float(score)
        out.append(dataclasses.replace(row, properties=props))
    return out


FIND_CLAIMS_CYPHER = """
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO {group_id: $gid}]->(b:Entity {group_id: $gid})
WHERE ($preds IS NULL OR r.name IN $preds)
  AND ($subjects IS NULL OR r.subject_key IN $subjects)
  AND ($objects IS NULL OR r.object_key IN $objects)
  AND ($claim_keys IS NULL OR r.claim_key IN $claim_keys)
  AND ($subgraphs IS NULL OR r.subgraph IN $subgraphs)
  AND ($mutation_ids IS NULL OR r.mutation_id IN $mutation_ids)
  AND ($source_refs IS NULL OR r.source_ref IN $source_refs OR any(ref IN coalesce(r.source_refs, []) WHERE ref IN $source_refs))
  AND ($sources IS NULL OR r.source_system IN $sources)
  AND ($include_invalid OR r.invalid_at IS NULL)
  AND ($as_of IS NULL OR r.valid_at IS NULL OR r.valid_at <= $as_of)
  AND ($va_after IS NULL OR (r.valid_at IS NOT NULL AND r.valid_at >= $va_after))
  AND ($va_before IS NULL OR r.valid_at IS NULL OR r.valid_at <= $va_before)
  AND ($subject_label IS NULL OR $subject_label IN labels(a))
  AND ($object_label IS NULL OR $object_label IN labels(b))
RETURN r{.*} AS props
"""


ENTITY_LABELS_CYPHER = """
MATCH (e:Entity {group_id: $gid})
WHERE e.entity_key IN $keys
RETURN e.entity_key AS key, labels(e) AS labels
"""


__all__ = [
    "ENTITY_LABELS_CYPHER",
    "FIND_CLAIMS_CYPHER",
    "CONTRACT_EDGE_KEYS",
    "RESERVED_EDGE_KEYS",
    "embedding_score",
    "iso",
    "parse_dt",
    "row_from_record",
    "stamp_similarity",
    "stamp_scored_rows",
    "vector_property",
]
