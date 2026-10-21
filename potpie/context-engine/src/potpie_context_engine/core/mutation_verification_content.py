"""Compare frozen mutation effects with exact current claim incarnations.

This module reads and compares only. The coordinator owns receipt authority,
revision coherence, quality checks, and failure reporting.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .claim_codec import (
    CONTRACT_EDGE_KEYS,
    ENTITY_TEMPORAL_FIELDS,
    normalize_timestamp,
    parse_dt,
)
from .claim_lifecycle import REVIEW_FIELDS
from .claim_preparation import assertion_row
from .ports.claim_query import ClaimQueryFilter


@dataclass
class _ExpectedClaim:
    record_id: str
    fields: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    endpoint_key: str | None = None


def _expected_claims(record):
    expected: dict[str, _ExpectedClaim] = {}
    unresolved = []
    batch = record.lowered_batch
    for op in (*batch.edge_deletes, *batch.invalidations):
        if op.closures is None:
            unresolved.append("legacy correction has no frozen incarnation targets")
        for closure in op.closures or ():
            expected[closure.target.record_id] = _ExpectedClaim(
                closure.target.record_id,
                {
                    "pot_id": record.pot_id,
                    "claim_key": closure.target.claim_key,
                    "valid_at": closure.after.valid_at,
                    "invalid_at": closure.after.invalid_at,
                    "valid_until": closure.after.valid_until,
                },
            )
            intended = expected[closure.target.record_id]
            triple = getattr(op, "target_edge", None)
            if hasattr(op, "edge_type"):
                triple = (op.edge_type, op.from_entity_key, op.to_entity_key)
            if triple is not None:
                intended.fields.update(
                    zip(("predicate", "subject_key", "object_key"), triple)
                )
            intended.endpoint_key = getattr(op, "target_entity_key", None)
    for edge in batch.edge_upserts:
        resolution = edge.resolution
        if resolution is None:
            unresolved.append("legacy assertion has no frozen incarnation identity")
            continue
        row = assertion_row(
            edge,
            pot_id=record.pot_id,
            record_id=resolution.record_id,
            anchor=(
                record.prepared.prepared_at if record.prepared else record.created_at
            ),
        )
        scope = {
            name: getattr(row, name)
            for name in (
                "pot_id",
                "subject_key",
                "predicate",
                "subgraph",
                "environment",
                "source_system",
                "source_ref",
            )
        }
        intended = expected.setdefault(
            resolution.record_id, _ExpectedClaim(resolution.record_id)
        )
        intended.fields.update(scope)
        intended.fields.update(
            object_key=row.object_key,
            valid_at=resolution.valid_at,
            invalid_at=resolution.invalid_at,
            valid_until=resolution.valid_until,
        )
        if row.claim_key is not None or not resolution.existing:
            intended.fields["claim_key"] = row.claim_key
        if resolution.metadata_review:
            intended.properties.update(
                {k: v for k, v in edge.properties.items() if k in REVIEW_FIELDS}
            )
        else:
            # Existing assertions preserve omitted optional content. Verify only
            # fields present in the frozen write, plus its canonical identity.
            for name in (
                "truth",
                "confidence",
                "fact",
                "description",
                "observed_at",
                "source_refs",
                "evidence",
                "evidence_strength",
                "graph_contract_version",
                "ontology_version",
            ):
                if name in edge.properties:
                    intended.fields[name] = getattr(row, name)
            intended.properties.update(
                {
                    k: v
                    for k, v in edge.properties.items()
                    if k not in CONTRACT_EDGE_KEYS
                }
            )
        for predecessor_id in resolution.superseded_record_ids:
            previous = expected.setdefault(
                predecessor_id, _ExpectedClaim(predecessor_id)
            )
            previous.fields.update(scope, invalid_at=resolution.effective_at)
    return expected, tuple(dict.fromkeys(unresolved))


def claim_content_readback(backend, *, pot_id, record, query_time: datetime):
    batch = record.lowered_batch
    if batch is None:
        return {
            "outcome": "inconclusive",
            "unresolved": ("legacy plan has no frozen batch",),
        }
    expected, unresolved = _expected_claims(record)
    # Empty record_id_in means unrestricted to the port. Never query on an
    # empty frozen selection, and never let a later logical incarnation match.
    rows = (
        backend.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=pot_id,
                record_id_in=tuple(expected),
                include_invalidated=True,
                include_retired=True,
            )
        )
        if expected
        else []
    )
    by_id = {}
    for row in rows:
        by_id.setdefault(row.record_id, []).append(row)
    mismatches = []
    outcomes = []
    readback_keys = []
    missing_keys = []
    for identity, intended in expected.items():
        candidates = by_id.get(identity, ())
        row = candidates[0] if len(candidates) == 1 else None
        differences = {}
        outcome = "matching"
        lifecycle = None
        if row is None:
            differences["incarnation"] = {"expected": "one", "actual": len(candidates)}
            if intended.fields.get("claim_key"):
                missing_keys.append(intended.fields["claim_key"])
        else:
            if row.claim_key:
                readback_keys.append(row.claim_key)
            differences = _claim_differences(intended, row)
            # A later shortening/retirement changes current lifecycle without
            # disputing the original receipt. Widening/removing an end remains a
            # mismatch; there is no evidence here of why any change occurred.
            changed_closure = row.retired
            for name in ("invalid_at", "valid_until"):
                difference = differences.get(name)
                if (
                    difference
                    and _shortened(intended.fields[name], getattr(row, name))
                    and (
                        _instant(row.valid_at) is None
                        or _instant(getattr(row, name)) > _instant(row.valid_at)
                    )
                ):
                    changed_closure = True
                    del differences[name]
            if changed_closure:
                outcome = "superseded_retracted"
            lifecycle = _lifecycle(row, query_time)
            if row.temporal_diagnostics:
                outcome = "inconclusive"
                lifecycle = {
                    "state": "unknown",
                    "temporal_diagnostics": dict(row.temporal_diagnostics),
                }
        if differences:
            outcome = "mismatched"
            mismatches.append(
                {
                    "claim_key": intended.fields.get("claim_key"),
                    "record_id": identity,
                    "fields": differences,
                }
            )
        outcomes.append(
            {
                "record_id": identity,
                "claim_key": intended.fields.get("claim_key"),
                "outcome": outcome,
                "lifecycle": lifecycle,
            }
        )
    entity_mismatches = _entity_readback(backend, pot_id, batch.entity_upserts)
    mismatches.extend(entity_mismatches)
    outcome = (
        "mismatched"
        if mismatches
        else (
            "inconclusive"
            if unresolved or any(x["outcome"] == "inconclusive" for x in outcomes)
            else (
                "superseded_retracted"
                if any(x["outcome"] == "superseded_retracted" for x in outcomes)
                else "matching"
            )
        )
    )
    closures = tuple(
        c
        for op in (*batch.edge_deletes, *batch.invalidations)
        for c in op.closures or ()
    )
    return {
        "outcome": outcome,
        "unresolved": unresolved,
        "claim_outcomes": tuple(outcomes),
        "mismatches": tuple(mismatches),
        "checked_claim_keys": tuple(
            dict.fromkeys(
                e.fields["claim_key"]
                for e in expected.values()
                if e.fields.get("claim_key")
            )
        ),
        "checked_claim_record_ids": tuple(expected),
        "checked_correction_targets": tuple(c.target.to_dict() for c in closures),
        "checked_retracted_claim_keys": tuple(
            dict.fromkeys(c.target.claim_key for c in closures if c.target.claim_key)
        ),
        "checked_entities": tuple(e.entity_key for e in batch.entity_upserts),
        "readback_claim_keys": tuple(dict.fromkeys(readback_keys)),
        "missing_claim_keys": tuple(dict.fromkeys(missing_keys)),
        "readback_count": sum(
            len(by_id.get(identity, ())) == 1 for identity in expected
        ),
    }


def _claim_differences(intended, row):
    differences = {}
    for name, value in intended.fields.items():
        actual = getattr(row, name)
        temporal = name in {"valid_at", "invalid_at", "valid_until", "observed_at"}
        if temporal and (
            name in row.temporal_diagnostics
            or name == "valid_at"
            and "valid_from" in row.temporal_diagnostics
        ):
            continue  # Hydrated value cannot prove equality or a specific mismatch.
        if _comparison(value, temporal=temporal) != _comparison(
            actual, temporal=temporal
        ):
            differences[name] = {
                "expected": _comparable_value(value),
                "actual": _comparable_value(actual),
            }
    property_diff = _property_differences(
        {
            k: v
            for k, v in intended.properties.items()
            if k not in row.temporal_diagnostics
        },
        row.properties,
    )
    if property_diff:
        differences["properties"] = property_diff
    if intended.endpoint_key is not None and intended.endpoint_key not in (
        row.subject_key,
        row.object_key,
    ):
        differences["endpoint"] = {
            "expected": intended.endpoint_key,
            "actual": (row.subject_key, row.object_key),
        }
    return differences


def _entity_readback(backend, pot_id, entities):
    mismatches = []
    if not entities:
        return mismatches
    labels = backend.claim_query.entity_labels(
        pot_id=pot_id, entity_keys=tuple(e.entity_key for e in entities)
    )
    # Multiple upserts may patch the same entity. Compare their final combined
    # authored properties; generated entity identity is not invented here.
    expected = {}
    for entity in entities:
        props, wanted_labels = expected.setdefault(entity.entity_key, ({}, set()))
        props.update(entity.properties)
        wanted_labels.update(entity.labels)
    for key, (props, wanted_labels) in expected.items():
        actual = backend.claim_query.entity_properties(pot_id=pot_id, entity_key=key)
        differences = {
            "properties." + k: v
            for k, v in _property_differences(props, actual).items()
        }
        if key not in labels:
            differences["entity"] = {"expected": "present", "actual": "missing"}
        if not wanted_labels.issubset(set(labels.get(key, ()))):
            differences["labels"] = {
                "expected": sorted(wanted_labels),
                "actual": labels.get(key, ()),
            }
        if differences:
            mismatches.append({"entity_key": key, "fields": differences})
    return mismatches


def _property_differences(expected, actual):
    differences = {}
    for key, value in expected.items():
        found = _normalize_persisted_json(value, actual.get(key))
        if _comparison(value, temporal=key in ENTITY_TEMPORAL_FIELDS) != _comparison(
            found, temporal=key in ENTITY_TEMPORAL_FIELDS
        ):
            differences[key] = {"expected": value, "actual": actual.get(key)}
    return differences


def _comparison(value, *, temporal=False):
    if temporal and value is not None and value != "":
        try:
            return normalize_timestamp(value, field="verification")
        except ValueError:
            return ("invalid_timestamp", str(value))
    return _comparable_value(value)


def _instant(value):
    if value is None:
        return None
    try:
        return parse_dt(normalize_timestamp(value, field="verification"))
    except ValueError:
        return None


def _shortened(expected, actual):
    actual_time = _instant(actual)
    return actual_time is not None and (
        expected is None
        or _instant(expected) is not None
        and actual_time < _instant(expected)
    )


def _lifecycle(row, now):
    ends = [_instant(v) for v in (row.invalid_at, row.valid_until) if v is not None]
    start = _instant(row.valid_at)
    if any(v is None for v in ends) or row.valid_at is not None and start is None:
        state = "unknown"
    elif row.retired:
        state = "retired"
    elif ends and min(ends) <= now:
        state = "closed"
    elif start is not None and start > now:
        state = "future"
    elif ends:
        state = "scheduled_close"
    else:
        state = "active"
    return {
        "state": state,
        "valid_at": _comparable_value(row.valid_at),
        "invalid_at": _comparable_value(row.invalid_at),
        "valid_until": _comparable_value(row.valid_until),
        "retired": row.retired,
    }


def _comparable_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {key: _comparable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_comparable_value(item) for item in value)
    return value


def _normalize_persisted_json(expected: Any, actual: Any) -> Any:
    """Decode adapter-serialized JSON only when the intended value is structured."""
    if not isinstance(expected, (Mapping, list, tuple)) or not isinstance(actual, str):
        return actual
    try:
        decoded = json.loads(actual)
    except (TypeError, ValueError):
        return actual
    if isinstance(expected, Mapping) and isinstance(decoded, Mapping):
        return decoded
    if isinstance(expected, (list, tuple)) and isinstance(decoded, list):
        return decoded
    return actual
