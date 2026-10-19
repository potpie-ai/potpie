"""Read-only verification of a committed prepared mutation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Callable

from potpie_context_engine.core.definition import (
    GraphDefinition,
)
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.graph_plans import (
    GraphIngestionVerificationResult,
    GraphMutationPlanRecord,
)
from potpie_context_engine.core.ports.claim_query import (
    ClaimQueryFilter,
)
from potpie_context_engine.core.ports.graph.backend import GraphBackend

from .mutation_state import (
    _claim_keys_from_record,
    _dedupe_claim_rows,
    _int_map,
    _row_claim_key,
    _safe_int,
    _subgraph_versions,
    _unsupported_from_exception,
)

_CLAIM_CONTRACT_PROPERTY_KEYS = frozenset(
    {
        "claim_key",
        "subgraph",
        "truth",
        "evidence_strength",
        "confidence",
        "fact",
        "description",
        "source_refs",
        "evidence",
        "source_system",
        "source_ref",
        "valid_at",
        "valid_from",
        "observed_at",
        "created_by",
        "graph_contract_version",
        "ontology_version",
        "idempotency_key",
        "identity_key",
        "environment",
        "valid_until",
        "mutation_id",
    }
)


def _verify_ingestion_commit(
    backend: GraphBackend,
    *,
    pot_id: str,
    record: GraphMutationPlanRecord,
    before_quality: Mapping[str, Any] | None,
    definition: GraphDefinition,
    quality_snapshot: Callable[..., Mapping[str, Any]],
) -> GraphIngestionVerificationResult:
    claim_keys = _claim_keys_from_record(record)
    readback = _verification_readback(backend, pot_id=pot_id, claim_keys=claim_keys)
    content_readback = _claim_content_readback(backend, pot_id=pot_id, record=record)
    if "protocols" in definition.extensions:
        from potpie_context_engine.core.protocol_validation import (
            protocol_content_readback,
        )

        protocol_readback = protocol_content_readback(
            backend, pot_id=pot_id, record=record
        )
        generic_mismatches = tuple(content_readback.get("mismatches", ()))
        content_readback = {**content_readback, **protocol_readback}
        if protocol_readback.get("mismatches"):
            content_readback["mismatches"] = generic_mismatches + tuple(
                protocol_readback["mismatches"]
            )
        elif generic_mismatches:
            content_readback["mismatches"] = generic_mismatches
    after_quality = quality_snapshot(
        backend,
        pot_id=pot_id,
        definition=definition,
    )
    before_counts = _quality_count_map(before_quality)
    after_counts = _quality_count_map(after_quality)
    deltas = _quality_count_delta(before_counts, after_counts) if before_quality else {}
    regressions = _quality_regressions(
        before_quality=before_quality,
        after_quality=after_quality,
        deltas=deltas,
    )
    unsupported = tuple(readback["unsupported"]) + tuple(after_quality["unsupported"])
    warnings: list[str] = []
    if before_quality is None:
        warnings.append(
            "pre-commit quality baseline unavailable; quality regression comparison skipped"
        )
    status = "ok"
    ok = True
    detail = None
    recommended = None

    if content_readback.get("mismatches"):
        ok = False
        status = "degraded"
        detail = "persisted graph content differs from the committed mutation"
        recommended = "Inspect content_readback and preserve conflicting extractions in graph inbox before repair."
    elif readback["missing_claim_keys"]:
        ok = False
        status = "degraded"
        detail = "committed plan did not read back all expected claim keys"
        recommended = (
            "Inspect graph history for the plan and rerun graph commit verification "
            "after backend repair if the mutation was applied."
        )
    elif readback["unsupported"]:
        ok = False
        status = "partial"
        detail = "claim readback is unavailable for the active backend"
        recommended = "Use graph status to inspect backend claim-query support."
    elif regressions:
        ok = False
        status = "degraded"
        detail = "quality findings increased after commit"
        recommended = (
            "Run the affected graph quality reports, then correct facts with "
            "graph propose/commit or add a graph inbox item for uncertain work."
        )
    elif unsupported:
        status = "partial"
        warnings.append("some verification checks are unavailable for this backend")
        recommended = "Use graph status to inspect backend capabilities."
    elif after_quality["status"] not in {"ok", "empty"}:
        status = "watch"
        warnings.append(f"graph quality status is {after_quality['status']}")
        recommended = (
            "Review graph quality summary before relying on newly committed memory."
        )
    elif not claim_keys and not content_readback.get("checked_retracted_claim_keys"):
        # Entity-only (or retraction-only) plans have nothing to read back, and
        # reporting the plain "ok" for them made verification read as proof the
        # write landed when no claim was ever checked. Say what was actually
        # verified; the commit itself still succeeded, so ``ok`` stays true.
        status = "no_claims"
        detail = "plan committed no claim keys; claim readback verified nothing"
        warnings.append("verification read back no claims: the plan asserted none")
        recommended = (
            "Read the affected entities back with graph search-entities, or add the "
            "claims that make this write findable."
        )

    return GraphIngestionVerificationResult(
        ok=ok,
        status=status,
        plan_id=record.plan_id,
        pot_id=pot_id,
        claim_keys=claim_keys,
        readback_claim_keys=tuple(readback["readback_claim_keys"]),
        missing_claim_keys=tuple(readback["missing_claim_keys"]),
        readback_count=int(readback["readback_count"]),
        content_readback=content_readback,
        quality_status=str(after_quality["status"]),
        quality_counts=after_counts,
        quality_delta=deltas,
        quality_regressions=regressions,
        checked_reports=tuple(after_quality["checked_reports"]),
        warnings=tuple(warnings),
        unsupported=unsupported,
        detail=detail,
        recommended_next_action=recommended,
        subgraph_versions=_subgraph_versions(backend, pot_id),
    )


def _claim_content_readback(
    backend: GraphBackend,
    *,
    pot_id: str,
    record: GraphMutationPlanRecord,
) -> dict[str, Any]:
    """Compare persisted ordinary claims with the exact lowered write payload."""
    batch = record.lowered_batch
    if batch is None:
        return {}
    expected: dict[str, dict[str, Any]] = {}
    for edge in batch.edge_upserts:
        props = edge.properties
        claim_key = str(props.get("claim_key") or "")
        if not claim_key:
            continue
        expected[claim_key] = {
            "predicate": edge.edge_type,
            "subject_key": edge.from_entity_key,
            "object_key": edge.to_entity_key,
            "subgraph": props.get("subgraph"),
            "truth": props.get("truth"),
            "confidence": props.get("confidence"),
            "fact": props.get("fact"),
            "description": props.get("description"),
            "environment": props.get("environment"),
            "source_refs": tuple(props.get("source_refs") or ()),
            "evidence": tuple(props.get("evidence") or ()),
            "valid_until": props.get("valid_until"),
            "properties": {
                key: value
                for key, value in props.items()
                if key not in _CLAIM_CONTRACT_PROPERTY_KEYS
            },
        }
    retracted = tuple(
        dict.fromkeys(
            key
            for invalidation in batch.invalidations
            for key in (getattr(invalidation, "target_claim_keys", None) or ())
        )
    )
    if not expected and not retracted and not batch.entity_upserts:
        return {}
    requested_keys = tuple(expected) + tuple(
        key for key in retracted if key not in expected
    )
    rows = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=pot_id,
            claim_key_in=requested_keys,
            include_invalidated=True,
            limit=max(20, len(requested_keys) * 2),
        )
    )
    by_key = {str(row.claim_key): row for row in rows if row.claim_key}
    mismatches: list[dict[str, Any]] = []
    checked_entities: list[str] = []
    for entity in batch.entity_upserts:
        checked_entities.append(entity.entity_key)
        actual_properties = backend.claim_query.entity_properties(
            pot_id=pot_id, entity_key=entity.entity_key
        )
        actual_labels = backend.claim_query.entity_labels(
            pot_id=pot_id, entity_keys=(entity.entity_key,)
        ).get(entity.entity_key, ())
        entity_differences: dict[str, Any] = {}
        for key, value in entity.properties.items():
            actual_value = _normalize_persisted_json(value, actual_properties.get(key))
            if _comparable_value(value) != _comparable_value(actual_value):
                entity_differences[f"properties.{key}"] = {
                    "expected": value,
                    "actual": actual_properties.get(key),
                }
        if not set(entity.labels).issubset(set(actual_labels)):
            entity_differences["labels"] = {
                "expected": tuple(entity.labels),
                "actual": tuple(actual_labels),
            }
        if entity_differences:
            mismatches.append(
                {"entity_key": entity.entity_key, "fields": entity_differences}
            )
    for claim_key, intended in expected.items():
        row = by_key.get(claim_key)
        if row is None:
            continue
        actual = {
            "predicate": row.predicate,
            "subject_key": row.subject_key,
            "object_key": row.object_key,
            "subgraph": row.subgraph,
            "truth": row.truth,
            "confidence": row.confidence,
            "fact": row.fact,
            "description": row.description,
            "environment": row.environment,
            "source_refs": tuple(row.source_refs),
            "evidence": tuple(row.evidence),
            "valid_until": row.valid_until.isoformat() if row.valid_until else None,
            "properties": dict(row.properties),
        }
        differences = {}
        for field, value in intended.items():
            actual_value = actual[field]
            if field == "properties" and isinstance(value, Mapping):
                actual_mapping = (
                    actual_value if isinstance(actual_value, Mapping) else {}
                )
                property_diff = {
                    key: {"expected": expected_value, "actual": actual_mapping.get(key)}
                    for key, expected_value in value.items()
                    if _comparable_value(expected_value)
                    != _comparable_value(
                        _normalize_persisted_json(
                            expected_value, actual_mapping.get(key)
                        )
                    )
                }
                if property_diff:
                    differences[field] = property_diff
            elif _comparable_value(value) != _comparable_value(actual_value):
                differences[field] = {"expected": value, "actual": actual_value}
        if differences:
            mismatches.append({"claim_key": claim_key, "fields": differences})
    valid_to_by_key = {
        key: invalidation.valid_to
        for invalidation in batch.invalidations
        for key in (getattr(invalidation, "target_claim_keys", None) or ())
    }
    for claim_key in retracted:
        row = by_key.get(claim_key)
        expected_invalid_at = valid_to_by_key.get(claim_key) or "set"
        actual_invalid_at = (
            row.invalid_at.isoformat() if row and row.invalid_at else None
        )
        if (
            row is None
            or row.invalid_at is None
            or (
                expected_invalid_at != "set"
                and _comparable_value(expected_invalid_at)
                != _comparable_value(actual_invalid_at)
            )
        ):
            mismatches.append(
                {
                    "claim_key": claim_key,
                    "fields": {
                        "invalid_at": {
                            "expected": expected_invalid_at,
                            "actual": actual_invalid_at,
                        }
                    },
                }
            )
    return {
        "checked_claim_keys": tuple(expected),
        "checked_entities": tuple(checked_entities),
        "checked_retracted_claim_keys": retracted,
        "mismatches": tuple(mismatches),
    }


def _comparable_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value.endswith("Z"):
        return value[:-1] + "+00:00"
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


def _verification_readback(
    backend: GraphBackend,
    *,
    pot_id: str,
    claim_keys: tuple[str, ...],
) -> dict[str, Any]:
    if not claim_keys:
        return {
            "readback_claim_keys": (),
            "missing_claim_keys": (),
            "readback_count": 0,
            "unsupported": (),
        }
    try:
        rows = backend.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=pot_id,
                claim_key_in=claim_keys,
                include_invalidated=True,
                limit=max(len(claim_keys) * 2, 20),
            )
        )
    except CapabilityNotImplemented as exc:
        return {
            "readback_claim_keys": (),
            "missing_claim_keys": (),
            "readback_count": 0,
            "unsupported": (
                _unsupported_from_exception(exc, fallback="claim_query.find_claims"),
            ),
        }
    readback_keys = tuple(
        dict.fromkeys(_row_claim_key(row) for row in _dedupe_claim_rows(list(rows)))
    )
    readback_set = set(readback_keys)
    return {
        "readback_claim_keys": readback_keys,
        "missing_claim_keys": tuple(
            key for key in claim_keys if key not in readback_set
        ),
        "readback_count": len(readback_keys),
        "unsupported": (),
    }


def _quality_count_map(snapshot: Mapping[str, Any] | None) -> dict[str, int]:
    if not snapshot:
        return {}
    counts = _int_map(snapshot.get("quality_counts"))
    if "total_findings" not in counts:
        counts["total_findings"] = _safe_int(snapshot.get("total_findings"))
    return counts


def _quality_count_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def _quality_regressions(
    *,
    before_quality: Mapping[str, Any] | None,
    after_quality: Mapping[str, Any],
    deltas: Mapping[str, int],
) -> dict[str, Mapping[str, int]]:
    if not before_quality:
        return {}
    regressions = {
        key: {
            "before": _safe_int(_quality_count_map(before_quality).get(key)),
            "after": _safe_int(_quality_count_map(after_quality).get(key)),
            "delta": int(delta),
        }
        for key, delta in deltas.items()
        if delta > 0
    }
    before_rank = _quality_status_rank(str(before_quality.get("status") or "unknown"))
    after_rank = _quality_status_rank(str(after_quality.get("status") or "unknown"))
    if after_rank > before_rank:
        regressions["health_status"] = {
            "before": before_rank,
            "after": after_rank,
            "delta": after_rank - before_rank,
        }
    return regressions


def _quality_status_rank(status: str) -> int:
    return {
        "ok": 0,
        "empty": 0,
        "watch": 1,
        "partial": 1,
        "unavailable": 1,
        "unknown": 1,
        "degraded": 2,
    }.get(status, 1)


def _verification_next_action(
    verification: GraphIngestionVerificationResult | None,
) -> str | None:
    if verification is None:
        return None
    if not verification.ok or verification.status != "ok":
        return verification.recommended_next_action
    return None
