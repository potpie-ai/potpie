"""One coordinator for coherent, read-only verification after execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Callable

from .definition import GraphDefinition
from .errors import CapabilityNotImplemented
from .graph_plans import GraphIngestionVerificationResult, GraphMutationPlanRecord
from .ports.graph.backend import GraphBackend
from .mutation_state import (
    _claim_keys_from_record,
    _int_map,
    _safe_int,
    _subgraph_versions,
    _unsupported_from_exception,
)
from .mutation_verification_content import claim_content_readback

_LOG = logging.getLogger(__name__)


def inconclusive_verification(*, plan_id, pot_id, detail, unsupported=(), status=None):
    return GraphIngestionVerificationResult(
        ok=False,
        status=status or ("partial" if unsupported else "inconclusive"),
        state="unsupported" if unsupported else "deferred",
        outcome="inconclusive",
        plan_id=plan_id,
        pot_id=pot_id,
        detail=detail,
        verified_at=datetime.now(timezone.utc),
        unsupported=tuple(unsupported),
        recommended_next_action="Retry verification for this plan_id; verification does not reapply the mutation.",
    )


def verify_completed_execution(verify_receipt, backend, *, pot_id, record, definition):
    """Contain only post-receipt checks; execution success remains authoritative."""
    try:
        return verify_receipt(
            backend,
            pot_id=pot_id,
            record=record,
            before_quality=record.verification_quality_before,
            definition=definition,
        )
    except CapabilityNotImplemented as exc:
        return inconclusive_verification(
            plan_id=record.plan_id,
            pot_id=pot_id,
            detail="current graph verification is unsupported by this backend",
            unsupported=(_unsupported_from_exception(exc, fallback="verification"),),
        )
    except (Exception, asyncio.CancelledError) as exc:
        _LOG.warning(
            "post-commit verification did not complete (plan_id=%s, error=%s)",
            record.plan_id,
            type(exc).__name__,
        )
        return inconclusive_verification(
            plan_id=record.plan_id,
            pot_id=pot_id,
            detail="The execution receipt is durable; current graph verification did not complete.",
        )


def capture_quality_baseline(quality_snapshot, backend, *, pot_id, definition):
    """A quality baseline is optional; pre-apply cancellation still propagates."""
    try:
        return quality_snapshot(backend, pot_id=pot_id, definition=definition)
    except Exception:
        _LOG.warning(
            "pre-commit quality baseline unavailable (pot_id=%s)", pot_id, exc_info=True
        )
        return None


def _verify_ingestion_commit(
    backend: GraphBackend,
    *,
    pot_id: str,
    record: GraphMutationPlanRecord,
    before_quality: Mapping[str, Any] | None,
    definition: GraphDefinition,
    quality_snapshot: Callable[..., Mapping[str, Any]],
) -> GraphIngestionVerificationResult:
    if record.lowered_batch is not None and record.lowered_batch.record_repairs:
        return inconclusive_verification(
            plan_id=record.plan_id,
            pot_id=pot_id,
            detail="Current-state verification of raw storage repairs is unsupported; the execution receipt remains authoritative.",
            unsupported=({"capability": "verification.record_repairs"},),
        )
    query_time = datetime.now(timezone.utc)
    before_versions = _subgraph_versions(backend, pot_id)
    if (
        record.execution is not None
        and before_versions["_global"] < record.execution.version_after
    ):
        return inconclusive_verification(
            plan_id=record.plan_id,
            pot_id=pot_id,
            detail="current revision contradicts the authoritative execution receipt",
        )
    claim_keys = _claim_keys_from_record(record)
    content = claim_content_readback(
        backend, pot_id=pot_id, record=record, query_time=query_time
    )
    if "protocols" in definition.extensions:
        from .protocol_validation import protocol_content_readback

        protocol = protocol_content_readback(backend, pot_id=pot_id, record=record)
        content["mismatches"] = tuple(content.get("mismatches", ())) + tuple(
            protocol.get("mismatches", ())
        )
        content["checked_protocol_entities"] = tuple(
            protocol.get("checked_entities", ())
        )
        if content["mismatches"]:
            content["outcome"] = "mismatched"
    after_quality = quality_snapshot(backend, pot_id=pot_id, definition=definition)
    after_versions = _subgraph_versions(backend, pot_id)
    if before_versions != after_versions:
        return inconclusive_verification(
            plan_id=record.plan_id,
            pot_id=pot_id,
            detail="graph revision changed during verification; no coherent result was established",
        )
    before_counts = _quality_count_map(before_quality)
    after_counts = _quality_count_map(after_quality)
    deltas = _quality_count_delta(before_counts, after_counts) if before_quality else {}
    regressions = _quality_regressions(
        before_quality=before_quality, after_quality=after_quality, deltas=deltas
    )
    unsupported = tuple(after_quality["unsupported"])
    warnings = []
    if before_quality is None:
        warnings.append(
            "pre-commit quality baseline unavailable; quality regression comparison skipped"
        )
    outcome = content["outcome"]
    status, detail, recommended = "ok", None, None
    ok = outcome in {"matching", "superseded_retracted"}
    state = "unsupported" if unsupported else "completed"
    if outcome == "mismatched":
        status, detail = (
            "degraded",
            "persisted graph content differs from the committed mutation",
        )
        recommended = (
            "Inspect content_readback and graph history before proposing a correction."
        )
    elif outcome == "inconclusive":
        state, status = "deferred", "inconclusive"
        detail = "current read evidence is insufficient to verify the frozen mutation; inspect unresolved targets and temporal diagnostics"
        recommended = (
            "Inspect graph history; the original execution receipt remains unchanged."
        )
    elif regressions:
        ok, status = False, "degraded"
        detail = "quality findings increased after commit"
        recommended = (
            "Review affected graph quality reports before proposing corrections."
        )
    elif unsupported:
        status = "partial"
        warnings.append("some verification checks are unavailable for this backend")
        recommended = "Use graph status to inspect backend capabilities."
    elif after_quality["status"] not in {"ok", "empty"}:
        status = "watch"
        warnings.append(f"graph quality status is {after_quality['status']}")
    elif outcome == "superseded_retracted":
        status = "superseded_retracted"
        detail = "the exact committed incarnation now has a later closure or retirement"
    elif not content.get("checked_claim_record_ids"):
        status = "no_claims"
        detail = "plan committed no claim keys; authored entity content was checked when present"
        warnings.append("verification read back no claims: the plan asserted none")
        recommended = (
            "Inspect checked_entities for the authored content verified by this plan."
        )
    return GraphIngestionVerificationResult(
        ok=ok,
        status=status,
        state=state,
        outcome=outcome,
        plan_id=record.plan_id,
        pot_id=pot_id,
        claim_keys=claim_keys,
        checked_revision=after_versions["_global"],
        verified_at=query_time,
        readback_claim_keys=content.get("readback_claim_keys", ()),
        missing_claim_keys=content.get("missing_claim_keys", ()),
        readback_count=content.get("readback_count", 0),
        content_readback=content,
        quality_status=str(after_quality["status"]),
        quality_counts=after_counts,
        quality_delta=deltas,
        quality_regressions=regressions,
        checked_reports=tuple(after_quality["checked_reports"]),
        warnings=tuple(warnings),
        unsupported=unsupported,
        detail=detail,
        recommended_next_action=recommended,
        subgraph_versions=after_versions,
    )


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
