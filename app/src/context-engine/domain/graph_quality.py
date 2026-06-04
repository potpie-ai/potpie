"""Graph quality, drift, and maintenance policy for agent-facing context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Iterable

from domain.ontology import (
    object_counterparty_uuid_for_edge,
    predicate_family_for_episodic_supersede,
    temporal_subject_key_for_edge,
)
from domain.source_references import SourceFallback, SourceReferenceRecord

if TYPE_CHECKING:
    from domain.intelligence_models import CoverageReport

FACT_FAMILY_FRESHNESS_TTL_HOURS: dict[str, int] = {
    "ownership": 24 * 14,
    "code": 24 * 7,
    "change": 24 * 30,
    "decision": 24 * 180,
    "discussion": 24 * 90,
    "document": 24 * 30,
    "runbook": 24 * 30,
    "service": 24 * 14,
    "environment": 24 * 7,
    "deployment": 24 * 3,
    "incident": 12,
    "alert": 2,
    "fix": 24 * 90,
    "bugpattern": 24 * 90,
    "diagnosticsignal": 24 * 30,
    "preference": 24 * 60,
    "agentinstruction": 24 * 30,
    "unknown": 24 * 30,
}

SOURCE_OF_TRUTH_POLICIES: dict[str, str] = {
    "ownership": "authoritative_external_truth",
    "code": "authoritative_code_truth",
    "change": "authoritative_external_truth",
    "decision": "canonicalized_memory",
    "discussion": "authoritative_external_truth",
    "document": "authoritative_external_truth",
    "runbook": "authoritative_external_truth",
    "service": "canonicalized_memory",
    "environment": "authoritative_external_truth",
    "deployment": "authoritative_external_truth",
    "incident": "authoritative_external_truth",
    "alert": "authoritative_external_truth",
    "fix": "canonicalized_memory",
    "bugpattern": "canonicalized_memory",
    "diagnosticsignal": "canonicalized_memory",
    "preference": "soft_inference",
    "agentinstruction": "authoritative_code_truth",
    "unknown": "canonicalized_memory",
}

MAINTENANCE_JOB_FAMILIES: tuple[str, ...] = (
    "verify_entity",
    "verify_edge",
    "refresh_scope",
    "resync_source_scope",
    "rebuild_scope_from_truth",
    "repair_code_bridges",
    "expire_stale_facts",
    "compact_or_archive_evidence",
    "resolve_alias_candidates",
    "cleanup_orphans",
    "classify_modified_edges",
)


@dataclass(slots=True)
class GraphQualityIssue:
    code: str
    message: str
    severity: str = "warning"
    refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphQualityReport:
    status: str = "unknown"
    metrics: dict[str, int] = field(default_factory=dict)
    issues: list[GraphQualityIssue] = field(default_factory=list)
    recommended_maintenance: list[dict[str, str]] = field(default_factory=list)
    policy: dict[str, object] = field(default_factory=dict)
    # Open predicate-family conflicts (see detect_family_conflicts).
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    # Auto-supersession or other automatic resolutions (informational).
    resolved_conflicts: list[dict[str, Any]] = field(default_factory=list)


def fact_family_for_source_type(source_type: str | None) -> str:
    value = (source_type or "unknown").strip().lower().replace("_", "")
    if value in {"pullrequest", "pr", "commit", "issue", "ticket"}:
        return "change"
    if value in {"diagnosticsignal"}:
        return "diagnosticsignal"
    if value in {"bugpattern"}:
        return "bugpattern"
    if value in {"agentinstruction"}:
        return "agentinstruction"
    if value in FACT_FAMILY_FRESHNESS_TTL_HOURS:
        return value
    return "unknown"


def freshness_ttl_hours_for_source_type(source_type: str | None) -> int:
    family = fact_family_for_source_type(source_type)
    return FACT_FAMILY_FRESHNESS_TTL_HOURS[family]


def source_of_truth_for_source_type(source_type: str | None) -> str:
    family = fact_family_for_source_type(source_type)
    return SOURCE_OF_TRUTH_POLICIES[family]


def make_source_ref(source_type: str, **kwargs) -> "SourceReferenceRecord":
    """Factory that pre-stamps freshness_ttl_hours from the family policy.

    Callers can override by passing freshness_ttl_hours explicitly.
    """
    from domain.source_references import SourceReferenceRecord

    kwargs.setdefault("freshness_ttl_hours", freshness_ttl_hours_for_source_type(source_type))
    return SourceReferenceRecord(source_type=source_type, **kwargs)


def is_reference_stale(
    ref: SourceReferenceRecord,
    *,
    now: datetime | None = None,
) -> bool:
    if ref.freshness == "stale" or ref.sync_status == "stale":
        return True
    raw = ref.last_verified_at or ref.last_seen_at
    if not raw:
        return False
    parsed = _parse_iso(raw)
    if parsed is None:
        return False
    ttl = ref.freshness_ttl_hours or freshness_ttl_hours_for_source_type(
        ref.source_type
    )
    current = now or datetime.now(timezone.utc)
    return current - parsed > timedelta(hours=ttl)


def assess_graph_quality(
    *,
    refs: Iterable[SourceReferenceRecord],
    coverage: "CoverageReport",
    fallbacks: Iterable[SourceFallback],
    now: datetime | None = None,
) -> GraphQualityReport:
    refs_list = list(refs)
    fallback_list = list(fallbacks)
    stale_refs = [ref.ref for ref in refs_list if is_reference_stale(ref, now=now)]
    needs_verification_refs = [
        ref.ref
        for ref in refs_list
        if ref.verification_state in {"unverified", "needs_verification"}
    ]
    verification_failed_refs = [
        ref.ref for ref in refs_list if ref.verification_state == "verification_failed"
    ]
    source_access_gap_refs = [
        ref.ref
        for ref in refs_list
        if ref.access in {"permission_denied", "source_unreachable", "missing"}
    ]

    metrics = {
        "source_ref_count": len(refs_list),
        "stale_ref_count": len(stale_refs),
        "needs_verification_ref_count": len(needs_verification_refs),
        "verification_failed_ref_count": len(verification_failed_refs),
        "source_access_gap_count": len(source_access_gap_refs),
        "missing_coverage_count": len(coverage.missing),
        "fallback_count": len(fallback_list),
    }
    issues: list[GraphQualityIssue] = []
    recommended: list[dict[str, str]] = []

    if stale_refs:
        issues.append(
            GraphQualityIssue(
                code="stale_facts",
                message="Some source references are beyond their freshness policy.",
                severity="warning",
                refs=stale_refs,
            )
        )
        recommended.append(
            {
                "job": "expire_stale_facts",
                "reason": "Stale references should be refreshed, superseded, or expired.",
            }
        )

    if verification_failed_refs or needs_verification_refs:
        issues.append(
            GraphQualityIssue(
                code="verification_gap",
                message="Some graph facts have not been verified against source truth.",
                severity="warning",
                refs=verification_failed_refs or needs_verification_refs,
            )
        )
        recommended.append(
            {
                "job": "verify_entity",
                "reason": "Verify high-value facts before agents rely on them.",
            }
        )

    if source_access_gap_refs:
        issues.append(
            GraphQualityIssue(
                code="source_access_gap",
                message="Some source references are missing, unreachable, or permission-denied.",
                severity="error",
                refs=source_access_gap_refs,
            )
        )
        recommended.append(
            {
                "job": "resync_source_scope",
                "reason": "Reconnect or resync source systems before trusting affected facts.",
            }
        )

    if coverage.missing:
        issues.append(
            GraphQualityIssue(
                code="incomplete_coverage",
                message="Some planned context families returned no data.",
                severity="warning",
                refs=list(coverage.missing),
            )
        )
        recommended.append(
            {
                "job": "refresh_scope",
                "reason": "Refresh this pot or scope to improve context coverage.",
            }
        )

    status = "good"
    if coverage.status == "empty" and not refs_list:
        status = "unknown"
    elif source_access_gap_refs or verification_failed_refs:
        status = "degraded"
    elif stale_refs or needs_verification_refs or coverage.missing or fallback_list:
        status = "watch"

    return GraphQualityReport(
        status=status,
        metrics=metrics,
        issues=issues,
        recommended_maintenance=_dedupe_recommendations(recommended),
        policy={
            "maintenance_jobs": list(MAINTENANCE_JOB_FAMILIES),
            "freshness_ttl_hours": dict(FACT_FAMILY_FRESHNESS_TTL_HOURS),
            "source_of_truth": dict(SOURCE_OF_TRUTH_POLICIES),
        },
    )


def temporal_supersession_resolved_issue(
    *,
    group_id: str,
    superseded_edge_uuid: str,
    superseding_edge_uuid: str,
    predicate_family: str,
) -> GraphQualityIssue:
    """Use when auto-supersede closes an older fact; surfaces as resolved temporal conflict."""
    return GraphQualityIssue(
        code="temporal_supersession_resolved",
        message=(
            f"Temporal contradiction in pot {group_id!r} was resolved by supersession "
            f"({predicate_family}): edge {superseded_edge_uuid} superseded by "
            f"{superseding_edge_uuid}."
        ),
        severity="info",
        refs=[superseded_edge_uuid, superseding_edge_uuid, group_id],
    )


@dataclass(frozen=True, slots=True)
class EpisodicEdgeConflictInput:
    """Live episodic ``RELATES_TO`` row used for predicate-family conflict detection."""

    uuid: str
    name: str
    source_uuid: str
    target_uuid: str
    valid_at: datetime | None = None
    created_at: datetime | None = None
    #: Target node labels (disambiguate overloaded edges such as ``CHOSE``).
    target_labels: tuple[str, ...] | None = None


def _effective_edge_time(edge: EpisodicEdgeConflictInput) -> datetime | None:
    return edge.valid_at or edge.created_at


def classify_predicate_family_pair(
    older: EpisodicEdgeConflictInput,
    newer: EpisodicEdgeConflictInput,
    *,
    family: str,
    subject_uuid: str,
) -> dict[str, Any] | None:
    """Return a conflict record for two *live* edges with different object endpoints, or None.

    ``older`` / ``newer`` refer to observation time: ``older`` has the earlier
    ``valid_at`` (fallback ``created_at``).
    """
    va_o = _effective_edge_time(older)
    va_n = _effective_edge_time(newer)
    if va_o is not None and va_n is not None:
        if va_o == va_n:
            ctype = "contradiction"
            auto_resolvable = False
            suggested = "human_review"
            severity = "blocking"
        elif va_n > va_o:
            ctype = "supersession_pending"
            auto_resolvable = True
            suggested = "supersede_older"
            severity = "warning"
        else:
            return None
    elif va_o is None and va_n is None:
        ctype = "overlap"
        auto_resolvable = False
        suggested = "human_review"
        severity = "warning"
    else:
        ctype = "overlap"
        auto_resolvable = False
        suggested = "human_review"
        severity = "warning"

    return {
        "family": family,
        "subject_uuid": subject_uuid,
        "edge_a_uuid": older.uuid,
        "edge_b_uuid": newer.uuid,
        "conflict_type": ctype,
        "auto_resolvable": auto_resolvable,
        "suggested_action": suggested,
        "severity": severity,
        "kind": "conflict",
    }


def detect_family_conflicts(edges: Iterable[EpisodicEdgeConflictInput]) -> list[dict[str, Any]]:
    """Find contradicting live edges per predicate family + subject (pairwise, O(k²) per bucket)."""
    from collections import defaultdict

    rows = list(edges)
    buckets: dict[tuple[str, str], list[EpisodicEdgeConflictInput]] = defaultdict(list)
    for e in rows:
        fam = predicate_family_for_episodic_supersede(e.name, e.target_labels)
        if fam is None:
            continue
        sk = temporal_subject_key_for_edge(
            e.name, e.source_uuid, e.target_uuid, predicate_family=fam
        )
        if sk is None:
            continue
        buckets[(fam, sk)].append(e)

    out: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    for (fam, subj), group in buckets.items():
        by_obj: dict[str, list[EpisodicEdgeConflictInput]] = defaultdict(list)
        for e in group:
            ok = object_counterparty_uuid_for_edge(
                e.name, e.source_uuid, e.target_uuid, predicate_family=fam
            )
            if ok is None:
                continue
            by_obj[ok].append(e)
        if sum(1 for k in by_obj if k) < 2:
            continue

        flat = list(group)
        for i, ei in enumerate(flat):
            for ej in flat[i + 1 :]:
                oi = object_counterparty_uuid_for_edge(
                    ei.name,
                    ei.source_uuid,
                    ei.target_uuid,
                    predicate_family=fam,
                )
                oj = object_counterparty_uuid_for_edge(
                    ej.name,
                    ej.source_uuid,
                    ej.target_uuid,
                    predicate_family=fam,
                )
                if not oi or not oj or oi == oj:
                    continue
                ti, tj = _effective_edge_time(ei), _effective_edge_time(ej)
                if ti is not None and tj is not None:
                    older, newer = (ei, ej) if ti <= tj else (ej, ei)
                else:
                    older, newer = sorted(
                        (ei, ej), key=lambda e: (_effective_edge_time(e) is None, e.uuid)
                    )
                key = (
                    fam,
                    subj,
                    min(older.uuid, newer.uuid),
                    max(older.uuid, newer.uuid),
                )
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                rec = classify_predicate_family_pair(
                    older, newer, family=fam, subject_uuid=subj
                )
                if rec is not None:
                    out.append(rec)
    return out


def _parse_iso(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _dedupe_recommendations(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for item in items:
        key = item.get("job", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
