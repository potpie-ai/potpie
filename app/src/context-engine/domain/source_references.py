"""Source reference, freshness, and verification primitives.

The context graph stores compact project memory and pointers back to source
systems. This module owns the normalized shape used by resolver responses and
validation; actual source-system reads are still provided by adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

SOURCE_POLICIES = frozenset(
    {"references_only", "summary", "verify", "snippets", "full_if_needed"}
)
RESOLVE_MODES = frozenset({"fast", "balanced", "deep", "verify"})
SOURCE_ACCESS_STATES = frozenset(
    {"allowed", "unknown", "permission_denied", "source_unreachable", "missing"}
)
FRESHNESS_STATES = frozenset(
    {
        "fresh",
        "mostly_fresh",
        "stale",
        "needs_verification",
        "source_unreachable",
        "unknown",
    }
)
VERIFICATION_STATES = frozenset(
    {"verified", "unverified", "needs_verification", "verification_failed"}
)
SOURCE_SYNC_STATES = frozenset(
    {
        "synced",
        "stale",
        "needs_resync",
        "source_unreachable",
        "permission_denied",
        "unknown",
    }
)


@dataclass(slots=True)
class SourceReferenceRecord:
    """Agent-facing pointer to an external artifact or durable source location."""

    ref: str
    source_type: str
    source_system: str | None = None
    external_id: str | None = None
    uri: str | None = None
    retrieval_uri: str | None = None
    title: str | None = None
    summary: str | None = None
    fetchable: bool = False
    access: str = "unknown"
    last_seen_at: str | None = None
    last_verified_at: str | None = None
    verified_against: str | None = None
    freshness_ttl_hours: int | None = None
    freshness: str = "unknown"
    sync_status: str = "unknown"
    staleness_reason: str | None = None
    verification_state: str = "unverified"
    resolver_hint: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FreshnessReport:
    """Overall freshness summary for a context resolution result."""

    status: str = "unknown"
    last_graph_update: str | None = None
    last_source_verification: str | None = None
    stale_refs: list[str] = field(default_factory=list)
    needs_verification_refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourceFallback:
    """Explicit fallback returned when source-backed context is partial."""

    code: str
    message: str
    impact: str | None = None
    ref: str | None = None


def normalize_resolve_mode(value: str | None) -> str:
    mode = (value or "fast").strip().lower()
    return mode if mode in RESOLVE_MODES else "fast"


def normalize_source_policy(value: str | None) -> str:
    policy = (value or "references_only").strip().lower()
    return policy if policy in SOURCE_POLICIES else "references_only"


def validate_source_reference_properties(properties: dict[str, object]) -> list[str]:
    """Validate optional SourceReference freshness metadata.

    Required source identity fields are enforced by the ontology. This function
    handles Phase 2 resolver fields without requiring every source ref to be
    fetchable on day one.
    """
    errors: list[str] = []
    for key in ("last_seen_at", "last_verified_at"):
        value = properties.get(key)
        if value is not None and not _is_iso_datetime(value):
            errors.append(f"SourceReference.{key} must be an ISO 8601 timestamp")

    access = properties.get("access")
    if access is not None and str(access) not in SOURCE_ACCESS_STATES:
        allowed = ", ".join(sorted(SOURCE_ACCESS_STATES))
        errors.append(
            f"SourceReference.access {access!r} is invalid; allowed: {allowed}"
        )

    freshness = properties.get("freshness")
    if freshness is not None and str(freshness) not in FRESHNESS_STATES:
        allowed = ", ".join(sorted(FRESHNESS_STATES))
        errors.append(
            f"SourceReference.freshness {freshness!r} is invalid; allowed: {allowed}"
        )

    verification = properties.get("verification_state")
    if verification is not None and str(verification) not in VERIFICATION_STATES:
        allowed = ", ".join(sorted(VERIFICATION_STATES))
        errors.append(
            "SourceReference.verification_state "
            f"{verification!r} is invalid; allowed: {allowed}"
        )

    sync_status = properties.get("sync_status")
    if sync_status is not None and str(sync_status) not in SOURCE_SYNC_STATES:
        allowed = ", ".join(sorted(SOURCE_SYNC_STATES))
        errors.append(
            f"SourceReference.sync_status {sync_status!r} is invalid; allowed: {allowed}"
        )

    freshness_ttl = properties.get("freshness_ttl_hours")
    if freshness_ttl is not None:
        try:
            ttl_value = int(str(freshness_ttl))
        except (TypeError, ValueError):
            errors.append("SourceReference.freshness_ttl_hours must be an integer")
        else:
            if ttl_value <= 0:
                errors.append("SourceReference.freshness_ttl_hours must be positive")

    return errors


def source_ref_key(
    source_type: str | None,
    external_id: str | None,
    *,
    source_system: str | None = None,
    uri: str | None = None,
) -> str:
    source = _compact(source_system) or _compact(source_type) or "unknown"
    ext = _compact(external_id) or _compact(uri) or "unknown"
    return f"{source}:{ext}"


def source_reference_from_mapping(data: dict[str, Any]) -> SourceReferenceRecord | None:
    """Build a normalized source ref from a loose provider/search row."""
    source_type = _first_str(
        data,
        "source_type",
        "ref_type",
        "kind",
        "type",
        default="unknown",
    )
    source_system = _first_str(data, "source_system", "provider", "system")
    external_id = _first_str(
        data,
        "external_id",
        "source_ref",
        "artifact_ref",
        "identifier",
        "uuid",
    )
    uri = _first_str(data, "retrieval_uri", "uri", "url", "source_uri")
    ref = _first_str(data, "ref", default=None) or source_ref_key(
        source_type,
        external_id,
        source_system=source_system,
        uri=uri,
    )
    if ref == "unknown:unknown":
        return None

    retrieval_uri = _first_str(data, "retrieval_uri", "uri", "url", "source_uri")
    access = _first_str(data, "access", default="unknown") or "unknown"
    if access not in SOURCE_ACCESS_STATES:
        access = "unknown"
    freshness = _first_str(data, "freshness", default="unknown") or "unknown"
    if freshness not in FRESHNESS_STATES:
        freshness = "unknown"
    verification_state = (
        _first_str(data, "verification_state", default="unverified") or "unverified"
    )
    if verification_state not in VERIFICATION_STATES:
        verification_state = "unverified"
    sync_status = _first_str(data, "sync_status", default="unknown") or "unknown"
    if sync_status not in SOURCE_SYNC_STATES:
        sync_status = "unknown"

    return SourceReferenceRecord(
        ref=ref,
        source_type=source_type or "unknown",
        source_system=source_system,
        external_id=external_id,
        uri=uri,
        retrieval_uri=retrieval_uri,
        title=_first_str(data, "title", "name", "headline"),
        summary=_first_str(data, "summary", "fact"),
        fetchable=bool(retrieval_uri),
        access=access,
        last_seen_at=_first_str(data, "last_seen_at", "created_at", "date"),
        last_verified_at=_first_str(data, "last_verified_at"),
        verified_against=_first_str(data, "verified_against"),
        freshness_ttl_hours=_first_int(data, "freshness_ttl_hours"),
        freshness=freshness,
        sync_status=sync_status,
        staleness_reason=_first_str(data, "staleness_reason"),
        verification_state=verification_state,
        resolver_hint={
            key: value
            for key, value in {
                "source_type": source_type,
                "source_system": source_system,
                "external_id": external_id,
                "retrieval_uri": retrieval_uri,
            }.items()
            if value
        },
    )


def dedupe_source_references(
    refs: Iterable[SourceReferenceRecord],
) -> list[SourceReferenceRecord]:
    out: list[SourceReferenceRecord] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.ref in seen:
            continue
        seen.add(ref.ref)
        out.append(ref)
    return out


def assess_freshness(refs: Iterable[SourceReferenceRecord]) -> FreshnessReport:
    refs_list = list(refs)
    if not refs_list:
        return FreshnessReport(status="unknown")

    latest_seen = _latest_timestamp(ref.last_seen_at for ref in refs_list)
    latest_verified = _latest_timestamp(ref.last_verified_at for ref in refs_list)
    stale = [ref.ref for ref in refs_list if ref.freshness == "stale"]
    unreachable = [
        ref.ref
        for ref in refs_list
        if ref.freshness == "source_unreachable"
        or ref.sync_status == "source_unreachable"
        or ref.access == "source_unreachable"
    ]
    needs = [
        ref.ref
        for ref in refs_list
        if ref.verification_state in {"unverified", "needs_verification"}
        or ref.freshness in {"unknown", "needs_verification"}
        or ref.sync_status in {"unknown", "needs_resync"}
    ]

    if unreachable:
        status = "source_unreachable"
    elif stale:
        status = "stale"
    elif needs:
        status = "needs_verification"
    elif latest_verified:
        status = "fresh"
    else:
        status = "unknown"

    return FreshnessReport(
        status=status,
        last_graph_update=latest_seen,
        last_source_verification=latest_verified,
        stale_refs=stale,
        needs_verification_refs=needs,
    )


def source_policy_fallbacks(
    *,
    source_policy: str,
    refs: Iterable[SourceReferenceRecord],
) -> list[SourceFallback]:
    """Explain what the resolver could not source-verify yet."""
    policy = normalize_source_policy(source_policy)
    refs_list = list(refs)
    fallbacks: list[SourceFallback] = []

    if policy == "references_only":
        return fallbacks

    if not refs_list:
        return [
            SourceFallback(
                code="no_source_references",
                message="No source references were available for this context.",
                impact="The answer can only use graph memory and provider results.",
            )
        ]

    fetchable = [ref for ref in refs_list if ref.fetchable]
    if not fetchable:
        fallbacks.append(
            SourceFallback(
                code="source_resolver_unavailable",
                message=(
                    "Source-backed reads were requested, but no fetchable resolver "
                    "references were available."
                ),
                impact="Use the returned references to inspect authoritative sources.",
            )
        )

    if policy == "verify":
        for ref in refs_list:
            if ref.verification_state != "verified":
                fallbacks.append(
                    SourceFallback(
                        code="source_unverified",
                        message="The graph reference has not been verified against source truth.",
                        impact="Treat this evidence as orientation, not final truth.",
                        ref=ref.ref,
                    )
                )

    return fallbacks


def _compact(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _first_str(
    data: dict[str, Any],
    *keys: str,
    default: str | None = None,
) -> str | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _first_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            return int(str(value))
        except (TypeError, ValueError):
            continue
    return None


def _is_iso_datetime(value: object) -> bool:
    if isinstance(value, datetime):
        return True
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _latest_timestamp(values: Iterable[str | None]) -> str | None:
    latest: datetime | None = None
    latest_raw: str | None = None
    for value in values:
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        if latest is None or parsed > latest:
            latest = parsed
            latest_raw = (
                parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            )
    return latest_raw
