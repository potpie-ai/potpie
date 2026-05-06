"""Context status read model: pot readiness, source health, event ledger, resolver capabilities.

Backs ``POST /api/v2/context/status`` so agents, CLIs, UIs, and SDKs can read a
single trust/readiness envelope without scraping multiple internal APIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Protocol, Sequence


@dataclass(slots=True)
class StatusSource:
    """Compact view of one attached pot source for status responses."""

    source_id: str
    pot_id: str
    source_kind: str
    provider: str
    provider_host: str | None = None
    sync_enabled: bool = True
    sync_mode: str | None = None
    last_sync_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None
    last_verified_at: datetime | None = None
    verification_state: str | None = None
    scope_summary: str | None = None
    health_score: str | None = None


@dataclass(slots=True)
class EventLedgerHealth:
    """Aggregate event-pipeline health for a pot."""

    counts: dict[str, int] = field(default_factory=dict)
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    recent_errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ReconciliationLedgerHealth:
    """Reconciliation pipeline health: runs + apply steps for a pot."""

    run_counts: dict[str, int] = field(default_factory=dict)
    step_counts: dict[str, int] = field(default_factory=dict)
    last_run_success_at: datetime | None = None
    last_run_failure_at: datetime | None = None
    recent_failed_runs: list[dict[str, Any]] = field(default_factory=list)
    stuck_step_samples: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class MaintenanceJob:
    """Recommended maintenance action derived from status inputs."""

    action: str
    reason: str
    severity: str = "info"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceCapabilityMatrixEntry:
    """One (provider, source_kind) row in the per-source capability matrix."""

    provider: str
    source_kind: str
    capabilities: list[ResolverCapability] = field(default_factory=list)


@dataclass(slots=True)
class ResolverCapability:
    """Whether a ``source_policy`` mode is wired end-to-end on this server."""

    policy: str
    available: bool
    reason: str | None = None


# Default capability matrix. ``references_only`` is implemented by the baseline
# resolver; richer modes are placeholders until source resolvers are wired
# (Phase 4 of planning-next-steps.md).
DEFAULT_RESOLVER_CAPABILITIES: tuple[ResolverCapability, ...] = (
    ResolverCapability(
        policy="references_only",
        available=True,
        reason=None,
    ),
    ResolverCapability(
        policy="summary",
        available=False,
        reason="Source-backed summary resolvers are not configured on this server.",
    ),
    ResolverCapability(
        policy="verify",
        available=False,
        reason="Source-of-truth verification resolvers are not configured on this server.",
    ),
    ResolverCapability(
        policy="snippets",
        available=False,
        reason="Bounded source snippet resolvers are not configured on this server.",
    ),
)


def resolver_capabilities_to_payload(
    capabilities: Iterable[ResolverCapability],
) -> list[dict[str, Any]]:
    return [asdict(c) for c in capabilities]


# Per-(provider, source_kind) capability matrix. Today no source-backed
# resolvers are wired (Phase 4 of planning-next-steps.md), so all source kinds
# advertise the same baseline as the global default. Once richer resolvers land
# this map flips entries to ``available=True`` per source kind that supports
# them, letting agents see *why* one source can do ``snippets`` while another
# can only do ``references_only``.
_SOURCE_CAPABILITY_OVERRIDES: dict[tuple[str, str], dict[str, ResolverCapability]] = {}


class ResolverCapabilityAdvertiser(Protocol):
    """Narrow read-only view of :class:`SourceResolverPort.capabilities`.

    Declared here (not imported) to avoid a cycle between ``context_status``
    and ``domain.ports.source_resolver``. The only thing the status layer
    needs is the list of ``(provider, source_kind, policies)`` triples.
    """

    def capabilities(self) -> Sequence[Any]:
        ...


def _resolver_capability_map(
    resolver: ResolverCapabilityAdvertiser | None,
) -> dict[tuple[str, str], frozenset[str]]:
    if resolver is None:
        return {}
    out: dict[tuple[str, str], frozenset[str]] = {}
    for entry in resolver.capabilities():
        provider = getattr(entry, "provider", "")
        source_kind = getattr(entry, "source_kind", "")
        policies = getattr(entry, "policies", frozenset())
        if not provider or not source_kind:
            continue
        out[(provider.lower(), source_kind.lower())] = frozenset(policies)
    return out


def source_capabilities_for(
    source: "StatusSource",
    *,
    resolver: ResolverCapabilityAdvertiser | None = None,
) -> list[ResolverCapability]:
    """Resolver capability matrix for one source.

    When a ``resolver`` is supplied, its :meth:`capabilities` output upgrades
    the baseline: any policy the resolver advertises for
    ``(source.provider, source.source_kind)`` is reported as available with
    no placeholder reason. ``sync_enabled=False`` still flips every entry to
    unavailable so agents do not request a policy that cannot be served right
    now.
    """
    overrides = _SOURCE_CAPABILITY_OVERRIDES.get(
        (source.provider, source.source_kind), {}
    )
    advertised = _resolver_capability_map(resolver).get(
        (source.provider.lower(), source.source_kind.lower()),
        frozenset(),
    )
    out: list[ResolverCapability] = []
    for default in DEFAULT_RESOLVER_CAPABILITIES:
        cap = overrides.get(default.policy, default)
        if default.policy in advertised and not cap.available:
            cap = ResolverCapability(policy=default.policy, available=True, reason=None)
        if not source.sync_enabled and cap.available:
            cap = ResolverCapability(
                policy=cap.policy,
                available=False,
                reason="Source sync is disabled.",
            )
        out.append(cap)
    return out


def status_source_to_payload(
    source: StatusSource,
    *,
    resolver: ResolverCapabilityAdvertiser | None = None,
) -> dict[str, Any]:
    payload = asdict(source)
    for k in (
        "last_sync_at",
        "last_success_at",
        "last_error_at",
        "last_verified_at",
    ):
        v = payload.get(k)
        if isinstance(v, datetime):
            payload[k] = v.isoformat()
    payload["capabilities"] = resolver_capabilities_to_payload(
        source_capabilities_for(source, resolver=resolver)
    )
    return payload


def reconciliation_ledger_health_to_payload(
    health: "ReconciliationLedgerHealth",
) -> dict[str, Any]:
    return {
        "run_counts": dict(health.run_counts),
        "step_counts": dict(health.step_counts),
        "last_run_success_at": (
            health.last_run_success_at.isoformat()
            if health.last_run_success_at
            else None
        ),
        "last_run_failure_at": (
            health.last_run_failure_at.isoformat()
            if health.last_run_failure_at
            else None
        ),
        "recent_failed_runs": list(health.recent_failed_runs),
        "stuck_step_samples": list(health.stuck_step_samples),
    }


def build_source_capability_matrix(
    sources: Sequence["StatusSource"],
    *,
    resolver: ResolverCapabilityAdvertiser | None = None,
) -> list["SourceCapabilityMatrixEntry"]:
    """Per-(provider, source_kind) capability matrix for the pot's attached sources.

    Deduplicates by lower-cased ``(provider, source_kind)``: multiple attached
    sources of the same kind collapse into one matrix row. Each row carries the
    full four-policy envelope so agents can see — at a glance — whether a
    particular provider/source_kind can serve ``summary``, ``verify``,
    ``snippets``, or only ``references_only``.
    """
    seen: dict[tuple[str, str], SourceCapabilityMatrixEntry] = {}
    for s in sources:
        key = (s.provider.lower(), s.source_kind.lower())
        if key in seen:
            continue
        seen[key] = SourceCapabilityMatrixEntry(
            provider=s.provider,
            source_kind=s.source_kind,
            capabilities=source_capabilities_for(s, resolver=resolver),
        )
    return list(seen.values())


def source_capability_matrix_to_payload(
    entries: Iterable["SourceCapabilityMatrixEntry"],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        out.append(
            {
                "provider": e.provider,
                "source_kind": e.source_kind,
                "capabilities": resolver_capabilities_to_payload(e.capabilities),
            }
        )
    return out


def derive_maintenance_jobs(
    *,
    event_ledger: "EventLedgerHealth",
    reconciliation: "ReconciliationLedgerHealth",
    open_conflicts: Sequence[dict[str, Any]] | None = None,
) -> list["MaintenanceJob"]:
    """Recommend concrete maintenance actions from the pot's health signals.

    Keeps the set small and operator-grade: each job names an action an operator
    can trigger (``events.replay``, ``maintenance.classify-modified-edges``,
    ``conflicts.resolve``) with the reason and a severity tag so UIs can sort.
    """
    jobs: list[MaintenanceJob] = []
    err_count = int(event_ledger.counts.get("error", 0))
    if err_count > 0:
        jobs.append(
            MaintenanceJob(
                action="events.replay",
                reason=f"{err_count} event(s) in error state; replay after investigating.",
                severity="warning",
                params={"event_count": err_count},
            )
        )
    failed_runs = int(reconciliation.run_counts.get("failed", 0))
    if failed_runs > 0:
        jobs.append(
            MaintenanceJob(
                action="reconciliation.retry_failed_runs",
                reason=f"{failed_runs} reconciliation run(s) failed; consider replaying associated events.",
                severity="warning",
                params={"run_count": failed_runs},
            )
        )
    stuck = len(reconciliation.stuck_step_samples)
    if stuck > 0:
        jobs.append(
            MaintenanceJob(
                action="reconciliation.retry_stuck_steps",
                reason=f"{stuck} apply step(s) stuck in applying/failed; retry or hard-reset after triage.",
                severity="warning",
                params={"step_count": stuck},
            )
        )
    conflicts = list(open_conflicts or [])
    hard_conflicts = [
        c for c in conflicts if isinstance(c, dict) and c.get("auto_resolvable") is False
    ]
    if hard_conflicts:
        jobs.append(
            MaintenanceJob(
                action="conflicts.resolve",
                reason=(
                    f"{len(hard_conflicts)} open predicate-family conflict(s) require manual resolution."
                ),
                severity="warning",
                params={"conflict_count": len(hard_conflicts)},
            )
        )
    return jobs


def maintenance_jobs_to_payload(
    jobs: Iterable["MaintenanceJob"],
) -> list[dict[str, Any]]:
    return [
        {
            "action": j.action,
            "reason": j.reason,
            "severity": j.severity,
            "params": dict(j.params),
        }
        for j in jobs
    ]


def event_ledger_health_to_payload(health: EventLedgerHealth) -> dict[str, Any]:
    return {
        "counts": dict(health.counts),
        "last_success_at": (
            health.last_success_at.isoformat() if health.last_success_at else None
        ),
        "last_error_at": (
            health.last_error_at.isoformat() if health.last_error_at else None
        ),
        "recent_errors": list(health.recent_errors),
    }


def derive_pot_last_success_at(
    sources: Sequence[StatusSource],
    ledger: EventLedgerHealth,
) -> datetime | None:
    """Pick the most recent successful ingestion across sources and the event ledger."""
    candidates: list[datetime] = []
    if ledger.last_success_at is not None:
        candidates.append(ledger.last_success_at)
    for s in sources:
        if s.last_success_at is not None:
            candidates.append(s.last_success_at)
        elif s.last_sync_at is not None and not s.last_error:
            candidates.append(s.last_sync_at)
    if not candidates:
        return None
    return max(candidates)
