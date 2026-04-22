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
