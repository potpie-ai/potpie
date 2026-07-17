"""Value types for the unified source-connector contract.

Adding a new source — GitHub, Linear, Notion, Slack, Sentry — means
writing one connector module that implements
:class:`potpie_context_engine.domain.ports.source_connector.SourceConnectorPort` and returns
the value types defined here. The application layer never imports a
concrete source name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceCapability:
    """One ``(provider, source_kind)`` pairing the connector advertises.

    Mirrors :class:`potpie_context_engine.domain.source_resolution.ResolverCapabilityEntry` but
    extended with the additional verbs a connector may expose (webhooks,
    listing) so ``context_status`` can render a single manifest instead
    of three independent ones.

    Rebuild plan P0: dropped ``plan_capable`` along with ``propose_plan``.
    """

    provider: str
    source_kind: str
    policies: frozenset[str] = field(default_factory=frozenset)
    fetch_capable: bool = False
    list_capable: bool = False
    webhook_capable: bool = False
    sync_capable: bool = False
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class ConnectorScope:
    """Pot-scoped instructions for ``list_artifacts`` / backfill calls.

    The connector decides what fields it understands. ``scope`` is a free
    dict so connector-specific knobs (Linear team id, GitHub repo full name,
    Notion workspace id) don't leak into a shared schema.
    """

    pot_id: str
    scope: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConnectorManifest:
    """Per-pot manifest returned by the registry.

    Drives ``context_status`` and gives agents enough information to
    decide which connector to talk to for a given resolve / record call.
    """

    kind: str
    enabled: bool
    capabilities: tuple[SourceCapability, ...] = ()
    health: str = "unknown"
    reason: str | None = None


def merge_capability_policies(
    capabilities: list[SourceCapability],
) -> list[SourceCapability]:
    """Collapse duplicate ``(provider, source_kind)`` entries by union'ing policies."""
    merged: dict[tuple[str, str], SourceCapability] = {}
    for cap in capabilities:
        key = (cap.provider.lower(), cap.source_kind.lower())
        existing = merged.get(key)
        if existing is None:
            merged[key] = cap
            continue
        merged[key] = SourceCapability(
            provider=existing.provider,
            source_kind=existing.source_kind,
            policies=existing.policies | cap.policies,
            fetch_capable=existing.fetch_capable or cap.fetch_capable,
            list_capable=existing.list_capable or cap.list_capable,
            webhook_capable=existing.webhook_capable or cap.webhook_capable,
            sync_capable=existing.sync_capable or cap.sync_capable,
            notes=existing.notes or cap.notes,
        )
    return sorted(merged.values(), key=lambda c: (c.provider, c.source_kind))
