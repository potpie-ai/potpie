"""Normalized inbound context events (reconciliation domain)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from domain.actor import Actor
from domain.ports.pot_resolution import ResolvedPot


@dataclass(frozen=True, slots=True)
class EventScope:
    """Pot + provider-scoped repo identity for an event."""

    pot_id: str
    provider: str
    provider_host: str
    repo_name: str


# Raw ingest for pots with no linked GitHub repo (``context_graph_pots`` without ``primary_repo_name``).
STANDALONE_POT_PROVIDER = "standalone"
STANDALONE_POT_HOST = "context"
STANDALONE_POT_REPO = "_"


def event_scope_from_resolved_pot(pot_id: str, resolved: ResolvedPot) -> EventScope:
    """Build scope for ledger/events; standalone pots use a synthetic repo placeholder."""
    primary = resolved.primary_repo()
    if primary is not None:
        return EventScope(
            pot_id=pot_id,
            provider=primary.provider,
            provider_host=primary.provider_host,
            repo_name=primary.repo_name,
        )
    return EventScope(
        pot_id=pot_id,
        provider=STANDALONE_POT_PROVIDER,
        provider_host=STANDALONE_POT_HOST,
        repo_name=STANDALONE_POT_REPO,
    )


@dataclass(frozen=True, slots=True)
class EventRef:
    """Stable reference to a persisted or in-flight context event."""

    event_id: str
    source_system: str
    pot_id: str


@dataclass(slots=True)
class ContextEvent:
    """Canonical inbound event after normalization."""

    event_id: str
    source_system: str
    event_type: str
    action: str
    pot_id: str
    provider: str
    provider_host: str
    repo_name: str
    source_id: str
    """Deterministic id for deduplication within scope (e.g. ``pr_42_merged``)."""
    source_event_id: str | None = None
    """External provider event id when available."""
    artifact_refs: list[str] = field(default_factory=list)
    occurred_at: datetime | None = None
    received_at: datetime | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    ingestion_kind: str | None = None
    """Event family such as ``agent_reconciliation`` or ``raw_episode``; persisted events are agent-planned."""
    idempotency_key: str | None = None
    """Optional client idempotency token (raw_episode family)."""
    source_channel: str | None = None
    """Inbound surface (``cli``, ``http``, ``webhook``); stored on ``context_events``."""
    actor: Actor | None = None
    """Principal that submitted the event (user + surface + client)."""
