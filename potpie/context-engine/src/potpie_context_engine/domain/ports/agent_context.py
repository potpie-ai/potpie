"""``AgentContextPort`` — the four-tool agent contract.

This is the single public agent surface for CLI and MCP. There are exactly four
tools and there will only ever be four:

    context_resolve   resolve(ResolveRequest) -> AgentEnvelope
    context_search    search(SearchRequest)   -> AgentEnvelope
    context_record    record(RecordRequest)   -> RecordReceipt
    context_status    status(StatusRequest)   -> StatusReport

New use cases become new ``intent`` / ``include`` families, ``record_type``
values, or skills — **never** new tools. ``AgentContextPort`` composes the
two services: ``resolve``/``search``/``record`` delegate to ``GraphService``;
``status`` composes ``GraphService`` data-plane status with
``PotManagementService`` aggregate status. Product skill readiness is added by
root Potpie.

Request fields mirror the documented contract: ``pot_id``, ``intent``,
``include``, ``scope``, ``mode`` (fast/balanced/verify/deep — retrieval depth),
and ``source_policy``. The intent/include/record-type *vocabulary* stays in
``domain.agent_context_port`` (one source of truth); these DTOs only carry it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol

from potpie_context_engine.domain.agent_envelope import AgentEnvelope

# Retrieval depth. Replaces the old ``goal=ANSWER/INVESTIGATE`` surface: depth
# is a dial on one read path, not a different read path.
ResolveMode = str  # "fast" | "balanced" | "verify" | "deep"


@dataclass(frozen=True, slots=True)
class ResolveRequest:
    """``context_resolve`` — bounded context wrap for a task."""

    pot_id: str
    task: str | None = None
    intent: str | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    scope: Mapping[str, Any] = field(default_factory=dict)
    mode: ResolveMode = "fast"
    source_policy: str = "references_only"
    max_items: int = 12
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    include_invalidated: bool = False
    freshness_preference: str = "balanced"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """``context_search`` — narrow follow-up lookup on a known phrase/entity."""

    pot_id: str
    query: str
    include: tuple[str, ...] = ()
    scope: Mapping[str, Any] = field(default_factory=dict)
    mode: ResolveMode = "fast"
    source_policy: str = "references_only"
    max_items: int = 12
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecordRequest:
    """``context_record`` — write a durable project learning."""

    pot_id: str
    record_type: str
    summary: str
    details: Mapping[str, Any] = field(default_factory=dict)
    scope: Mapping[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    idempotency_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StatusRequest:
    """``context_status`` — cheap pot readiness + next-action aggregate."""

    pot_id: str
    intent: str | None = None
    harness: str | None = None  # agent harness, for the skill nudge
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecordReceipt:
    """Outcome of ``context_record``."""

    pot_id: str
    record_type: str
    accepted: bool
    record_id: str | None = None
    status: str = "recorded"  # recorded | duplicate | rejected | not_implemented
    mutations_applied: int = 0
    detail: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StatusReport:
    """Outcome of ``context_status`` — composed across the three services."""

    pot_id: str
    profile: str  # local | managed
    daemon_up: bool
    active_pot: str | None
    backend_ready: bool
    data_plane: Mapping[str, Any] = field(default_factory=dict)
    pot_summary: Mapping[str, Any] = field(default_factory=dict)
    recommended_next_action: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AgentContextPort(Protocol):
    """The four-tool agent contract. The whole public agent surface."""

    def resolve(self, request: ResolveRequest) -> AgentEnvelope: ...

    def search(self, request: SearchRequest) -> AgentEnvelope: ...

    def record(self, request: RecordRequest) -> RecordReceipt: ...

    def status(self, request: StatusRequest) -> StatusReport: ...


__all__ = [
    "AgentContextPort",
    "RecordReceipt",
    "RecordRequest",
    "ResolveMode",
    "ResolveRequest",
    "SearchRequest",
    "StatusReport",
    "StatusRequest",
]
