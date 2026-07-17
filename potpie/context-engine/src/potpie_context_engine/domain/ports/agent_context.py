"""``AgentContextPort`` — the four-tool agent contract.

This is the single public agent surface for CLI and MCP. There are exactly four
tools and there will only ever be four:

    context_resolve   resolve(ResolveRequest) -> AgentEnvelope
    context_search    search(SearchRequest)   -> AgentEnvelope
    context_record    record(RecordRequest)   -> RecordReceipt
    context_status    status(StatusRequest)   -> StatusReport

New use cases become new ``intent`` / ``include`` families, ``record_type``
values, or skills — **never** new tools. ``AgentContextPort`` composes the
three services: ``resolve``/``search``/``record`` delegate to ``GraphService``;
``status`` composes ``GraphService`` data-plane status + ``PotManagementService``
aggregate status + a ``SkillManager`` nudge.

Request fields mirror the documented contract: ``pot_id``, ``intent``,
``include``, ``scope``, ``mode`` (fast/balanced/verify/deep — retrieval depth),
and ``source_policy``. The intent/include/record-type *vocabulary* stays in
``potpie_context_engine.domain.agent_context_port`` (one source of truth); these DTOs only carry it.
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
class SkillNudge:
    """Advisory skills block embedded in ``context_status``.

    The only skill signal an agent ever sees: missing/outdated skills for its
    harness plus the exact CLI command a human runs to close the gap.
    Installation is never an agent action. Owned here (not in
    ``skill_manager``) because it is part of the status contract — defining it
    here keeps the ports layer acyclic.
    """

    agent: str
    missing: tuple[str, ...] = ()
    outdated: tuple[str, ...] = ()
    install_command: str | None = None


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
    skills: SkillNudge | None = None
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
    "SkillNudge",
    "StatusReport",
    "StatusRequest",
]
