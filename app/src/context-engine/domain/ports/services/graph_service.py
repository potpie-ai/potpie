"""``GraphService`` — the data-plane service.

The data plane behind three of the four tools: ``resolve``, ``search``, and
``record``. It owns the readers, ranking, record lowering, and envelope
assembly, and it talks to a ``GraphBackend`` (claim_query + mutation + semantic)
— never to a store directly.

``GraphService`` is *not* the agent contract. ``AgentContextPort`` is the public
4-tool surface; ``GraphService`` is the data-plane half it composes (the other
halves being ``PotManagementService`` and ``SkillManager``). The two share the
same request DTOs but sit at different altitudes: ``GraphService.resolve`` does
the read work; ``AgentContextPort.resolve`` is the public binding to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from domain.agent_envelope import AgentEnvelope
from domain.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)


@dataclass(frozen=True, slots=True)
class DataPlaneStatus:
    """Data-plane half of ``context_status``: backend readiness + coverage."""

    pot_id: str
    backend_profile: str
    backend_ready: bool
    reader_backed_includes: tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    freshness: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    detail: str | None = None


class GraphService(Protocol):
    """Data plane for resolve/search/record over a ``GraphBackend``."""

    def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        """Expand intent → includes, run readers over the backend, rank, and
        assemble one ``AgentEnvelope``."""
        ...

    def search(self, request: SearchRequest) -> AgentEnvelope:
        """Narrow lookup; same envelope shape as ``resolve``."""
        ...

    def record(self, request: RecordRequest) -> RecordReceipt:
        """Lower a durable record into a mutation plan and apply it through the
        backend's mutation port."""
        ...

    def data_plane_status(self, pot_id: str) -> DataPlaneStatus:
        """Backend readiness + coverage for ``context_status``."""
        ...


__all__ = ["DataPlaneStatus", "GraphService"]
