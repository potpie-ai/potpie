"""``AgentContextPort`` implementation — composes the three services.

This is the public four-tool surface. ``resolve``/``search``/``record`` delegate
straight to the ``GraphService`` data plane; ``status`` is the only composite —
it joins ``GraphService`` data-plane status, ``PotManagementService`` control-
plane status, and a ``SkillManager`` nudge into one ``StatusReport``.

CLI and MCP bind here. The managed HTTP ingestion surface is a legacy adapter
while it migrates onto the host shell; it must not define new agent tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from potpie_context_core.agent_context_port import normalize_context_intent
from potpie_context_core.agent_envelope import AgentEnvelope
from potpie_context_core.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    StatusReport,
    StatusRequest,
)
from potpie_context_core.ports.graph_service import GraphService
from potpie_context_engine.domain.ports.services.pot_management import PotManagementService
from potpie_context_engine.domain.ports.services.skill_manager import SkillManager


@dataclass(slots=True)
class AgentContextService:
    """The 4-tool agent contract, composed over the three services."""

    graph: GraphService
    pots: PotManagementService
    skills: SkillManager
    profile: str = "local"

    def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        return self.graph.resolve(request)

    def search(self, request: SearchRequest) -> AgentEnvelope:
        return self.graph.search(request)

    def record(self, request: RecordRequest) -> RecordReceipt:
        return self.graph.record(request)

    def status(self, request: StatusRequest) -> StatusReport:
        agg = self.pots.aggregate_status(pot_id=request.pot_id)
        active = agg.active_pot
        pot_id = request.pot_id or (active.pot_id if active else "")
        data_plane = self.graph.data_plane_status(pot_id) if pot_id else None
        nudge = self.skills.nudge(agent=request.harness) if request.harness else None
        backend_ready = bool(data_plane and data_plane.backend_ready)
        return StatusReport(
            pot_id=pot_id,
            profile=self.profile,
            daemon_up=True,  # in-process host; real daemon liveness is host.daemon
            active_pot=active.name if active else None,
            backend_ready=backend_ready,
            data_plane=_data_plane_dict(data_plane),
            pot_summary={
                "pot_count": agg.pot_count,
                "sources": [s.name for s in agg.sources],
            },
            skills=nudge,
            recommended_next_action=_next_action(active is not None, backend_ready),
            metadata={"intent": normalize_context_intent(request.intent)},
        )


def _data_plane_dict(dp) -> dict:
    if dp is None:
        return {}
    return {
        "backend_profile": dp.backend_profile,
        "backend_ready": dp.backend_ready,
        "reader_backed_includes": list(dp.reader_backed_includes),
        "counts": dict(dp.counts),
        "freshness": dict(dp.freshness),
        "quality": dict(dp.quality),
    }


def _next_action(has_pot: bool, backend_ready: bool) -> str:
    if not has_pot:
        return "Run 'potpie setup' to create and activate a pot."
    if not backend_ready:
        return "Backend not ready — run 'potpie backend doctor'."
    return "Run 'potpie resolve \"<task>\"' to pull context for your work."


__all__ = ["AgentContextService"]
