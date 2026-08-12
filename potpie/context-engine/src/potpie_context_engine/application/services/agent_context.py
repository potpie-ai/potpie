"""``AgentContextPort`` implementation — composes the three services.

This is the public four-tool surface. ``resolve``/``search``/``record`` delegate
straight to the ``GraphService`` data plane; ``status`` is the only composite —
it joins ``GraphService`` data-plane status, ``PotManagementService`` control-
plane status, and a ``SkillManager`` nudge into one ``StatusReport``.

CLI and HTTP adapters bind here. The managed HTTP ingestion surface is a legacy adapter
while it migrates onto the host shell; it must not define new agent tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from potpie_context_core.agent_context_port import normalize_context_intent
from potpie_context_core.agent_envelope import AgentEnvelope
from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    StatusReport,
    StatusRequest,
)
from potpie_context_core.ports.graph_service import GraphService
from potpie_context_engine.domain.ports.services.pot_management import (
    PotManagementService,
)
from potpie_context_engine.domain.ports.services.skill_manager import SkillManager

_QUALITY_SUMMARY_LIMIT = 20
_LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass(slots=True)
class AgentContextService:
    """The 4-tool agent contract, composed over the three services."""

    graph: GraphService
    pots: PotManagementService
    skills: SkillManager
    profile: str = "local"
    workbench: Any = None
    """Optional ``GraphWorkbenchService`` used to report real graph quality.

    The backend's analytics projection only knows counts, so ``status`` used to
    report a healthy graph no matter how many findings ``graph quality`` had
    open. When the workbench is wired, status asks it the same summary question
    the quality command asks."""

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
        quality = _quality_block(data_plane, workbench=self.workbench, pot_id=pot_id)
        return StatusReport(
            pot_id=pot_id,
            profile=self.profile,
            daemon_up=True,  # in-process host; real daemon liveness is host.daemon
            active_pot=active.name if active else None,
            backend_ready=backend_ready,
            data_plane=_data_plane_dict(data_plane, quality=quality),
            pot_summary={
                "pot_count": agg.pot_count,
                "sources": [s.name for s in agg.sources],
            },
            skills=nudge,
            recommended_next_action=_next_action(
                active is not None, backend_ready, quality
            ),
            metadata={"intent": normalize_context_intent(request.intent)},
        )


def _data_plane_dict(dp, *, quality: Mapping[str, Any] | None = None) -> dict:
    if dp is None:
        return {}
    return {
        "backend_profile": dp.backend_profile,
        "backend_ready": dp.backend_ready,
        "reader_backed_includes": list(dp.reader_backed_includes),
        "counts": dict(dp.counts),
        "freshness": dict(dp.freshness),
        "quality": dict(quality if quality is not None else dp.quality),
    }


def _quality_block(dp, *, workbench: Any, pot_id: str) -> dict[str, Any]:
    """Join the backend's quality projection with the open quality findings."""
    base: dict[str, Any] = dict(dp.quality) if dp is not None else {}
    report = getattr(workbench, "quality", None) if workbench is not None else None
    if report is None or not pot_id:
        return base
    try:
        result = report(
            pot_id=pot_id,
            report="summary",
            subgraph=None,
            limit=_QUALITY_SUMMARY_LIMIT,
            confidence_threshold=_LOW_CONFIDENCE_THRESHOLD,
        )
    except CapabilityNotImplemented as exc:
        return {**base, "findings_status": "unavailable", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001 - status must survive a bad probe
        return {
            **base,
            "findings_status": "unavailable",
            "detail": f"quality summary unavailable: {exc}",
        }
    body = result.to_dict()
    metrics = dict(body.get("metrics") or {})
    return {
        **base,
        "status": body.get("status") or base.get("status"),
        "findings_status": body.get("status"),
        "source": "quality_summary",
        "open_findings": int(
            metrics.get("total_findings") or body.get("finding_count") or 0
        ),
        "quality_counts": dict(metrics.get("quality_counts") or {}),
    }


def _next_action(
    has_pot: bool,
    backend_ready: bool,
    quality: Mapping[str, Any] | None = None,
) -> str:
    if not has_pot:
        return "Run 'potpie setup' to create and activate a pot."
    if not backend_ready:
        return "Backend not ready — run 'potpie backend doctor'."
    if quality and int(quality.get("open_findings") or 0) > 0:
        return (
            f"{quality['open_findings']} open graph quality finding(s) — run "
            "'potpie graph quality summary --json'."
        )
    return "Run 'potpie resolve \"<task>\"' to pull context for your work."


__all__ = ["AgentContextService"]
