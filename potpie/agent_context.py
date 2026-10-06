"""Potpie agent-context composition over graph, pot, and skill services.

This is the public four-tool surface. ``resolve``/``search``/``record`` delegate
to ``GraphService``; ``status`` is the only composite — it joins graph status,
pot/source status, open graph-quality findings and a ``SkillManager`` nudge into
one ``StatusReport``.

The agent-facing door is also where an unset intent is filled: ``resolve``
infers it from the task text and ``search`` asks for the definition intent when
the query names an acronym. The graph service's internal callers pick their own
includes and never see that heuristic.

The root local-runtime composition binds here. The standalone Context Engine
HTTP surface owns its own delivery composition and must not import this root
composition or define new agent tools through it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from potpie_context_engine.core.agent_context_port import (
    infer_context_intent,
    normalize_context_intent,
    normalize_context_values,
)
from potpie_context_engine.core.agent_envelope import AgentEnvelope
from potpie_context_engine.core.definition_query import definition_subject
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    StatusReport,
    StatusRequest,
)
from potpie_context_engine.core.ports.graph_service import GraphService
from potpie.pots.contracts import (
    PotManagementService,
)
from potpie.skills.contracts import SkillManager

_RequestT = TypeVar("_RequestT", ResolveRequest, SearchRequest)

QUALITY_SUMMARY_LIMIT = 20
LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass(slots=True)
class AgentContextService:
    """The 4-tool agent contract, composed over the three services."""

    graph: GraphService
    pots: PotManagementService
    skills: SkillManager
    profile: str = "local"
    workbench: Any = None
    """Optional graph workbench used to report open graph-quality findings.

    The backend's analytics projection only knows counts, so without it
    ``status`` reports a healthy graph however many findings ``graph quality``
    has open. When it is wired, status asks the same summary question the
    quality command asks."""

    def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        return self.graph.resolve(
            _with_document_includes(_with_effective_intent(request))
        )

    def search(self, request: SearchRequest) -> AgentEnvelope:
        explicit = (request.intent or "").strip()
        intent = explicit or (
            "definition" if definition_subject(request.query) else "unknown"
        )
        return self.graph.search(
            _with_document_includes(
                _with_intent(request, intent, explicit=bool(explicit))
            )
        )

    def record(self, request: RecordRequest) -> RecordReceipt:
        return self.graph.record(request)

    def status(self, request: StatusRequest) -> StatusReport:
        agg = self.pots.aggregate_status(pot_id=request.pot_id)
        active = agg.active_pot
        pot_id = request.pot_id or (active.pot_id if active else "")
        data_plane = self.graph.data_plane_status(pot_id) if pot_id else None
        nudge = self.skills.nudge(agent=request.harness) if request.harness else None
        backend_ready = bool(data_plane and data_plane.backend_ready)
        quality = quality_block(
            dict(data_plane.quality) if data_plane is not None else {},
            summary=_workbench_quality_summary(self.workbench, pot_id=pot_id),
        )
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
            recommended_next_action=status_next_action(
                has_pot=active is not None,
                backend_ready=backend_ready,
                quality=quality,
            ),
            metadata={"intent": normalize_context_intent(request.intent)},
        )


def _with_document_includes(request: _RequestT) -> _RequestT:
    include = _document_includes(request.include)
    if include == tuple(request.include):
        return request
    return dataclasses.replace(request, include=include)


def _document_includes(include: tuple[str, ...]) -> tuple[str, ...]:
    """The agent-facing ``docs`` filter searches section summaries *and* text.

    ``docs`` alone answers from agent-written section summaries, so a phrase
    that appears in a document but in no summary was unreachable through it.
    Adding ``resources`` reaches the chunk text itself. Named graph views keep
    their precise summary/passage split, and an explicit ``resources``-only
    request stays text-only.
    """
    normalized = tuple(normalize_context_values(include))
    if "docs" in normalized and "resources" not in normalized:
        return (*normalized, "resources")
    return tuple(include)


def _with_effective_intent(request: ResolveRequest) -> ResolveRequest:
    """Fill an unset intent from the task text, and say which happened.

    An explicit intent passes through untouched; an unset or blank one is
    inferred from the task (``infer_context_intent``). Either way the envelope
    metadata carries ``intent_source``, so an agent that sees ``inferred`` next
    to a surprising intent knows to name one.

    This lives on the agent-facing door rather than in ``GraphService.resolve``
    on purpose: internal callers of the graph service (graph reads, the
    reconciliation tools) pick their own includes and must not have their
    family set moved by a heuristic over the query text.
    """
    explicit = (request.intent or "").strip()
    intent = explicit or infer_context_intent(request.task)
    return _with_intent(request, intent, explicit=bool(explicit))


def _with_intent(request, intent: str, *, explicit: bool):
    return dataclasses.replace(
        request,
        intent=intent,
        metadata={
            **dict(request.metadata),
            "intent_source": "explicit" if explicit else "inferred",
        },
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


@dataclass(frozen=True, slots=True)
class QualitySummaryUnavailable:
    """The quality summary could not be read; ``detail`` says why."""

    detail: str


def _workbench_quality_summary(workbench: Any, *, pot_id: str):
    """Ask the workbench for its quality summary, or report why it could not."""
    report = getattr(workbench, "quality", None) if workbench is not None else None
    if report is None or not pot_id:
        return None
    try:
        return report(
            pot_id=pot_id,
            report="summary",
            subgraph=None,
            limit=QUALITY_SUMMARY_LIMIT,
            confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        )
    except CapabilityNotImplemented as exc:
        return QualitySummaryUnavailable(detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - status must survive a bad probe
        return QualitySummaryUnavailable(detail=f"quality summary unavailable: {exc}")


def quality_block(base: Mapping[str, Any], *, summary: Any = None) -> dict[str, Any]:
    """Join the backend's quality projection with the open quality findings.

    ``summary`` is a graph-quality summary result (anything with ``to_dict``),
    a :class:`QualitySummaryUnavailable`, or ``None`` when no summary was asked
    for. Shared by this service and the CLI's ``status`` so both report the
    same block.
    """
    block: dict[str, Any] = dict(base)
    if summary is None:
        return block
    if isinstance(summary, QualitySummaryUnavailable):
        return {**block, "findings_status": "unavailable", "detail": summary.detail}
    body = summary.to_dict()
    metrics = dict(body.get("metrics") or {})
    return {
        **block,
        "status": body.get("status") or block.get("status"),
        "findings_status": body.get("status"),
        "source": "quality_summary",
        "open_findings": int(
            metrics.get("total_findings") or body.get("finding_count") or 0
        ),
        "quality_counts": dict(metrics.get("quality_counts") or {}),
    }


def status_next_action(
    *,
    has_pot: bool,
    backend_ready: bool,
    quality: Mapping[str, Any] | None = None,
) -> str:
    """The one next step ``status`` recommends, naming open findings first."""
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


__all__ = [
    "AgentContextService",
    "QualitySummaryUnavailable",
    "quality_block",
    "status_next_action",
]
