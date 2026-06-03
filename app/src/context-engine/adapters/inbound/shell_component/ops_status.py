"""context.status operation — delegates to HostShell.agent_context.status."""
from __future__ import annotations
import asyncio
from typing import Any
from pydantic import BaseModel, Field
from domain.ports.daemon.operations import OperationSpec, OperationContext, AuthRequirement
from domain.ports.agent_context import StatusRequest


class ContextStatusInput(BaseModel):
    intent: str | None = None
    harness: str | None = None
    pot_id: str | None = None


class SkillNudge(BaseModel):
    agent: str | None = None
    missing: list[str] = Field(default_factory=list)
    outdated: list[str] = Field(default_factory=list)
    install_command: str | None = None


class StatusReport(BaseModel):
    pot_id: str | None = None
    profile: str | None = None
    daemon_up: bool = True
    active_pot: str | None = None
    backend_ready: bool = False
    data_plane: dict[str, Any] = Field(default_factory=dict)
    pot_summary: dict[str, Any] = Field(default_factory=dict)
    skills: SkillNudge | None = None
    recommended_next_action: str | None = None


def build_op(component) -> OperationSpec:
    async def handler(inp: ContextStatusInput, ctx: OperationContext) -> StatusReport:
        pot_id = component.resolve_pot_id(inp.pot_id)
        req = StatusRequest(pot_id=pot_id, intent=inp.intent, harness=inp.harness)
        report = await asyncio.to_thread(component.agent_context.status, req)
        skills = None
        if report.skills is not None:
            skills = SkillNudge(
                agent=report.skills.agent,
                missing=list(report.skills.missing),
                outdated=list(report.skills.outdated),
                install_command=report.skills.install_command,
            )
        return StatusReport(
            pot_id=report.pot_id,
            profile=report.profile,
            daemon_up=report.daemon_up,
            active_pot=report.active_pot,
            backend_ready=report.backend_ready,
            data_plane=dict(report.data_plane),
            pot_summary=dict(report.pot_summary),
            skills=skills,
            recommended_next_action=report.recommended_next_action,
        )

    return OperationSpec(
        name="context.status", input_model=ContextStatusInput, output_model=StatusReport,
        handler=handler, summary="Health, readiness, freshness, and skills.",
        mutating=False, auth=AuthRequirement.NONE,
    )
