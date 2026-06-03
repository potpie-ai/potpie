"""context.resolve operation — delegates to HostShell.agent_context.resolve."""
from __future__ import annotations
import asyncio
from typing import Any
from pydantic import BaseModel, Field
from domain.ports.daemon.operations import OperationSpec, OperationContext, AuthRequirement
from domain.ports.agent_context import ResolveRequest


class ResolveInput(BaseModel):
    task: str = Field(..., description="Task description / agent prompt.")
    intent: str = Field("feature", description="Task shape (feature/debugging/review/...).")
    include: list[str] = Field(default_factory=list, description="Evidence families to retrieve.")
    scope: dict[str, Any] = Field(default_factory=dict)
    mode: str = "fast"
    source_policy: str = "references_only"
    pot_id: str | None = None


class WireEvidenceItem(BaseModel):
    include: str
    score: float = 0.0
    payload: dict = Field(default_factory=dict)


class WireCoverage(BaseModel):
    include: str
    status: str


class WireUnsupported(BaseModel):
    name: str
    reason: str


class AgentEnvelope(BaseModel):
    """Wire shape of the canonical agent envelope (mirrors domain.agent_envelope.AgentEnvelope)."""
    pot_id: str | None = None
    intent: str | None = None
    overall_confidence: str = "unknown"
    items: list[WireEvidenceItem] = Field(default_factory=list)
    coverage: list[WireCoverage] = Field(default_factory=list)
    unsupported_includes: list[WireUnsupported] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


def to_wire_envelope(env) -> AgentEnvelope:
    """Translate a domain ``AgentEnvelope`` into the wire shape."""
    return AgentEnvelope(
        pot_id=env.pot_id,
        intent=env.intent,
        overall_confidence=env.overall_confidence,
        items=[
            WireEvidenceItem(include=i.include, score=i.score, payload=dict(i.payload))
            for i in env.items
        ],
        coverage=[WireCoverage(include=c.include, status=c.status) for c in env.coverage],
        unsupported_includes=[
            WireUnsupported(name=u.name, reason=u.reason) for u in env.unsupported_includes
        ],
        metadata=dict(env.metadata),
    )


def build_op(component) -> OperationSpec:
    async def handler(inp: ResolveInput, ctx: OperationContext) -> AgentEnvelope:
        pot_id = component.resolve_pot_id(inp.pot_id)
        req = ResolveRequest(
            pot_id=pot_id,
            task=inp.task,
            intent=inp.intent or None,
            include=tuple(inp.include),
            scope=inp.scope,
            mode=inp.mode,
            source_policy=inp.source_policy,
        )
        env = await asyncio.to_thread(component.agent_context.resolve, req)
        return to_wire_envelope(env)

    return OperationSpec(
        name="context.resolve", input_model=ResolveInput, output_model=AgentEnvelope,
        handler=handler, summary="Resolve bounded task context.",
        mutating=False, auth=AuthRequirement.REQUIRED,
    )
