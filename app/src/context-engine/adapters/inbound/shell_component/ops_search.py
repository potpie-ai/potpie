"""context.search operation — delegates to HostShell.agent_context.search."""
from __future__ import annotations
import asyncio
from typing import Any
from pydantic import BaseModel, Field
from domain.ports.daemon.operations import OperationSpec, OperationContext, AuthRequirement
from domain.ports.agent_context import SearchRequest
from adapters.inbound.shell_component.ops_resolve import AgentEnvelope, to_wire_envelope


class SearchInput(BaseModel):
    lookup: str = Field(..., description="Targeted lookup string.")
    include: list[str] = Field(default_factory=list)
    scope: dict[str, Any] = Field(default_factory=dict)
    mode: str = "fast"
    pot_id: str | None = None


def build_op(component) -> OperationSpec:
    async def handler(inp: SearchInput, ctx: OperationContext) -> AgentEnvelope:
        pot_id = component.resolve_pot_id(inp.pot_id)
        req = SearchRequest(
            pot_id=pot_id,
            query=inp.lookup,
            include=tuple(inp.include),
            scope=inp.scope,
            mode=inp.mode,
        )
        env = await asyncio.to_thread(component.agent_context.search, req)
        return to_wire_envelope(env)

    return OperationSpec(
        name="context.search", input_model=SearchInput, output_model=AgentEnvelope,
        handler=handler, summary="Targeted context lookup.",
        mutating=False, auth=AuthRequirement.REQUIRED,
    )
