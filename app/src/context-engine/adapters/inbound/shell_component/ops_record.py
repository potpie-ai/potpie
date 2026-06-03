"""context.record operation — delegates to HostShell.agent_context.record."""
from __future__ import annotations
import asyncio
from typing import Any
from pydantic import BaseModel, Field
from domain.ports.daemon.operations import OperationSpec, OperationContext, AuthRequirement
from domain.ports.agent_context import RecordRequest


class ContextRecordInput(BaseModel):
    type: str = Field(..., description="Record type (fix, preference, decision, ...).")
    summary: str = Field(..., description="One-line durable memory.")
    details: dict[str, Any] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None
    pot_id: str | None = None


class RecordResult(BaseModel):
    record_id: str | None = None
    accepted: bool = True
    status: str = "recorded"
    mutations_applied: int = 0
    detail: str | None = None


def build_op(component) -> OperationSpec:
    async def handler(inp: ContextRecordInput, ctx: OperationContext) -> RecordResult:
        pot_id = component.resolve_pot_id(inp.pot_id)
        req = RecordRequest(
            pot_id=pot_id,
            record_type=inp.type,
            summary=inp.summary,
            details=inp.details,
            scope=inp.scope,
            source_refs=tuple(inp.source_refs),
            idempotency_key=inp.idempotency_key,
        )
        receipt = await asyncio.to_thread(component.agent_context.record, req)
        return RecordResult(
            record_id=receipt.record_id,
            accepted=receipt.accepted,
            status=receipt.status,
            mutations_applied=receipt.mutations_applied,
            detail=receipt.detail,
        )

    return OperationSpec(
        name="context.record", input_model=ContextRecordInput, output_model=RecordResult,
        handler=handler, summary="Durable project-memory write.",
        mutating=True, auth=AuthRequirement.REQUIRED,
    )
