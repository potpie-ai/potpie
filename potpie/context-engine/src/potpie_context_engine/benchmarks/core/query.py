"""Benchmark reads through an explicitly supplied canonical graph service."""

from __future__ import annotations

from typing import Any

from potpie_context_engine.application.services.envelope_builder import envelope_to_dict
from potpie_context_engine.core.ports.agent_context import ResolveRequest
from potpie_context_engine.core.ports.graph_service import GraphService
from potpie_context_engine.benchmarks.core.scenario import QuerySpec


def resolve_context(
    service: GraphService, pot_id: str, query: QuerySpec
) -> dict[str, Any]:
    """Return an evidence envelope in the benchmark report's response field."""
    envelope = service.resolve(
        ResolveRequest(
            pot_id=pot_id,
            intent=query.intent,
            include=tuple(query.include),
            mode=query.mode,
            source_policy=query.source_policy,
            scope=dict(query.scope),
        )
    )
    return {"result": envelope_to_dict(envelope)}
