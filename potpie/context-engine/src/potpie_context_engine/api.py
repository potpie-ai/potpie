"""Stable consumer-facing API surface.

Consumers composing a graph runtime import backend and persistence ports from
here instead of from internal layer paths, so internal moves do not break
them:

    from potpie_context_engine.api import (
        GraphBackend,
        GraphInboxStorePort,
        GraphPlanStorePort,
        GraphService,
    )

This module must import only stdlib-light domain contracts — no delivery
adapters, no backend drivers.
"""

from __future__ import annotations

from potpie_context_core.domain.ports.graph.backend import GraphBackend
from potpie_context_core.domain.ports.graph.inbox_store import GraphInboxStorePort
from potpie_context_core.domain.ports.graph.plan_store import GraphPlanStorePort
from potpie_context_core.domain.ports.services.graph_service import GraphService

__all__ = [
    "GraphBackend",
    "GraphInboxStorePort",
    "GraphPlanStorePort",
    "GraphService",
]
