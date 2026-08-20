"""Stable, closed consumer API for the default context-engine implementation."""

from __future__ import annotations

from potpie_context_engine.core.api import *  # noqa: F403
from potpie_context_engine.core.api import __all__ as _CORE_API
from potpie_context_engine.application.readers._common import ReadRequest, ReadResponse
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.composition import build_graph_service
from potpie_context_engine.context_engine import (
    ContextEngine,
    ContextIdentity,
    ContextOperations,
    EngineConfig,
    EngineDependencies,
    EngineResource,
    GraphOperations,
    IngestionOperations,
    NudgeOperations,
    ResourceOwnership,
    WorkbenchOperations,
    create_engine,
)
from potpie_context_engine.domain.ranking import (
    Candidate,
    RankedItem,
    RankingService,
    TaskContext,
)
from potpie_context_engine.outcomes import *  # noqa: F403
from potpie_context_engine.outcomes import __all__ as _OUTCOME_API
from potpie_context_engine.requests import *  # noqa: F403
from potpie_context_engine.requests import __all__ as _REQUEST_API

__all__ = [
    *_CORE_API,
    *_OUTCOME_API,
    *_REQUEST_API,
    "ContextEngine",
    "ContextIdentity",
    "ContextOperations",
    "DefaultGraphService",
    "EngineConfig",
    "EngineDependencies",
    "EngineResource",
    "GraphOperations",
    "IngestionOperations",
    "NudgeOperations",
    "ResourceOwnership",
    "WorkbenchOperations",
    "create_engine",
    "Candidate",
    "RankedItem",
    "RankingService",
    "ReadRequest",
    "ReadResponse",
    "TaskContext",
    "build_graph_service",
]
