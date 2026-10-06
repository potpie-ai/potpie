"""Stable, closed consumer API for the default context-engine implementation."""

from __future__ import annotations

from potpie_context_engine.core.api import *  # noqa: F403
from potpie_context_engine.core.api import __all__ as _CORE_API
from potpie_context_engine.application.readers._common import ReadResponse
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.resource_facade import ResourceFacade
from potpie_context_engine.composition import build_graph_service

# Supported composition surface for embedding hosts that bring their own
# backend and plan/inbox stores (see ``build_graph_runtime``).
from potpie_context_engine.core.runtime import GraphRuntime, build_graph_runtime
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
    ResourceOperations,
    ResourceOwnership,
    WorkbenchOperations,
    create_engine,
)

# Opt-in protocol ontology: a definition factory to pass to the builder, plus
# the identity helpers writers use to mint its entity keys. Not an extension
# registration hook; ``GraphExtension`` stays internal.
from potpie_context_engine.core.protocols import protocol_entity, protocol_entity_key
from potpie_context_engine.protocols import protocols_definition
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
from potpie_context_engine.requests import ReadRequest as ReadRequest
from potpie_context_engine.results import *  # noqa: F403
from potpie_context_engine.results import __all__ as _RESULT_API

__all__ = [
    *_CORE_API,
    *_OUTCOME_API,
    *_REQUEST_API,
    *_RESULT_API,
    "ContextEngine",
    "ContextIdentity",
    "ContextOperations",
    "DefaultGraphService",
    "GraphRuntime",
    "EngineConfig",
    "EngineDependencies",
    "EngineResource",
    "GraphOperations",
    "IngestionOperations",
    "NudgeOperations",
    "ResourceFacade",
    "ResourceOperations",
    "ResourceOwnership",
    "WorkbenchOperations",
    "create_engine",
    "Candidate",
    "RankedItem",
    "RankingService",
    "ReadRequest",
    "ReadResponse",
    "TaskContext",
    "build_graph_runtime",
    "build_graph_service",
    "protocol_entity",
    "protocol_entity_key",
    "protocols_definition",
]
