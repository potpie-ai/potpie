"""Stable, closed consumer API for the default context-engine implementation."""

from __future__ import annotations

from potpie_context_core.api import *  # noqa: F403
from potpie_context_core.api import __all__ as _CORE_API
from potpie_context_engine.application.readers._common import ReadRequest, ReadResponse
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.composition import build_graph_service
from potpie_context_engine.domain.ranking import (
    Candidate,
    RankedItem,
    RankingService,
    TaskContext,
)

__all__ = [
    *_CORE_API,
    "DefaultGraphService",
    "Candidate",
    "RankedItem",
    "RankingService",
    "ReadRequest",
    "ReadResponse",
    "TaskContext",
    "build_graph_service",
]
