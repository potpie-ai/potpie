"""Potpie Context Engine — the context-bound project-memory library.

Supported imports live in two places:

- ``potpie_context_engine`` (this module) — the context-bound engine factory,
  lifecycle, outcomes, and non-extensible default graph definition.
- ``potpie_context_engine.api`` — stable contract DTOs and ports for
  consumers composing their own runtime (``GraphBackend``,
  ``GraphPlanStorePort``, ``GraphInboxStorePort``, ``GraphService``).

Everything under ``potpie_context_engine.domain`` / ``.application`` /
``.adapters`` / ``.bootstrap`` / ``.host`` / ``.benchmarks`` is internal and
may change without notice.

Importing this package must stay dependency-light: no delivery-surface or
backend third-party imports (FastAPI, Typer, MCP, FalkorDB, Neo4j,
SQLAlchemy, Hatchet, OpenTelemetry, Sentry) at module import time.
"""

from __future__ import annotations

from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.context_engine import (
    ContextEngine,
    ContextIdentity,
    EngineConfig,
    EngineDependencies,
    EngineResource,
    create_engine,
)
from potpie_context_engine.outcomes import (
    DependencyError,
    DomainError,
    EngineError,
    EngineLifecycleError,
    Failure,
    Outcome,
    Success,
)

__all__ = [
    "DEFAULT_GRAPH_DEFINITION",
    "ContextEngine",
    "ContextIdentity",
    "DependencyError",
    "DomainError",
    "EngineConfig",
    "EngineDependencies",
    "EngineError",
    "EngineLifecycleError",
    "EngineResource",
    "Failure",
    "GraphDefinition",
    "Outcome",
    "Success",
    "create_engine",
]
