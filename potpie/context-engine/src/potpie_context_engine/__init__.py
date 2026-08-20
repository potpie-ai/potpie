"""Potpie Context Engine — the extensible project-context graph runtime.

Supported imports live in two places:

- ``potpie_context_engine`` (this module) — the graph definition/runtime
  surface. The definition and runtime-factory exports
  (``GraphDefinition``, ``GraphExtension``, ``GraphRuntime``,
  ``build_graph_runtime``, ``DEFAULT_GRAPH_DEFINITION``) land with the
  definition-injection and runtime-factory milestones of the
  modularization plan.
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
    GraphExtension,
)
from potpie_context_engine.core.runtime import GraphRuntime, build_graph_runtime
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
    "GraphExtension",
    "GraphRuntime",
    "Outcome",
    "Success",
    "build_graph_runtime",
    "create_engine",
]
