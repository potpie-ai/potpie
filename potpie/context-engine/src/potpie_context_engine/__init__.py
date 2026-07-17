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

__all__: list[str] = []
