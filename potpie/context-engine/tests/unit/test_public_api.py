"""Packaging gates for the supported import surface.

Two invariants from the modularization plan:

1. The supported imports (``potpie_context_engine`` and
   ``potpie_context_engine.api``) resolve.
2. Importing them does not load any optional delivery-surface or backend
   third-party dependency — consumers on the base install must be able to
   import the graph API without FastAPI, Typer, MCP, FalkorDB, Neo4j,
   SQLAlchemy, Hatchet, OpenTelemetry, or Sentry installed.

The isolation check runs in a subprocess so modules imported by other tests
cannot mask a regression.
"""

from __future__ import annotations

import json
import subprocess
import sys

_FORBIDDEN_TOP_LEVEL = (
    "fastapi",
    "typer",
    "mcp",
    "uvicorn",
    "rich",
    "keyring",
    "PIL",
    "falkordb",
    "falkordblite",
    "redis",
    "neo4j",
    "sqlalchemy",
    "psycopg",
    "hatchet_sdk",
    "opentelemetry",
    "sentry_sdk",
    "sentence_transformers",
)

_IMPORT_SNIPPET = """
import json
import sys

import potpie_context_engine
from potpie_context_engine.api import (
    GraphBackend,
    GraphInboxStorePort,
    GraphPlanStorePort,
    GraphService,
)

print(json.dumps(sorted({module.split(".")[0] for module in sys.modules})))
"""


def test_supported_imports_do_not_load_optional_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SNIPPET],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    loaded = set(json.loads(result.stdout))
    offending = sorted(loaded & set(_FORBIDDEN_TOP_LEVEL))
    assert not offending, (
        "importing the public API surface loaded optional dependencies: "
        f"{offending}"
    )


def test_api_reexports_are_the_internal_contracts() -> None:
    from potpie_context_engine import api
    from potpie_context_core.ports.graph.backend import GraphBackend
    from potpie_context_core.ports.graph.inbox_store import (
        GraphInboxStorePort,
    )
    from potpie_context_core.ports.graph.plan_store import GraphPlanStorePort
    from potpie_context_core.ports.graph_service import GraphService

    assert api.GraphBackend is GraphBackend
    assert api.GraphInboxStorePort is GraphInboxStorePort
    assert api.GraphPlanStorePort is GraphPlanStorePort
    assert api.GraphService is GraphService
    assert set(api.__all__) == {
        "GraphBackend",
        "GraphInboxStorePort",
        "GraphPlanStorePort",
        "GraphService",
    }


def test_legacy_top_level_packages_are_gone() -> None:
    import importlib.util

    for legacy in ("domain", "application", "adapters", "bootstrap", "host"):
        spec = importlib.util.find_spec(legacy)
        assert spec is None, (
            f"legacy top-level package {legacy!r} is still importable; the "
            "wheel must ship only the potpie_context_engine namespace"
        )
