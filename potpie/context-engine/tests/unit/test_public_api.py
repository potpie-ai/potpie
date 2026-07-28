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
from pathlib import Path
import subprocess
import sys
import tomllib

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
        f"importing the public API surface loaded optional dependencies: {offending}"
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
    assert {
        "GraphBackend",
        "GraphInboxStorePort",
        "GraphPlanStorePort",
        "GraphService",
    } <= set(api.__all__)
    assert {
        "DEFAULT_GRAPH_DEFINITION",
        "GraphDefinition",
        "GraphExtension",
        "GraphRuntime",
        "build_graph_runtime",
    } <= set(api.__all__)


def test_package_root_exports_the_documented_runtime_surface() -> None:
    import potpie_context_engine
    from potpie_context_core.definition import (
        DEFAULT_GRAPH_DEFINITION,
        GraphDefinition,
        GraphExtension,
    )
    from potpie_context_core.runtime import GraphRuntime, build_graph_runtime

    assert potpie_context_engine.DEFAULT_GRAPH_DEFINITION is DEFAULT_GRAPH_DEFINITION
    assert potpie_context_engine.GraphDefinition is GraphDefinition
    assert potpie_context_engine.GraphExtension is GraphExtension
    assert potpie_context_engine.GraphRuntime is GraphRuntime
    assert potpie_context_engine.build_graph_runtime is build_graph_runtime
    assert set(potpie_context_engine.__all__) == {
        "DEFAULT_GRAPH_DEFINITION",
        "GraphDefinition",
        "GraphExtension",
        "GraphRuntime",
        "build_graph_runtime",
    }


def test_public_distributions_package_pep561_markers() -> None:
    engine_root = Path(__file__).resolve().parents[2]
    projects = (
        (engine_root, "potpie_context_engine"),
        (engine_root.parent / "context-core", "potpie_context_core"),
    )

    for project_root, package in projects:
        marker = project_root / "src" / package / "py.typed"
        config = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        wheel_include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["include"]

        assert marker.is_file()
        assert f"src/{package}/py.typed" in wheel_include
        assert "Typing :: Typed" in config["project"]["classifiers"]


def test_legacy_top_level_packages_are_gone() -> None:
    import importlib.util

    for legacy in ("domain", "application", "adapters", "bootstrap", "host"):
        spec = importlib.util.find_spec(legacy)
        assert spec is None, (
            f"legacy top-level package {legacy!r} is still importable; the "
            "wheel must ship only the potpie_context_engine namespace"
        )
