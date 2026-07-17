"""Packaging gate: the core is a pydantic-only library.

Importing ``potpie_context_core`` (and its heaviest entry points) must load
nothing beyond the standard library and pydantic — no delivery surfaces, no
backend drivers, no engine modules. Runs in a subprocess so modules imported
by other tests cannot mask a regression.
"""

from __future__ import annotations

import json
import subprocess
import sys

_FORBIDDEN_TOP_LEVEL = (
    "potpie",
    "potpie_context_engine",
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
    "httpx",
    "aiohttp",
)

_IMPORT_SNIPPET = """
import json
import sys

import potpie_context_core
from potpie_context_core.application.services.graph_workbench import (
    GraphWorkbenchService,
)
from potpie_context_core.domain.ontology import PUBLIC_RECORD_TYPES
from potpie_context_core.domain.ports.graph.backend import GraphBackend

print(json.dumps(sorted({module.split(".")[0] for module in sys.modules})))
"""


def test_core_imports_load_only_stdlib_and_pydantic() -> None:
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
        f"importing potpie_context_core loaded forbidden modules: {offending}"
    )
