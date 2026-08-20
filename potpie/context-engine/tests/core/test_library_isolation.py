"""Packaging gate: the embedded engine core stays dependency-light.

Importing ``potpie_context_engine.core`` (and its heaviest entry points) must load
nothing beyond the engine package, standard library, and pydantic — no delivery
surfaces or backend drivers. Runs in a subprocess so modules imported
by other tests cannot mask a regression.
"""

from __future__ import annotations

import json
import subprocess
import sys

_FORBIDDEN_TOP_LEVEL = (
    "potpie",
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

import potpie_context_engine.core
from potpie_context_engine.core import api

assert api.__all__

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
        f"importing potpie_context_engine.core loaded forbidden modules: {offending}"
    )
