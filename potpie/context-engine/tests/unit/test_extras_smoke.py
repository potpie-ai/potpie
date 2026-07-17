"""Per-extra import smoke tests.

One test per dependency extra: skip when the extra's third-party dependency
is not installed, otherwise import a representative adapter module from that
family. In the dev environment (installed with ``[all]`` + dev groups) every
test runs, so a module-level import regression in any adapter family fails
here instead of at a consumer's first import.

``sentence-transformers`` (the ``embeddings`` extra) is deliberately checked
via ``find_spec`` instead of an import — importing it loads torch, which
would dominate the unit-suite runtime.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

_EXTRA_SMOKES = (
    (
        "cli-auth",
        "keyring",
        "potpie_context_engine.adapters.outbound.cli_auth.e2e_keyring",
    ),
    ("http", "fastapi", "potpie_context_engine.adapters.inbound.http.app"),
    ("mcp", "mcp", "potpie_context_engine.adapters.inbound.mcp.server"),
    # The falkordblite distribution installs the ``redislite`` module; gate on
    # the falkordb client, which the [local] extra also carries.
    (
        "local",
        "falkordb",
        "potpie_context_engine.adapters.outbound.graph.falkordb_writer",
    ),
    ("neo4j", "neo4j", "potpie_context_engine.adapters.outbound.graph.neo4j_writer"),
    (
        "postgres",
        "sqlalchemy",
        "potpie_context_engine.adapters.outbound.postgres.models",
    ),
    (
        "hatchet",
        "hatchet_sdk",
        "potpie_context_engine.adapters.outbound.hatchet.hatchet_job_queue",
    ),
    (
        "observability",
        "opentelemetry",
        "potpie_context_engine.adapters.outbound.observability.otel",
    ),
    (
        "github",
        "github",
        "potpie_context_engine.adapters.outbound.connectors.github.connector",
    ),
    (
        "reconciliation-agent",
        "pydantic_deep",
        "potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent",
    ),
    ("benchmarks", "yaml", "potpie_context_engine.benchmarks.core.scenario"),
)


@pytest.mark.parametrize(
    ("extra", "dependency", "module"),
    _EXTRA_SMOKES,
    ids=[extra for extra, _, _ in _EXTRA_SMOKES],
)
def test_extra_adapter_module_imports(extra: str, dependency: str, module: str) -> None:
    pytest.importorskip(dependency, reason=f"[{extra}] extra not installed")
    importlib.import_module(module)


def test_embeddings_extra_is_resolvable() -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("[embeddings] extra not installed")
    assert (
        importlib.util.find_spec(
            "potpie_context_engine.adapters.outbound.intelligence.local_embedder"
        )
        is not None
    )
