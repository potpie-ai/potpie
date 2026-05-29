"""build_container selects the graph backend from GRAPH_DB_BACKEND.

Construction is lazy (adapters don't connect on init), so this exercises the
real selection branch without a live Neo4j/FalkorDB.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adapters.outbound.graph.falkordb_reader import FalkorDBClaimQueryStore
from adapters.outbound.graph.falkordb_writer import FalkorDBGraphWriter
from adapters.outbound.graph.neo4j_reader import Neo4jClaimQueryStore
from adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter
from adapters.outbound.settings_env import EnvContextEngineSettings
from bootstrap.container import build_container

pytestmark = pytest.mark.unit


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GRAPH_DB_BACKEND",
        "CONTEXT_ENGINE_FALKORDB_URL",
        "FALKORDB_URL",
        "FALKORDB_MODE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_to_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    c = build_container(settings=EnvContextEngineSettings(), pots=MagicMock())
    assert isinstance(c.graph_writer, Neo4jGraphWriter)
    assert isinstance(c.context_graph._orchestrator.claim_query, Neo4jClaimQueryStore)


def test_selects_falkordb(monkeypatch: pytest.MonkeyPatch) -> None:
    # Lite is the default mode, so backend selection needs no URL; the shared
    # graph provider is lazy, so nothing connects at wiring time.
    _clear(monkeypatch)
    monkeypatch.setenv("GRAPH_DB_BACKEND", "falkordb")
    c = build_container(settings=EnvContextEngineSettings(), pots=MagicMock())
    assert isinstance(c.graph_writer, FalkorDBGraphWriter)
    assert isinstance(c.context_graph._orchestrator.claim_query, FalkorDBClaimQueryStore)
