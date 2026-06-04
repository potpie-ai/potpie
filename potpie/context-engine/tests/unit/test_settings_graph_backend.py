"""Unit tests for GRAPH_DB_BACKEND + FalkorDB settings accessors.

Covers default (neo4j), backend selection, and ``CONTEXT_ENGINE_*`` → bare-env
fallback precedence on ``EnvContextEngineSettings``.
"""

from __future__ import annotations

import pytest

from adapters.outbound.settings_env import EnvContextEngineSettings

pytestmark = pytest.mark.unit


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GRAPH_DB_BACKEND",
        "CONTEXT_ENGINE_FALKORDB_URL",
        "FALKORDB_URL",
        "CONTEXT_ENGINE_FALKORDB_GRAPH_NAME",
        "FALKORDB_GRAPH_NAME",
        "CONTEXT_ENGINE_FALKORDB_MODE",
        "FALKORDB_MODE",
        "CONTEXT_ENGINE_FALKORDB_LITE_PATH",
        "FALKORDB_LITE_PATH",
    ):
        monkeypatch.delenv(var, raising=False)


def test_backend_defaults_to_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert EnvContextEngineSettings().graph_db_backend() == "neo4j"


def test_backend_falkordb_lowercased(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("GRAPH_DB_BACKEND", "FalkorDB")
    assert EnvContextEngineSettings().graph_db_backend() == "falkordb"


def test_falkordb_url_none_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert EnvContextEngineSettings().falkordb_url() is None


def test_falkordb_url_context_engine_overrides_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("FALKORDB_URL", "redis://bare:6379")
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_URL", "redis://dedicated:6379")
    assert EnvContextEngineSettings().falkordb_url() == "redis://dedicated:6379"


def test_falkordb_url_falls_back_to_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("FALKORDB_URL", "redis://bare:6379")
    assert EnvContextEngineSettings().falkordb_url() == "redis://bare:6379"


def test_graph_name_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert EnvContextEngineSettings().falkordb_graph_name() == "context_graph"
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_GRAPH_NAME", "my_graph")
    assert EnvContextEngineSettings().falkordb_graph_name() == "my_graph"


def test_mode_default_lite_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert EnvContextEngineSettings().falkordb_mode() == "lite"
    monkeypatch.setenv("FALKORDB_MODE", "server")
    assert EnvContextEngineSettings().falkordb_mode() == "server"


def test_lite_path_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert EnvContextEngineSettings().falkordb_lite_path() == ".potpie/context_graph/falkordb.db"
    monkeypatch.setenv("FALKORDB_LITE_PATH", "/tmp/cg.db")
    assert EnvContextEngineSettings().falkordb_lite_path() == "/tmp/cg.db"
