"""FalkorDB backend selection on the monolith's PotpieContextEngineSettings.

This is the settings class an actual local/self-hosted user runs (via
``build_container_for_session`` in ``app.modules.context_graph.wiring``), so the
``GRAPH_DB_BACKEND`` / ``FALKORDB_*`` accessors must honor the same
``CONTEXT_ENGINE_*`` → bare-env → default precedence as the context-engine
``EnvContextEngineSettings``. The four accessors read only env vars (not the
config provider), so a stub ``cp`` is enough.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.modules.context_graph.wiring import PotpieContextEngineSettings

pytestmark = pytest.mark.unit


def _settings() -> PotpieContextEngineSettings:
    # The FalkorDB accessors are env-driven and never touch the config provider.
    return PotpieContextEngineSettings(cp=SimpleNamespace())


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
    assert _settings().graph_db_backend() == "neo4j"


def test_backend_falkordb_lowercased(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("GRAPH_DB_BACKEND", "FalkorDB")
    assert _settings().graph_db_backend() == "falkordb"


def test_url_none_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert _settings().falkordb_url() is None


def test_url_context_engine_overrides_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("FALKORDB_URL", "redis://bare:6379")
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_URL", "redis://dedicated:6379")
    assert _settings().falkordb_url() == "redis://dedicated:6379"


def test_url_falls_back_to_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("FALKORDB_URL", "redis://bare:6379")
    assert _settings().falkordb_url() == "redis://bare:6379"


def test_graph_name_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert _settings().falkordb_graph_name() == "context_graph"
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_GRAPH_NAME", "my_graph")
    assert _settings().falkordb_graph_name() == "my_graph"


def test_mode_default_lite_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert _settings().falkordb_mode() == "lite"
    monkeypatch.setenv("FALKORDB_MODE", "server")
    assert _settings().falkordb_mode() == "server"


def test_lite_path_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert _settings().falkordb_lite_path() == ".potpie/context_graph/falkordb.db"
    monkeypatch.setenv("FALKORDB_LITE_PATH", "/tmp/cg.db")
    assert _settings().falkordb_lite_path() == "/tmp/cg.db"
