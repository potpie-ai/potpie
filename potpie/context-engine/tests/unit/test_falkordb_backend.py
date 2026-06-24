"""FalkorDB GraphBackend wiring tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adapters.outbound.graph.backends import KNOWN_PROFILES, build_backend
from adapters.outbound.graph.backends.falkordb_backend import FalkorDBGraphBackend
from bootstrap.host_wiring import build_host_shell, default_backend_profile
from bootstrap.ingestion_server import build_ingestion_server
from domain.context_events import EventRef
from domain.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceRef
from domain.lifecycle import SetupPlan
from domain.ports.graph.backend import GraphBackend
from domain.reconciliation import ReconciliationPlan

pytestmark = pytest.mark.unit


class _Settings:
    def __init__(self, *, backend: str = "falkordb", mode: str = "lite") -> None:
        self._backend = backend
        self._mode = mode

    def is_enabled(self) -> bool:
        return True

    def graph_db_backend(self) -> str:
        return self._backend

    def falkordb_url(self) -> str | None:
        return None

    def falkordb_graph_name(self) -> str:
        return "context_graph"

    def falkordb_mode(self) -> str:
        return self._mode

    def falkordb_lite_path(self) -> str:
        return ".potpie/test/falkordb.db"


class _Writer:
    enabled = True

    def __init__(self) -> None:
        self.entities = []
        self.edges = []

    async def ensure_indexes(self) -> bool:
        return True

    async def upsert_entities(self, pot_id, items, provenance: ProvenanceRef) -> int:
        self.entities.extend(items)
        return len(items)

    async def upsert_edges(self, pot_id, items, provenance: ProvenanceRef) -> int:
        self.edges.extend(items)
        return len(items)

    async def delete_edges(self, pot_id, items, provenance: ProvenanceRef) -> int:
        return len(items)

    async def invalidate(self, pot_id, items, provenance: ProvenanceRef) -> int:
        return len(items)

    async def reset_pot(self, pot_id: str) -> dict:
        return {"ok": True, "group_id_nodes_before": 0, "group_id_nodes_remaining": 0}


class _FakeResult:
    def __init__(self, names, rows):
        self.header = [[1, name] for name in names]
        self.result_set = rows


class _RepairGraph:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def query(self, cypher: str, params: dict | None = None) -> _FakeResult:
        params = params or {}
        if "RETURN e.entity_key AS key" in cypher:
            return _FakeResult(
                ["key", "props"],
                [
                    ["service:web", {"description": "Web frontend service."}],
                    ["service:auth", {"name": "auth", "summary": ""}],
                ],
            )
        if "SET e += $props" in cypher:
            self.updates.append(params)
            return _FakeResult(["cnt"], [[1]])
        return _FakeResult([], [])


def test_build_backend_registers_falkordb_without_connecting() -> None:
    assert "falkordb" in KNOWN_PROFILES
    backend = build_backend("falkordb", settings=_Settings(mode="server"))
    assert isinstance(backend, GraphBackend)
    assert backend.profile == "falkordb"
    assert backend.capabilities().implemented() == (
        "mutation",
        "claim_query",
        "semantic",
        "inspection",
        "analytics",
    )


@pytest.mark.parametrize("profile", ["falkordb_lite", "falkordblite", "falkordb-lite"])
def test_build_backend_registers_falkordb_lite_without_connecting(profile) -> None:
    assert "falkordb_lite" in KNOWN_PROFILES
    # Mode is intentionally set to server here; the explicit Lite profile pins
    # the runtime to embedded Lite and therefore needs no server URL.
    backend = build_backend(profile, settings=_Settings(mode="server"))
    assert isinstance(backend, GraphBackend)
    assert backend.profile == "falkordb_lite"
    assert backend.graph_writer.enabled is True
    assert backend.capabilities().implemented() == (
        "mutation",
        "claim_query",
        "semantic",
        "inspection",
        "analytics",
    )


async def test_falkordb_backend_apply_uses_writer() -> None:
    writer = _Writer()
    backend = FalkorDBGraphBackend(_Settings(), writer=writer)
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="agent", pot_id="p1"),
        entity_upserts=[
            EntityUpsert(entity_key="service:web", labels=("Entity", "Service"))
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DEPENDS_ON",
                from_entity_key="service:web",
                to_entity_key="service:auth",
                properties={"fact": "web depends on auth"},
            )
        ],
    )

    result = await backend.mutation.apply_async(plan, expected_pot_id="p1")

    assert result.ok
    assert len(writer.entities) == 1
    assert len(writer.edges) == 1


def test_ingestion_server_accepts_falkordb_backend() -> None:
    container = build_ingestion_server(settings=_Settings(), pots=MagicMock())

    assert container.backend is not None
    assert container.backend.profile == "falkordb"
    assert container.context_graph is not None
    assert container.graph_writer is not None


def test_ingestion_server_accepts_falkordb_lite_backend() -> None:
    container = build_ingestion_server(
        settings=_Settings(backend="falkordb_lite", mode="server"),
        pots=MagicMock(),
    )

    assert container.backend is not None
    assert container.backend.profile == "falkordb_lite"
    assert container.context_graph is not None
    assert container.graph_writer is not None


def test_host_shell_accepts_falkordb_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "falkordb")

    host = build_host_shell()

    assert host.backend.profile == "falkordb"


def test_host_shell_defaults_to_falkordb_lite(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.delenv("CONTEXT_ENGINE_BACKEND", raising=False)
    monkeypatch.delenv("GRAPH_DB_BACKEND", raising=False)

    host = build_host_shell()

    assert default_backend_profile() == "falkordb_lite"
    assert host.backend.profile == "falkordb_lite"
    assert SetupPlan().backend == "falkordb_lite"


def test_default_backend_ignores_blank_primary_env(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "   ")
    monkeypatch.setenv("GRAPH_DB_BACKEND", " embedded ")

    assert default_backend_profile() == "embedded"


def test_host_shell_accepts_falkordb_lite_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "falkordb_lite")
    monkeypatch.setenv("FALKORDB_MODE", "server")

    host = build_host_shell()

    assert host.backend.profile == "falkordb_lite"


def test_falkordb_repair_backfills_entity_summaries() -> None:
    graph = _RepairGraph()
    backend = FalkorDBGraphBackend(
        _Settings(),
        graph_provider=lambda: graph,
    )

    report = backend.analytics.repair("p1", targets=["entity_summaries"])

    assert report.repaired == {"entity_summaries": 2}
    assert [u["key"] for u in graph.updates] == ["service:web", "service:auth"]
    assert graph.updates[0]["props"]["summary"] == "Web frontend service."
    assert graph.updates[0]["props"]["description"] == "Web frontend service."
    assert graph.updates[1]["props"]["summary"] == "auth"
    assert graph.updates[1]["props"]["description"] == "auth"
