"""Unit tests for the FalkorDB GraphWriterPort adapter.

No live FalkorDB: an injected fake graph captures queries and feeds canned
results, so we exercise the ``enabled`` gate, the ``reset_pot`` contract
(client-side batched delete, same result-dict shape as Neo4j), the unnamed
best-effort index DDL, the async-shim record mapping, and that the reused
``cypher.py`` mutation path issues the expected MERGE.
"""

from __future__ import annotations

import pytest

from adapters.outbound.graph.backends.falkor.graph_handle import (
    build_falkordb_graph,
)
from adapters.outbound.graph.backends.falkor.writer import (
    FalkorDBGraphWriter,
    _records_from_result,
)
from domain.graph_mutations import EntityUpsert, ProvenanceRef

pytestmark = pytest.mark.unit


class _FakeResult:
    def __init__(self, header=None, result_set=None):
        self.header = header or []
        self.result_set = result_set or []


class _FakeGraph:
    """Captures queries; answers count() and absorbs DETACH DELETE."""

    def __init__(self, count: int = 0, *, raise_on_index: bool = False):
        self.queries: list[tuple[str, dict]] = []
        self._count = count
        self._deleted = False
        self._raise_on_index = raise_on_index

    def query(self, cypher: str, params=None):
        self.queries.append((cypher, params or {}))
        if "CREATE INDEX" in cypher and self._raise_on_index:
            raise RuntimeError("already indexed")
        if "DETACH DELETE" in cypher:
            self._deleted = True
            return _FakeResult()
        if "count(n) AS cnt" in cypher:
            val = 0 if self._deleted else self._count
            return _FakeResult(header=[[1, "cnt"]], result_set=[[val]])
        return _FakeResult()


class _FakeSettings:
    def __init__(
        self,
        *,
        enabled: bool = True,
        url: str | None = "redis://localhost:6379",
        name: str = "g",
    ):
        self._enabled = enabled
        self._url = url
        self._name = name

    def is_enabled(self) -> bool:
        return self._enabled

    def falkordb_url(self):
        return self._url

    def falkordb_graph_name(self) -> str:
        return self._name

    def falkordb_lite_path(self) -> str:
        return ".potpie/test/falkordb.db"


def test_records_from_result_maps_columns() -> None:
    res = _FakeResult(header=[[1, "key"], [1, "labels"]], result_set=[["s", ["Entity", "Service"]]])
    recs = _records_from_result(res)
    assert recs == [{"key": "s", "labels": ["Entity", "Service"]}]


def test_records_from_result_empty() -> None:
    assert _records_from_result(_FakeResult()) == []


def test_enabled_false_when_context_graph_disabled() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(enabled=False), graph=_FakeGraph())
    assert w.enabled is False


def test_enabled_false_when_server_mode_unconfigured() -> None:
    # Server mode with no url and no injected graph → not configured.
    w = FalkorDBGraphWriter(_FakeSettings(url=None), mode="server")
    assert w.enabled is False


def test_enabled_true_when_lite_mode() -> None:
    # Lite is the embedded path: needs no url/graph to be enabled.
    w = FalkorDBGraphWriter(_FakeSettings(url=None), mode="lite")
    assert w.enabled is True


def test_enabled_true_when_server_url_set() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(url="redis://localhost:6379"), mode="server")
    assert w.enabled is True


def test_enabled_true_when_graph_injected() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(url=None), mode="server", graph=_FakeGraph())
    assert w.enabled is True


async def test_reset_pot_disabled_returns_error() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(enabled=False), graph=_FakeGraph())
    out = await w.reset_pot("pot-1")
    assert out == {"ok": False, "error": "context_graph_disabled"}


async def test_reset_pot_rejects_invalid_pot_id() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(), graph=_FakeGraph(count=3))
    out = await w.reset_pot("bad pot id")
    assert out["ok"] is False
    assert out["error"].startswith("invalid_pot_id")


async def test_reset_pot_success_shape() -> None:
    graph = _FakeGraph(count=3)
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    out = await w.reset_pot("pot-1")
    assert out == {
        "ok": True,
        "group_id_nodes_before": 3,
        "group_id_nodes_remaining": 0,
    }
    assert any("DETACH DELETE" in q for q, _ in graph.queries)


async def test_ensure_indexes_best_effort_swallows_errors() -> None:
    graph = _FakeGraph(raise_on_index=True)
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    # Even though every CREATE INDEX raises, ensure_indexes must not propagate.
    assert await w.ensure_indexes() is True
    assert sum("CREATE INDEX" in q for q, _ in graph.queries) == 4


async def test_ensure_indexes_uses_unnamed_form() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    await w.ensure_indexes()
    index_qs = [q for q, _ in graph.queries if "CREATE INDEX" in q]
    # FalkorDB rejects named indexes + IF NOT EXISTS; assert neither is used.
    for q in index_qs:
        assert "IF NOT EXISTS" not in q
        assert q.startswith("CREATE INDEX FOR")


async def test_upsert_entities_issues_merge_via_shim() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    prov = ProvenanceRef(pot_id="p1", source_event_id="e1")
    n = await w.upsert_entities(
        "p1", [EntityUpsert(entity_key="service:web", labels=("Entity",), properties={})], prov
    )
    assert n == 1
    assert any("MERGE (e:Entity" in q for q, _ in graph.queries)


async def test_upsert_entities_empty_is_noop() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    n = await w.upsert_entities("p1", [], ProvenanceRef(pot_id="p1", source_event_id="e1"))
    assert n == 0
    assert graph.queries == []


def test_build_falkordb_graph_server_mode_requires_url() -> None:
    # Server mode with no URL must fail loudly, not silently fall back to Lite.
    with pytest.raises(RuntimeError, match="server mode requires a URL"):
        build_falkordb_graph(_FakeSettings(url=None), mode="server")


def test_enabled_false_server_mode_no_url_even_with_provider() -> None:
    # The backend always injects a shared graph_provider; the enabled gate must
    # still report False for an unsatisfiable config (server mode, no URL), so
    # it never disagrees with what build_falkordb_graph can actually honor.
    w = FalkorDBGraphWriter(
        _FakeSettings(url=None),
        mode="server",
        graph_provider=lambda: _FakeGraph(),
    )
    assert w.enabled is False


def test_build_falkordb_graph_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown falkor mode"):
        build_falkordb_graph(_FakeSettings(), mode="bogus")
