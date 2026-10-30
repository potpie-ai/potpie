"""Embedding-dimension source of truth for vector index creation (#909).

The bundled ``HashingEmbedder`` emits 256-dim vectors and a wired
``SentenceTransformerEmbedder`` typically 384, while the index creators used
to fall back to a hard-coded 1536 whenever no embedder was wired. These tests
pin the fix: the created index width always comes from the configured
embedder, and no embedder means no vector index — never a guessed width.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph import neo4j_writer
from potpie_context_engine.adapters.outbound.graph.cypher import (
    ensure_canonical_indexes,
)
from potpie_context_engine.adapters.outbound.graph.embedding_dim import (
    resolve_embedding_dim,
)
from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
    FalkorDBGraphWriter,
    _index_statements,
)
from potpie_context_engine.adapters.outbound.intelligence import local_embedder
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_embedding_config(monkeypatch):
    """Pin width resolution to the model lookup table.

    ``SentenceTransformerEmbedder.dimensions`` honours the
    ``CONTEXT_ENGINE_EMBEDDING_DIMENSIONS`` env var and the local config
    before consulting the known-model table, so a configured developer
    machine would resolve a different width and break these assertions.
    """
    monkeypatch.delenv("CONTEXT_ENGINE_EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.setattr(local_embedder, "_local_config", lambda: {})


class _FakeResult:
    def __init__(self, header=None, result_set=None):
        self.header = header or []
        self.result_set = result_set or []


class _FakeGraph:
    """Captures FalkorDB queries; swallows index DDL."""

    def __init__(self):
        self.queries: list[str] = []

    def query(self, cypher: str, params=None):
        self.queries.append(cypher)
        return _FakeResult()


class _FakeFalkorSettings:
    def is_enabled(self) -> bool:
        return True

    def falkordb_url(self):
        return "redis://localhost:6379"

    def falkordb_graph_name(self) -> str:
        return "g"

    def falkordb_mode(self) -> str:
        return "server"


class _FakeNeo4jSettings:
    def is_enabled(self) -> bool:
        return True

    def neo4j_uri(self):
        return "bolt://localhost:7687"

    def neo4j_user(self):
        return "neo4j"

    def neo4j_password(self):
        return "secret"


class _FakeSession:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def run(self, query: str, **kwargs):
        self._calls.append((query, kwargs))
        return None


class _FakeDriver:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def session(self) -> _FakeSession:
        return _FakeSession(self.calls)


# ---------------------------------------------------------------------------
# resolve_embedding_dim — the shared source of truth
# ---------------------------------------------------------------------------


def test_resolve_returns_none_without_embedder() -> None:
    assert resolve_embedding_dim(None) is None


def test_resolve_returns_hashing_embedder_width() -> None:
    assert resolve_embedding_dim(HashingEmbedder()) == 256


def test_resolve_returns_sentence_transformer_width_without_loading_model() -> None:
    # all-MiniLM-L6-v2 is a known model, so the width resolves from the
    # lookup table — no torch download in unit tests.
    assert resolve_embedding_dim(SentenceTransformerEmbedder()) == 384


def test_resolve_rejects_embedder_without_declared_width() -> None:
    class _NoDims:
        def embed(self, text: str):
            return (0.0,)

    with pytest.raises(ValueError, match="does not declare"):
        resolve_embedding_dim(_NoDims())


def test_resolve_rejects_non_integer_width() -> None:
    class _FloatDims:
        dimensions = 256.5

    class _BoolDims:
        dimensions = True

    for embedder in (_FloatDims(), _BoolDims()):
        with pytest.raises(ValueError, match="non-integer"):
            resolve_embedding_dim(embedder)


def test_resolve_rejects_non_positive_width() -> None:
    class _BadDims:
        dimensions = 0

    with pytest.raises(ValueError, match="non-positive"):
        resolve_embedding_dim(_BadDims())


# ---------------------------------------------------------------------------
# FalkorDB writer
# ---------------------------------------------------------------------------


async def test_falkordb_skips_vector_index_without_embedder() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(_FakeFalkorSettings(), graph=graph)
    assert await w.ensure_indexes() is True
    assert not any("CREATE VECTOR INDEX" in q for q in graph.queries)
    # Scalar indexes are still created — lexical mode stays fully usable.
    assert sum("CREATE INDEX" in q for q in graph.queries) == 4


async def test_falkordb_index_width_matches_hashing_embedder() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(
        _FakeFalkorSettings(), graph=graph, embedder=HashingEmbedder()
    )
    await w.ensure_indexes()
    vector_qs = [q for q in graph.queries if "CREATE VECTOR INDEX" in q]
    assert len(vector_qs) == 1
    assert "dimension:256" in vector_qs[0]


async def test_falkordb_index_width_matches_sentence_transformer_embedder() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(
        _FakeFalkorSettings(), graph=graph, embedder=SentenceTransformerEmbedder()
    )
    await w.ensure_indexes()
    vector_qs = [q for q in graph.queries if "CREATE VECTOR INDEX" in q]
    assert len(vector_qs) == 1
    assert "dimension:384" in vector_qs[0]


def test_falkordb_index_statements_never_embed_a_default_width() -> None:
    assert not any("CREATE VECTOR INDEX" in q for q in _index_statements(None))
    assert "dimension:99" in _index_statements(99)[-1]


# ---------------------------------------------------------------------------
# Neo4j writer / shared canonical indexes
# ---------------------------------------------------------------------------


async def test_neo4j_canonical_indexes_use_given_width() -> None:
    driver = _FakeDriver()
    await ensure_canonical_indexes(driver, embedding_dim=384)
    vector = [c for c in driver.calls if "CREATE VECTOR INDEX" in c[0]]
    assert len(vector) == 1
    assert vector[0][1]["embedding_dim"] == 384


async def test_neo4j_canonical_indexes_skip_vector_index_without_width() -> None:
    driver = _FakeDriver()
    await ensure_canonical_indexes(driver, embedding_dim=None)
    assert not any("CREATE VECTOR INDEX" in q for q, _ in driver.calls)
    # The four scalar indexes are still created.
    assert sum("CREATE INDEX" in q for q, _ in driver.calls) == 4


async def test_neo4j_writer_resolves_width_from_wired_embedder(monkeypatch) -> None:
    captured: list[int | None] = []

    async def _capture(driver, *, embedding_dim=None):
        captured.append(embedding_dim)

    async def _with_driver(self, fn):
        return await fn(object())

    monkeypatch.setattr(neo4j_writer, "ensure_canonical_indexes", _capture)
    monkeypatch.setattr(neo4j_writer.Neo4jGraphWriter, "_with_driver", _with_driver)

    w = neo4j_writer.Neo4jGraphWriter(_FakeNeo4jSettings(), embedder=HashingEmbedder())
    assert await w.ensure_indexes() is True
    assert captured == [256]

    captured.clear()
    w = neo4j_writer.Neo4jGraphWriter(_FakeNeo4jSettings(), embedder=None)
    assert await w.ensure_indexes() is True
    assert captured == [None]
