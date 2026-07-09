"""Tests for the semantic-index re-embed repair (POT-1918 / P0-1).

The repair is both the recovery pass for silently-failed embedding attaches
and the migration pass for embedder changes (model or dimensions) — e.g. the
switch from the 256-dim hashing embedder to sentence-transformers.
"""

from __future__ import annotations

import pytest

from adapters.outbound.graph.backends.falkordb_backend import FalkorDBGraphBackend
from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from adapters.outbound.graph.semantic_index_repair import (
    claim_needs_reembed,
    wants_semantic_index_repair,
)
from domain.ports.claim_query import ClaimRow

pytestmark = pytest.mark.unit

POT = "p1"


class _Embedder:
    def __init__(self, name: str = "new-model", dimensions: int = 4) -> None:
        self.name = name
        self.dimensions = dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        return tuple(1.0 if i == 0 else 0.0 for i in range(self.dimensions))

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


def test_wants_semantic_index_repair_target_vocabulary() -> None:
    assert wants_semantic_index_repair(()) is True  # no targets = repair all
    assert wants_semantic_index_repair(("semantic_index",)) is True
    assert wants_semantic_index_repair(("embeddings",)) is True
    assert wants_semantic_index_repair(("entity_summaries",)) is False


def test_claim_needs_reembed_predicate() -> None:
    healthy = {
        "fact_embedding": [1.0, 0.0, 0.0, 0.0],
        "embedding_model": "new-model",
        "embedding_dim": 4,
    }
    assert (
        claim_needs_reembed(healthy, embedder_name="new-model", embedder_dim=4) is False
    )
    assert claim_needs_reembed({}, embedder_name="new-model", embedder_dim=4) is True
    assert (
        claim_needs_reembed(
            {**healthy, "embedding_model": "old-model"},
            embedder_name="new-model",
            embedder_dim=4,
        )
        is True
    )
    assert (
        claim_needs_reembed(
            {**healthy, "embedding_dim": 3},
            embedder_name="new-model",
            embedder_dim=4,
        )
        is True
    )


# ---------------------------------------------------------------------------
# FalkorDB backend repair (fake graph)
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, names, rows):
        self.header = [[1, name] for name in names]
        self.result_set = rows


def _claim_props(
    claim_key: str | None,
    *,
    embedding=None,
    model: str | None = None,
    dim: int | None = None,
) -> dict:
    props = {
        "group_id": POT,
        "name": "PROVIDES",
        "subject_key": "repo:x",
        "object_key": f"feature:{claim_key or 'anon'}",
        "fact": "repo provides a feature",
    }
    if claim_key:
        props["claim_key"] = claim_key
    if embedding is not None:
        props["fact_embedding"] = embedding
        props["embedding_model"] = model
        props["embedding_dim"] = dim
    return props


class _SemanticRepairGraph:
    """Fake graph: canned claim scan + records index DDL and re-embed SETs."""

    def __init__(self, scan_rows):
        self._scan_rows = scan_rows
        self.reembeds: list[dict] = []
        self.index_statements: list[str] = []

    def query(self, cypher: str, params: dict | None = None) -> _FakeResult:
        params = params or {}
        if "RETURN r{.*} AS props" in cypher and "LIMIT" in cypher:
            return _FakeResult(["props"], [[row] for row in self._scan_rows])
        if "DROP VECTOR INDEX" in cypher or "CREATE" in cypher:
            self.index_statements.append(cypher)
            return _FakeResult([], [])
        if "SET r.fact_embedding" in cypher:
            self.reembeds.append(params)
            return _FakeResult(["updated"], [[1]])
        return _FakeResult([], [])


def test_falkordb_semantic_repair_reembeds_stale_claims_and_migrates_index() -> None:
    healthy = _claim_props(
        "claim:healthy", embedding=[1.0, 0.0, 0.0, 0.0], model="new-model", dim=4
    )
    missing = _claim_props("claim:missing")
    stale_model = _claim_props(
        "claim:stale-model", embedding=[0.0, 1.0, 0.0, 0.0], model="old-model", dim=4
    )
    stale_dim = _claim_props(
        "claim:stale-dim", embedding=[0.0, 1.0, 0.0], model="new-model", dim=3
    )
    graph = _SemanticRepairGraph([healthy, missing, stale_model, stale_dim])

    class _Settings:
        def is_enabled(self):
            return True

        def falkordb_url(self):
            return None

        def falkordb_graph_name(self):
            return "context_graph"

        def falkordb_mode(self):
            return "lite"

        def falkordb_lite_path(self):
            return ".potpie/test/falkordb.db"

    backend = FalkorDBGraphBackend(
        _Settings(), graph_provider=lambda: graph, embedder=_Embedder()
    )

    report = backend.analytics.repair(POT, targets=["semantic_index"])

    assert report.repaired["semantic_index"] == 3
    assert "semantic_index_failed" not in report.repaired
    reembedded_keys = {p.get("claim_key") for p in graph.reembeds}
    assert reembedded_keys == {"claim:missing", "claim:stale-model", "claim:stale-dim"}
    # Every re-embed carries the active embedder's identity at its dimensions.
    for params in graph.reembeds:
        assert params["embedding_model"] == "new-model"
        assert params["embedding_dim"] == 4
        assert len(params["embedding"]) == 4
    # The 3-dim stale claim forces an index drop + recreate at 4 dims.
    assert any("DROP VECTOR INDEX" in s for s in graph.index_statements)
    assert any(
        "CREATE VECTOR INDEX" in s and "dimension:4" in s
        for s in graph.index_statements
    )
    assert "vector index rebuilt" in (report.detail or "")


def test_falkordb_semantic_repair_counts_failures() -> None:
    class _ZeroMatchGraph(_SemanticRepairGraph):
        def query(self, cypher: str, params: dict | None = None) -> _FakeResult:
            if "SET r.fact_embedding" in cypher:
                return _FakeResult(["updated"], [[0]])
            return super().query(cypher, params)

    graph = _ZeroMatchGraph([_claim_props("claim:missing")])

    class _Settings:
        def is_enabled(self):
            return True

        def falkordb_url(self):
            return None

        def falkordb_graph_name(self):
            return "context_graph"

        def falkordb_mode(self):
            return "lite"

        def falkordb_lite_path(self):
            return ".potpie/test/falkordb.db"

    backend = FalkorDBGraphBackend(
        _Settings(), graph_provider=lambda: graph, embedder=_Embedder()
    )

    report = backend.analytics.repair(POT, targets=["semantic_index"])

    assert report.repaired["semantic_index"] == 0
    assert report.repaired["semantic_index_failed"] == 1
    assert "FAILED" in (report.detail or "")


# ---------------------------------------------------------------------------
# In-memory backend repair
# ---------------------------------------------------------------------------


def _row(claim_key: str, *, fact_embedding=None) -> ClaimRow:
    return ClaimRow(
        pot_id=POT,
        predicate="PROVIDES",
        subject_key="repo:x",
        object_key=f"feature:{claim_key}",
        valid_at=None,
        invalid_at=None,
        evidence_strength="attested",
        source_system="agent",
        source_ref="repo:x:README",
        fact="repo provides a feature",
        properties={},
        fact_embedding=fact_embedding,
        claim_key=claim_key,
    )


def test_in_memory_semantic_repair_reembeds_missing_and_mis_sized() -> None:
    backend = InMemoryGraphBackend(embedder=_Embedder(dimensions=4))
    store = backend.claim_query
    store.add(_row("claim:missing"))
    store.add(_row("claim:short", fact_embedding=(1.0, 0.0, 0.0)))
    store.add(_row("claim:healthy", fact_embedding=(1.0, 0.0, 0.0, 0.0)))

    report = backend.analytics.repair(POT, targets=["semantic_index"])

    assert report.repaired["semantic_index"] == 2
    assert all(
        row.fact_embedding is not None and len(row.fact_embedding) == 4
        for row in store.rows
    )
