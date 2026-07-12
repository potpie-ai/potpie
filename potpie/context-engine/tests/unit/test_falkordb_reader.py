"""Unit tests for the FalkorDB ClaimQueryPort adapter.

No live FalkorDB: an injected fake graph returns canned ``result_set`` rows so
we exercise param-building, row parsing (reusing ``_row_from_record``), the
fact_query token-overlap stamping + ordering, limit truncation, and
``entity_labels`` — mirroring ``test_neo4j_claim_query`` for parity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph.falkordb_reader import (
    FalkorDBClaimQueryStore,
)
from potpie_context_engine.domain.ports.claim_query import ClaimQueryFilter

pytestmark = pytest.mark.unit


class _FakeResult:
    def __init__(self, header, result_set):
        self.header = header
        self.result_set = result_set


class _FakeGraph:
    def __init__(self, header, result_set):
        self._header = header
        self._result_set = result_set
        self.captured: list[tuple[str, dict]] = []

    def query(self, cypher, params=None):
        self.captured.append((cypher, params or {}))
        return _FakeResult(self._header, self._result_set)


def _props_graph(*prop_dicts):
    return _FakeGraph(header=[[1, "props"]], result_set=[[p] for p in prop_dicts])


class _FakeEmbedder:
    name = "fake-embedder"
    dimensions = 3

    def embed(self, text: str) -> tuple[float, ...]:
        return (0.1, 0.2, 0.3)

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


class _FailingEmbedder:
    name = "failing-embedder"
    dimensions = 384

    def embed(self, text: str) -> tuple[float, ...]:
        raise RuntimeError("model unavailable")

    def embed_many(self, texts):
        raise RuntimeError("model unavailable")


def test_find_claims_builds_params_and_parses_rows() -> None:
    graph = _props_graph(
        {
            "group_id": "p1",
            "name": "DEPENDS_ON",
            "subject_key": "service:web",
            "object_key": "service:auth",
            "valid_at": "2026-02-01T00:00:00+00:00",
            "evidence_strength": "attested",
        }
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    rows = store.find_claims(
        ClaimQueryFilter(
            pot_id="p1",
            predicate_in=("DEPENDS_ON",),
            as_of=datetime(2026, 3, 1, tzinfo=timezone.utc),
            limit=10,
        )
    )
    assert len(rows) == 1
    assert rows[0].predicate == "DEPENDS_ON"
    assert rows[0].subject_key == "service:web"
    _, params = graph.captured[0]
    assert params["gid"] == "p1"
    assert params["preds"] == ["DEPENDS_ON"]
    assert params["include_invalid"] is False
    assert params["as_of"] == "2026-03-01T00:00:00+00:00"


def test_find_claims_hydrates_v15_metadata_from_rows() -> None:
    graph = _props_graph(
        {
            "group_id": "p1",
            "name": "DEPLOYED_TO",
            "subject_key": "service:web",
            "object_key": "environment:prod",
            "valid_at": "2026-02-01T00:00:00+00:00",
            "valid_until": "2026-05-01T00:00:00+00:00",
            "claim_key": "claim:p1:deploy:web:prod",
            "subgraph": "infra_topology",
            "truth": "source_observation",
            "confidence": 1.0,
            "description": "web is deployed to prod",
            "environment": "prod",
            "observed_at": "2026-02-02T00:00:00+00:00",
            "mutation_id": "mut-2",
            "source_refs": ["repo:x:k8s.yaml"],
            "evidence": [
                {"source_ref": "repo:x:k8s.yaml", "authority": "repository_metadata"}
            ],
            "graph_contract_version": "1.5",
            "ontology_version": "2026-06-01",
            "source_system": "kubernetes",
            "source_ref": "repo:x:k8s.yaml",
            "fact": "web is deployed to prod",
            "code_scope": {"service": "web"},
        }
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]

    row = store.find_claims(
        ClaimQueryFilter(pot_id="p1", predicate_in=("DEPLOYED_TO",), limit=1)
    )[0]

    assert row.claim_key == "claim:p1:deploy:web:prod"
    assert row.subgraph == "infra_topology"
    assert row.truth == "source_observation"
    assert row.evidence_strength == "deterministic"
    assert row.environment == "prod"
    assert row.valid_until == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert row.source_refs == ("repo:x:k8s.yaml",)
    assert row.evidence == (
        {"source_ref": "repo:x:k8s.yaml", "authority": "repository_metadata"},
    )
    assert row.graph_contract_version == "1.5"
    assert row.ontology_version == "2026-06-01"
    assert "truth" not in row.properties
    assert "environment" not in row.properties
    assert row.properties["code_scope"] == {"service": "web"}


def test_fact_query_stamps_similarity_and_orders() -> None:
    graph = _props_graph(
        {
            "group_id": "p1",
            "name": "X",
            "subject_key": "a",
            "object_key": "b",
            "fact": "connection pool exhausted in checkout",
        },
        {
            "group_id": "p1",
            "name": "X",
            "subject_key": "c",
            "object_key": "d",
            "fact": "unrelated note about logging",
        },
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    rows = store.find_claims(
        ClaimQueryFilter(pot_id="p1", fact_query="connection pool exhausted")
    )
    assert rows[0].subject_key == "a"
    assert (
        rows[0].properties["semantic_similarity"]
        > rows[1].properties["semantic_similarity"]
    )


def test_fact_query_uses_native_relationship_vector_index_when_embedder_present() -> (
    None
):
    graph = _FakeGraph(
        header=[[1, "props"], [1, "score"]],
        result_set=[
            [
                {
                    "group_id": "p1",
                    "name": "X",
                    "subject_key": "a",
                    "object_key": "b",
                    "fact": "connection pool exhausted in checkout",
                },
                0.12,
            ]
        ],
    )
    store = FalkorDBClaimQueryStore(
        settings=object(), graph=graph, embedder=_FakeEmbedder()
    )  # type: ignore[arg-type]

    rows = store.find_claims(
        ClaimQueryFilter(pot_id="p1", fact_query="connection pool exhausted", limit=3)
    )

    cypher, params = graph.captured[0]
    assert "db.idx.vector.queryRelationships" in cypher
    assert "vecf32($embedding)" in cypher
    assert "id(rel) = id(r)" in cypher
    assert "ORDER BY score ASC" in cypher
    assert params["embedding"] == [0.1, 0.2, 0.3]
    assert params["k"] == 50
    assert rows[0].subject_key == "a"
    assert rows[0].properties["semantic_similarity"] == pytest.approx(0.88)


def test_fact_query_falls_back_to_lexical_when_embedder_fails() -> None:
    graph = _props_graph(
        {
            "group_id": "p1",
            "name": "X",
            "subject_key": "a",
            "object_key": "b",
            "fact": "connection pool exhausted in checkout",
        },
        {
            "group_id": "p1",
            "name": "X",
            "subject_key": "c",
            "object_key": "d",
            "fact": "unrelated note about logging",
        },
    )
    store = FalkorDBClaimQueryStore(
        settings=object(), graph=graph, embedder=_FailingEmbedder()
    )  # type: ignore[arg-type]

    rows = store.find_claims(
        ClaimQueryFilter(pot_id="p1", fact_query="connection pool exhausted")
    )

    assert "db.idx.vector.queryRelationships" not in graph.captured[0][0]
    assert rows[0].subject_key == "a"
    assert rows[0].properties["semantic_similarity"] > 0


def test_limit_truncates() -> None:
    graph = _props_graph(
        *[
            {"group_id": "p1", "name": "X", "subject_key": str(i), "object_key": "o"}
            for i in range(5)
        ]
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    rows = store.find_claims(
        ClaimQueryFilter(pot_id="p1", predicate_in=("X",), limit=2)
    )
    assert len(rows) == 2


def test_entity_labels_maps_keys() -> None:
    graph = _FakeGraph(
        header=[[1, "key"], [1, "labels"]],
        result_set=[
            ["service:web", ["Entity", "Service"]],
            ["team:platform", ["Entity", "Team"]],
        ],
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    out = store.entity_labels(pot_id="p1", entity_keys=["service:web", "team:platform"])
    assert out["service:web"] == ("Entity", "Service")
    assert out["team:platform"] == ("Entity", "Team")


def test_entity_labels_empty_keys_short_circuits() -> None:
    graph = _FakeGraph(header=[], result_set=[])
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    assert store.entity_labels(pot_id="p1", entity_keys=[]) == {}
    assert graph.captured == []


def test_entity_properties_returns_node_props() -> None:
    graph = _FakeGraph(
        header=[[1, "props"]],
        result_set=[
            [
                {
                    "entity_key": "service:web",
                    "name": "web",
                    "summary": "Web frontend service.",
                    "description": "Web frontend service for browser clients.",
                }
            ]
        ],
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]

    props = store.entity_properties(pot_id="p1", entity_key="service:web")

    assert props["summary"] == "Web frontend service."
    assert props["description"] == "Web frontend service for browser clients."
    cypher, params = graph.captured[0]
    assert "RETURN properties(e) AS props" in cypher
    assert params == {"gid": "p1", "key": "service:web"}
