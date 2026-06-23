"""Unit tests for the Neo4j ClaimQueryPort adapter.

No live Neo4j: a fake driver/session feeds canned records so we exercise
param-building, row parsing (ISO → datetime, extras → properties), the
fact_query token-overlap stamping + ordering, and limit truncation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from context_engine.adapters.outbound.graph.neo4j_reader import Neo4jClaimQueryStore, _row_from_record
from context_engine.domain.ports.claim_query import ClaimQueryFilter

pytestmark = pytest.mark.unit


class _FakeSession:
    def __init__(self, records, captured):
        self._records = records
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **params):
        self._captured.append((cypher, params))
        return list(self._records)


class _FakeDriver:
    def __init__(self, records):
        self._records = records
        self.captured: list = []
        self.closed = False

    def session(self):
        return _FakeSession(self._records, self.captured)

    def close(self):
        self.closed = True


def _rec(**props):
    return {"props": props}


class _FakeEmbedder:
    name = "fake-embedder"
    dimensions = 3

    def embed(self, text: str) -> tuple[float, ...]:
        return (0.1, 0.2, 0.3)

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


def test_row_parsing_maps_reserved_and_extras() -> None:
    row = _row_from_record(
        _rec(
            group_id="p1",
            name="OWNED_BY",
            subject_key="service:web",
            object_key="team:platform",
            valid_at="2026-01-01T00:00:00+00:00",
            valid_until="2026-06-01T00:00:00+00:00",
            invalid_at=None,
            evidence_strength="hypothesized",
            claim_key="claim:p1:owned-by:web",
            subgraph="infra_topology",
            truth="source_observation",
            confidence="0.97",
            description="web ownership comes from CODEOWNERS",
            environment="prod",
            observed_at="2026-01-02T00:00:00+00:00",
            mutation_id="mut-1",
            source_refs=["repo:x:CODEOWNERS", "repo:x:owners.yaml"],
            evidence=json.dumps(
                [
                    {
                        "source_ref": "repo:x:CODEOWNERS",
                        "authority": "repository_metadata",
                    }
                ]
            ),
            graph_contract_version="1.5",
            ontology_version="2026-06-01",
            source_system="codeowners",
            source_ref="repo:x:CODEOWNERS",
            fact="web is owned by platform",
            code_scope=json.dumps({"language": "py"}),
            policy_kind="ownership",
        )
    )
    assert row.predicate == "OWNED_BY"
    assert row.subject_key == "service:web"
    assert row.object_key == "team:platform"
    assert row.valid_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert row.valid_until == datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert row.observed_at == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert row.claim_key == "claim:p1:owned-by:web"
    assert row.subgraph == "infra_topology"
    assert row.truth == "source_observation"
    assert row.confidence == pytest.approx(0.97)
    assert row.description == "web ownership comes from CODEOWNERS"
    assert row.environment == "prod"
    assert row.mutation_id == "mut-1"
    assert row.source_refs == ("repo:x:CODEOWNERS", "repo:x:owners.yaml")
    assert row.evidence == (
        {"source_ref": "repo:x:CODEOWNERS", "authority": "repository_metadata"},
    )
    assert row.graph_contract_version == "1.5"
    assert row.ontology_version == "2026-06-01"
    # Derived from V1.5 truth; the legacy edge value is ignored.
    assert row.evidence_strength == "deterministic"
    # Contract keys are lifted out; extras stay in properties and JSON extras
    # stored by Neo4j are decoded.
    assert "name" not in row.properties
    assert "truth" not in row.properties
    assert "environment" not in row.properties
    assert "claim_key" not in row.properties
    assert row.properties["code_scope"] == {"language": "py"}
    assert row.properties["policy_kind"] == "ownership"


def test_find_claims_builds_params_and_parses_rows() -> None:
    driver = _FakeDriver(
        [
            _rec(
                group_id="p1",
                name="DEPENDS_ON",
                subject_key="service:web",
                object_key="service:auth",
                valid_at="2026-02-01T00:00:00+00:00",
                evidence_strength="attested",
            )
        ]
    )
    store = Neo4jClaimQueryStore(settings=object(), driver=driver)  # type: ignore[arg-type]
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
    _, params = driver.captured[0]
    assert params["gid"] == "p1"
    assert params["preds"] == ["DEPENDS_ON"]
    assert params["include_invalid"] is False
    assert params["as_of"] == "2026-03-01T00:00:00+00:00"


def test_entity_properties_returns_node_props() -> None:
    driver = _FakeDriver(
        [
            {
                "props": {
                    "entity_key": "service:web",
                    "name": "web",
                    "summary": "Web frontend service.",
                    "description": "Web frontend service for browser clients.",
                }
            }
        ]
    )
    store = Neo4jClaimQueryStore(settings=object(), driver=driver)  # type: ignore[arg-type]

    props = store.entity_properties(pot_id="p1", entity_key="service:web")

    assert props["summary"] == "Web frontend service."
    assert props["description"] == "Web frontend service for browser clients."
    cypher, params = driver.captured[0]
    assert "RETURN properties(e) AS props" in cypher
    assert params == {"gid": "p1", "key": "service:web"}


def test_fact_query_stamps_similarity_and_orders() -> None:
    driver = _FakeDriver(
        [
            _rec(
                group_id="p1",
                name="X",
                subject_key="a",
                object_key="b",
                fact="connection pool exhausted in checkout",
            ),
            _rec(
                group_id="p1",
                name="X",
                subject_key="c",
                object_key="d",
                fact="unrelated note about logging",
            ),
        ]
    )
    store = Neo4jClaimQueryStore(settings=object(), driver=driver)  # type: ignore[arg-type]
    rows = store.find_claims(
        ClaimQueryFilter(pot_id="p1", fact_query="connection pool exhausted")
    )
    # Best lexical match first; every row carries a similarity stamp.
    assert rows[0].subject_key == "a"
    assert (
        rows[0].properties["semantic_similarity"]
        > rows[1].properties["semantic_similarity"]
    )


def test_fact_query_uses_native_relationship_vector_index_when_embedder_present() -> (
    None
):
    driver = _FakeDriver(
        [
            {
                "props": {
                    "group_id": "p1",
                    "name": "X",
                    "subject_key": "a",
                    "object_key": "b",
                    "fact": "connection pool exhausted in checkout",
                },
                "score": 0.91,
            }
        ]
    )
    store = Neo4jClaimQueryStore(
        settings=object(), driver=driver, embedder=_FakeEmbedder()
    )  # type: ignore[arg-type]

    rows = store.find_claims(
        ClaimQueryFilter(pot_id="p1", fact_query="connection pool exhausted", limit=3)
    )

    cypher, params = driver.captured[0]
    assert "db.index.vector.queryRelationships" in cypher
    assert params["embedding"] == [0.1, 0.2, 0.3]
    assert params["k"] == 50
    assert rows[0].subject_key == "a"
    assert rows[0].properties["semantic_similarity"] == pytest.approx(0.91)


def test_limit_truncates() -> None:
    driver = _FakeDriver(
        [
            _rec(group_id="p1", name="X", subject_key=str(i), object_key="o")
            for i in range(5)
        ]
    )
    store = Neo4jClaimQueryStore(settings=object(), driver=driver)  # type: ignore[arg-type]
    rows = store.find_claims(
        ClaimQueryFilter(pot_id="p1", predicate_in=("X",), limit=2)
    )
    assert len(rows) == 2


def test_entity_labels_maps_keys() -> None:
    driver = _FakeDriver(
        [
            {"key": "service:web", "labels": ["Entity", "Service"]},
            {"key": "team:platform", "labels": ["Entity", "Team"]},
        ]
    )
    store = Neo4jClaimQueryStore(settings=object(), driver=driver)  # type: ignore[arg-type]
    out = store.entity_labels(pot_id="p1", entity_keys=["service:web", "team:platform"])
    assert out["service:web"] == ("Entity", "Service")
    assert out["team:platform"] == ("Entity", "Team")
