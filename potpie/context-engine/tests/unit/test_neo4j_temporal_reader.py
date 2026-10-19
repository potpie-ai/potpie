"""Shared temporal and semantic filters run before Neo4j reader limits."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from potpie_context_engine.adapters.outbound.graph.neo4j_reader import (
    Neo4jClaimQueryStore,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter

pytestmark = pytest.mark.unit
POINT = datetime(2026, 1, 2, tzinfo=timezone.utc)


class Session:
    def __init__(self, records):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **params):
        return self.records


class Driver:
    def __init__(self, records):
        self.records = records

    def session(self):
        return Session(self.records)


def record(identity, score, **properties):
    return {
        "score": score,
        "props": {
            "group_id": "pot",
            "record_id": identity,
            "name": "DEPENDS_ON",
            "subject_key": "service:api",
            "object_key": "service:db",
            "valid_from": "2026-01-01T05:30:00+05:30",
            "environment": "prod",
            "truth": "source_observation",
            "fact": "needle dependency",
            **properties,
        },
    }


@pytest.mark.parametrize("vector", [False, True], ids=["lexical", "vector"])
def test_post_filters_precede_ranking_and_final_limit(vector):
    records = [
        record("expired", 0.99, invalid_at="2026-01-02T05:30:00+05:30"),
        record("wrong-truth", 0.98, truth="agent_claim"),
        record("wrong-environment", 0.97, environment="staging"),
        record("wrong-text", 0.96, fact="haystack dependency"),
        record("eligible", 0.80),
    ]
    store = Neo4jClaimQueryStore(
        settings=object(),
        driver=Driver(records),
        embedder=SimpleNamespace(embed=lambda _: (1.0, 0.0, 0.0)) if vector else None,
    )
    rows = store.find_claims(
        ClaimQueryFilter(
            pot_id="pot",
            as_of=POINT,
            fact_query="needle",
            limit=1,
            environment_in=("prod",),
            truth_in=("source_observation",),
            exact_text_in=("needle",),
            exact_text_pattern=r".*\bneedle\b.*",
        )
    )
    assert [row.record_id for row in rows] == ["eligible"]
    if vector:
        assert rows[0].properties["semantic_similarity"] == 0.80


@pytest.mark.parametrize("vector", [False, True], ids=["lexical", "vector"])
def test_range_filters_use_hydrated_start_alias_and_equal_instants(vector):
    store = Neo4jClaimQueryStore(
        settings=object(),
        driver=Driver([record("eligible", 0.9)]),
        embedder=SimpleNamespace(embed=lambda _: (1.0, 0.0, 0.0)) if vector else None,
    )
    point = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = store.find_claims(
        ClaimQueryFilter(
            pot_id="pot",
            as_of=point,
            valid_at_after=point,
            valid_at_before=point,
            fact_query="needle",
            limit=1,
        )
    )
    assert [row.record_id for row in rows] == ["eligible"]


def test_zero_limit_does_not_query_or_embed():
    def unexpected(*args):
        pytest.fail("zero-row query must stop before driver or embedder work")

    store = Neo4jClaimQueryStore(
        settings=object(),
        driver=SimpleNamespace(session=unexpected),
        embedder=SimpleNamespace(embed=unexpected),
    )
    assert (
        store.find_claims(
            ClaimQueryFilter(pot_id="pot", as_of=POINT, fact_query="needle", limit=0)
        )
        == []
    )
