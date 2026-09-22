from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from potpie_context_core.ports.claim_query import ClaimQueryFilter, ClaimRow
from potpie_context_core.semantic_mutation_lowering import _utc_iso
from potpie_context_engine.adapters.outbound.graph.falkordb_reader import (
    FalkorDBClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_reader import (
    Neo4jClaimQueryStore,
)
from potpie_context_engine.application.readers._common import ReadRequest
from potpie_context_engine.application.readers.timeline_reader import TimelineReader
from potpie_context_engine.domain.ranking import RankingService
from potpie_context_core.workbench_service import GraphWorkbenchService

pytestmark = pytest.mark.unit


def test_write_time_normalization_makes_offsets_backend_comparable() -> None:
    assert _utc_iso("2026-01-01T05:00:00+05:30") == "2025-12-31T23:30:00+00:00"
    assert _utc_iso("2026-01-01T00:00:00") == "2026-01-01T00:00:00+00:00"


def test_public_commit_verifies_future_claim_and_preserves_historical_read(
    tmp_path,
) -> None:
    backend = InMemoryGraphBackend()
    workbench = GraphWorkbenchService(
        backend=backend, plan_store=LocalJsonGraphPlanStore(home=tmp_path)
    )
    start = datetime.now(timezone.utc) + timedelta(days=2)
    end = start + timedelta(days=2)
    proposal = workbench.propose(
        {
            "operations": [
                {
                    "op": "link_entities",
                    "subgraph": "infra_topology",
                    "subject": {"key": "service:future", "type": "Service"},
                    "predicate": "DEPENDS_ON",
                    "object": {"key": "service:scheduled", "type": "Service"},
                    "truth": "source_observation",
                    "evidence": [{"source_ref": "repo:future-manifest"}],
                    "description": "future service dependency",
                    "valid_from": start.isoformat(),
                    "valid_until": end.isoformat(),
                }
            ]
        },
        pot_id="pot",
    )

    committed = workbench.commit(proposal.plan_id, pot_id="pot", verify=True)

    assert committed.ok is True
    assert committed.verification is not None
    assert committed.verification.ok is True
    assert committed.verification.missing_claim_keys == ()
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id="pot")) == []
    historical = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="pot", as_of=start + timedelta(days=1))
    )
    expired = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="pot", as_of=end))
    assert [row.claim_key for row in historical] == list(proposal.claim_keys)
    assert expired == []


def _row(
    key: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    invalid_at: datetime | None = None,
) -> ClaimRow:
    return ClaimRow(
        pot_id="pot",
        claim_key=key,
        predicate="RELATES_TO",
        subject_key=f"subject:{key}",
        object_key=f"object:{key}",
        valid_at=start,
        valid_until=end,
        invalid_at=invalid_at,
    )


def test_current_and_as_of_use_half_open_validity_interval() -> None:
    now = datetime.now(timezone.utc)
    store = InMemoryClaimQueryStore(
        rows=[
            _row("future-start", start=now + timedelta(days=1)),
            _row("expired", start=now - timedelta(days=2), end=now),
            _row(
                "future-end", start=now - timedelta(days=2), end=now + timedelta(days=1)
            ),
            _row(
                "corrected-later",
                start=now - timedelta(days=2),
                invalid_at=now + timedelta(days=1),
            ),
        ]
    )
    current = store.find_claims(ClaimQueryFilter(pot_id="pot"))
    assert {row.claim_key for row in current} == {"future-end", "corrected-later"}

    boundary = store.find_claims(
        ClaimQueryFilter(pot_id="pot", as_of=now + timedelta(days=1))
    )
    assert {row.claim_key for row in boundary} == {"future-start"}

    historical = store.find_claims(
        ClaimQueryFilter(pot_id="pot", as_of=now - timedelta(days=1))
    )
    assert {row.claim_key for row in historical} == {
        "expired",
        "future-end",
        "corrected-later",
    }


def test_validity_normalizes_naive_and_offset_datetimes_to_utc() -> None:
    store = InMemoryClaimQueryStore(
        rows=[
            _row(
                "offset",
                start=datetime.fromisoformat("2026-01-01T05:00:00+05:30"),
                end=datetime.fromisoformat("2026-01-01T06:00:00+05:30"),
            ),
            _row("naive", start=datetime(2025, 12, 31, 23, 0)),
        ]
    )
    rows = store.find_claims(
        ClaimQueryFilter(pot_id="pot", as_of=datetime(2026, 1, 1, tzinfo=timezone.utc))
    )
    assert {row.claim_key for row in rows} == {"offset", "naive"}


class _Neo4jSession:
    def __init__(self, captured: list[tuple[str, dict[str, object]]]) -> None:
        self.captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params: object):
        self.captured.append((query, params))
        return []


class _Neo4jDriver:
    def __init__(self) -> None:
        self.captured: list[tuple[str, dict[str, object]]] = []

    def session(self):
        return _Neo4jSession(self.captured)


class _FalkorResult:
    header: list[object] = []
    result_set: list[object] = []


class _FalkorGraph:
    def __init__(self) -> None:
        self.captured: list[tuple[str, dict[str, object]]] = []

    def query(self, query: str, params: dict[str, object]):
        self.captured.append((query, params))
        return _FalkorResult()


@pytest.mark.parametrize("adapter", ["neo4j", "falkordb"])
def test_database_adapters_apply_same_validity_predicate(adapter: str) -> None:
    point = datetime(2026, 9, 21, tzinfo=timezone.utc)
    if adapter == "neo4j":
        driver = _Neo4jDriver()
        Neo4jClaimQueryStore(settings=object(), driver=driver).find_claims(  # type: ignore[arg-type]
            ClaimQueryFilter(pot_id="pot", as_of=point)
        )
        query, params = driver.captured[0]
    else:
        graph = _FalkorGraph()
        FalkorDBClaimQueryStore(settings=object(), graph=graph).find_claims(  # type: ignore[arg-type]
            ClaimQueryFilter(pot_id="pot", as_of=point)
        )
        query, params = graph.captured[0]

    assert params["query_time"] == point.isoformat()
    assert "coalesce(r.valid_from, r.valid_at) <= $query_time" in query
    assert "$query_time < r.valid_until" in query
    assert "$query_time < r.invalid_at" in query


def test_timeline_event_date_is_identical_for_repo_service_and_source_routes() -> None:
    event = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)
    store = InMemoryClaimQueryStore(
        rows=[
            ClaimRow(
                pot_id="pot",
                claim_key="repo-link",
                predicate="MENTIONS",
                subject_key="activity:pr-1074",
                object_key="repo:potpie",
                valid_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
                source_ref="github:pr:1074",
            ),
            ClaimRow(
                pot_id="pot",
                claim_key="service-link",
                predicate="TOUCHED",
                subject_key="activity:pr-1074",
                object_key="service:daemon",
                valid_at=datetime(2026, 9, 20, tzinfo=timezone.utc),
                source_ref="resource:pr-1074",
            ),
            ClaimRow(
                pot_id="pot",
                claim_key="event-link",
                predicate="PERFORMED",
                subject_key="person:author",
                object_key="activity:pr-1074",
                valid_at=event,
                properties={"occurred_at": event.isoformat()},
                source_ref="github:pr:1074",
            ),
        ]
    )
    store.set_entity_properties(
        pot_id="pot",
        entity_key="activity:pr-1074",
        properties={"occurred_at": event.isoformat()},
    )
    # Activity metadata is authoritative even when a legacy edge disagrees.
    store.rows[-1] = dataclasses.replace(
        store.rows[-1],
        properties={"occurred_at": "2026-09-14T12:00:00+00:00"},
    )
    reader = TimelineReader(claim_query=store, ranker=RankingService())
    requests = (
        ReadRequest(
            pot_id="pot",
            scope={"repo": "repo:potpie"},
            as_of=datetime(2026, 9, 22, tzinfo=timezone.utc),
        ),
        ReadRequest(
            pot_id="pot",
            scope={"service": "service:daemon"},
            as_of=datetime(2026, 9, 22, tzinfo=timezone.utc),
        ),
        ReadRequest(
            pot_id="pot",
            source_refs=("resource:pr-1074",),
            as_of=datetime(2026, 9, 22, tzinfo=timezone.utc),
        ),
    )

    dates = [
        reader.read(request).items[0].candidate.payload["occurred_at"]
        for request in requests
    ]
    assert dates == [event.isoformat()] * 3
