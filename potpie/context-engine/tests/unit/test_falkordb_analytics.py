"""Unit tests for the aggregate-Cypher FalkorDB analytics adapter."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.claim_query_analytics import (
    ClaimQueryAnalytics,
)
from potpie_context_engine.adapters.outbound.graph.backends.falkordb_analytics import (
    FalkorDBAnalytics,
)
from potpie_context_core.ports.claim_query import ClaimRow

pytestmark = pytest.mark.unit


class _FakeResult:
    def __init__(self, result_set):
        self.result_set = result_set


class _AggregateGraph:
    """Answers the three aggregate queries with canned rows."""

    def __init__(self):
        self.captured: list[tuple[str, dict]] = []

    def query(self, cypher, params=None):
        self.captured.append((cypher, params or {}))
        if "count(DISTINCT r.name)" in cypher:
            return _FakeResult([[2365, 26, 3]])
        if "size(subjects)" in cypher:
            return _FakeResult([[790]])
        if "min(r.valid_at)" in cypher:
            return _FakeResult(
                [["2025-11-24T00:00:00Z", "2026-07-21T11:34:40.384594+00:00", 2365]]
            )
        raise AssertionError(f"unexpected query: {cypher}")


class _BrokenGraph:
    def query(self, cypher, params=None):
        raise RuntimeError("graph unavailable")


class _ScanClaimQuery:
    """Minimal ClaimQueryPort standing in for the fallback path."""

    def __init__(self, rows):
        self.rows = rows

    def find_claims(self, filter_):
        return list(self.rows)


def _row(valid_at=None, invalid_at=None):
    return ClaimRow(
        pot_id="p1",
        predicate="X",
        subject_key="a",
        object_key="b",
        valid_at=valid_at,
        invalid_at=invalid_at,
    )


def _fallback(rows=()):
    return ClaimQueryAnalytics(_ScanClaimQuery(rows))


def test_counts_uses_aggregate_queries() -> None:
    graph = _AggregateGraph()
    analytics = FalkorDBAnalytics(graph_provider=lambda: graph, fallback=_fallback())

    counts = analytics.counts("p1")

    assert counts == {
        "claims": 2365,
        "entities": 790,
        "predicates": 26,
        "invalidated": 3,
    }
    assert all(params == {"gid": "p1"} for _, params in graph.captured)


def test_freshness_normalizes_stamp_formats() -> None:
    graph = _AggregateGraph()
    analytics = FalkorDBAnalytics(graph_provider=lambda: graph, fallback=_fallback())

    freshness = analytics.freshness("p1")

    assert freshness["oldest"] == "2025-11-24T00:00:00+00:00"
    assert freshness["newest"] == "2026-07-21T11:34:40.384594+00:00"
    assert freshness["stamped_claims"] == 2365


def test_quality_derives_from_counts() -> None:
    graph = _AggregateGraph()
    analytics = FalkorDBAnalytics(graph_provider=lambda: graph, fallback=_fallback())

    assert analytics.quality("p1") == {
        "status": "ok",
        "open_conflicts": 0,
        "claim_count": 2365,
    }


def test_falls_back_to_scan_analytics_when_aggregates_fail() -> None:
    rows = [
        _row(valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        _row(invalid_at=datetime(2026, 2, 1, tzinfo=timezone.utc)),
    ]
    analytics = FalkorDBAnalytics(
        graph_provider=lambda: _BrokenGraph(), fallback=_fallback(rows)
    )

    counts = analytics.counts("p1")
    freshness = analytics.freshness("p1")

    assert counts["claims"] == 2
    assert counts["invalidated"] == 1
    assert freshness["stamped_claims"] == 1


def test_scan_freshness_survives_mixed_naive_and_aware_stamps() -> None:
    # Regression: naive + aware datetimes in one pot crashed min()/max().
    rows = [
        _row(valid_at=datetime(2025, 11, 24)),
        _row(valid_at=datetime(2026, 7, 21, 11, 34, tzinfo=timezone.utc)),
    ]

    freshness = _fallback(rows).freshness("p1")

    assert freshness["oldest"] == "2025-11-24T00:00:00+00:00"
    assert freshness["newest"] == "2026-07-21T11:34:00+00:00"
    assert freshness["stamped_claims"] == 2


def test_scan_repair_audits_document_keys_from_claim_rows() -> None:
    # The neo4j / falkordb-fallback path scans document keys off the claim
    # rows it already reads; nothing is written back.
    rows = [
        ClaimRow(
            pot_id="p1",
            predicate="RELATED_TO",
            subject_key="document:a1b2c3d4e5f6",
            object_key="service:payments-api",
        )
    ]

    report = _fallback(rows).repair("p1", targets=["document_keys"])

    assert report.repaired == {}
    assert [f.target for f in report.findings] == ["document_keys"]
    assert report.findings[0].samples == ("document:a1b2c3d4e5f6",)
    assert "legacy content-hash" in (report.detail or "")
