from __future__ import annotations

from datetime import datetime, timezone

from potpie_context_core.api import (
    ClaimRow,
    GraphReadRequest,
    from_transport,
    to_transport,
)


def test_graph_read_request_transport_round_trip_restores_datetimes() -> None:
    request = GraphReadRequest(
        pot_id="pot:transport",
        subgraph="recent_changes",
        view="timeline",
        since=datetime(2026, 7, 1, tzinfo=timezone.utc),
        until=datetime(2026, 7, 2, tzinfo=timezone.utc),
        source_refs=("github:potpie-ai/potpie#1018",),
    )

    restored = from_transport(GraphReadRequest, to_transport(request))

    assert restored == request
    assert isinstance(restored.since, datetime)
    assert isinstance(restored.until, datetime)


def test_claim_row_transport_round_trip_restores_typed_collections() -> None:
    row = ClaimRow(
        pot_id="pot:transport",
        predicate="RELATED_TO",
        subject_key="service:api",
        object_key="service:worker",
        valid_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        source_refs=("repo:manifest",),
    )

    restored = from_transport(ClaimRow, to_transport(row))

    assert restored == row
    assert isinstance(restored.valid_at, datetime)
    assert isinstance(restored.source_refs, tuple)
