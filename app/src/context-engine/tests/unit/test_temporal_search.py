"""Temporal ranking and flags for search rows."""

from datetime import datetime, timezone

from application.services.temporal_search import (
    annotate_search_rows_temporally,
    compute_temporal_flag,
)


def test_compute_temporal_flag_planned() -> None:
    ref = datetime(2025, 1, 1, tzinfo=timezone.utc)
    row = {"valid_at": "2025-06-01T00:00:00+00:00"}
    assert compute_temporal_flag(row, as_of=ref) == "planned"


def test_compute_temporal_flag_superseded() -> None:
    ref = datetime(2025, 8, 15, tzinfo=timezone.utc)
    row = {
        "valid_at": "2025-03-01T00:00:00+00:00",
        "invalid_at": "2025-08-12T00:00:00+00:00",
    }
    assert compute_temporal_flag(row, as_of=ref) == "superseded"


def test_annotate_rerank_superseded_last() -> None:
    ref = datetime(2025, 8, 15, tzinfo=timezone.utc)
    rows = [
        {
            "uuid": "old",
            "valid_at": "2025-03-01T00:00:00+00:00",
            "invalid_at": "2025-08-12T00:00:00+00:00",
        },
        {
            "uuid": "new",
            "valid_at": "2025-08-12T00:00:00+00:00",
            "invalid_at": None,
        },
    ]
    out = annotate_search_rows_temporally(
        rows, as_of=ref, include_invalidated=True
    )
    assert out[0]["uuid"] == "new"
    assert out[1]["uuid"] == "old"
    assert out[1].get("superseded_label") == "[superseded]"
