"""Legacy DB status ↔ canonical ingestion status mapping."""

from domain.ingestion_db_status import (
    canonical_status_to_db,
    canonical_statuses_to_db_filters,
    db_status_to_canonical,
)


def test_db_to_canonical() -> None:
    assert db_status_to_canonical("received") == "queued"
    assert db_status_to_canonical("queued") == "queued"
    assert db_status_to_canonical("processing") == "processing"
    assert db_status_to_canonical("episodes_queued") == "processing"
    assert db_status_to_canonical("reconciled") == "done"
    assert db_status_to_canonical("failed") == "error"


def test_canonical_to_db() -> None:
    assert canonical_status_to_db("queued") == "queued"
    assert canonical_status_to_db("processing") == "processing"
    assert canonical_status_to_db("done") == "reconciled"
    assert canonical_status_to_db("error") == "failed"


def test_filter_expansion() -> None:
    fs = canonical_statuses_to_db_filters(("queued", "done"))
    assert "received" in fs
    assert "reconciled" in fs
