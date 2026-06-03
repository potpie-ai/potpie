"""Legacy episode step status ↔ canonical mapping."""

from domain.ingestion_kinds import (
    EPISODE_STEP_APPLIED,
    EPISODE_STEP_APPLYING,
    EPISODE_STEP_FAILED,
    EPISODE_STEP_PENDING,
    EPISODE_STEP_QUEUED,
)
from domain.ingestion_step_db_status import (
    canonical_step_status_to_db,
    canonical_step_statuses_to_db_filters,
    db_step_status_to_canonical,
)


def test_db_to_canonical() -> None:
    assert db_step_status_to_canonical(EPISODE_STEP_PENDING) == "queued"
    assert db_step_status_to_canonical(EPISODE_STEP_QUEUED) == "queued"
    assert db_step_status_to_canonical(EPISODE_STEP_APPLYING) == "processing"
    assert db_step_status_to_canonical(EPISODE_STEP_APPLIED) == "done"
    assert db_step_status_to_canonical(EPISODE_STEP_FAILED) == "error"


def test_canonical_to_db() -> None:
    assert canonical_step_status_to_db("queued") == EPISODE_STEP_QUEUED
    assert canonical_step_status_to_db("processing") == EPISODE_STEP_APPLYING
    assert canonical_step_status_to_db("done") == EPISODE_STEP_APPLIED
    assert canonical_step_status_to_db("error") == EPISODE_STEP_FAILED


def test_filter_expansion() -> None:
    fs = canonical_step_statuses_to_db_filters(("queued", "done"))
    assert EPISODE_STEP_PENDING in fs
    assert EPISODE_STEP_APPLIED in fs
