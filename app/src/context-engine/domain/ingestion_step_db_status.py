"""Map ``context_episode_steps.status`` (legacy) ↔ canonical :class:`IngestionStepStatus`."""

from __future__ import annotations

from domain.ingestion_event_models import IngestionStepStatus
from domain.ingestion_kinds import (
    EPISODE_STEP_APPLIED,
    EPISODE_STEP_APPLYING,
    EPISODE_STEP_FAILED,
    EPISODE_STEP_PENDING,
    EPISODE_STEP_QUEUED,
    EPISODE_STEP_SUPERSEDED,
)

_DB_PENDING = frozenset({EPISODE_STEP_PENDING, EPISODE_STEP_QUEUED})
_DB_PROCESSING = frozenset({EPISODE_STEP_APPLYING})
_DB_DONE = frozenset({EPISODE_STEP_APPLIED, EPISODE_STEP_SUPERSEDED})
_DB_ERROR = frozenset({EPISODE_STEP_FAILED})


def db_step_status_to_canonical(db_status: str) -> IngestionStepStatus:
    s = (db_status or "").strip()
    if s in _DB_PENDING:
        return "queued"
    if s in _DB_PROCESSING:
        return "processing"
    if s in _DB_DONE:
        return "done"
    if s in _DB_ERROR:
        return "error"
    return "queued"


def canonical_step_status_to_db(status: IngestionStepStatus) -> str:
    return {
        "queued": EPISODE_STEP_QUEUED,
        "processing": EPISODE_STEP_APPLYING,
        "done": EPISODE_STEP_APPLIED,
        "error": EPISODE_STEP_FAILED,
    }[status]


def canonical_step_statuses_to_db_filters(statuses: tuple[IngestionStepStatus, ...]) -> frozenset[str]:
    out: set[str] = set()
    for st in statuses:
        if st == "queued":
            out |= _DB_PENDING
        elif st == "processing":
            out |= _DB_PROCESSING
        elif st == "done":
            out |= _DB_DONE
        elif st == "error":
            out |= _DB_ERROR
    return frozenset(out)
