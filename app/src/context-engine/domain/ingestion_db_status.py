"""Map ``context_events.status`` (legacy pipeline strings) ↔ canonical :class:`IngestionEventStatus`."""

from __future__ import annotations

from domain.ingestion_event_models import IngestionEventStatus

# Stored in ``context_events.status``; includes historical and async-pipeline values.
_DB_STATUSES_QUEUED = frozenset({"received", "queued"})
_DB_STATUSES_PROCESSING = frozenset({"processing", "episodes_queued", "applying"})
_DB_STATUSES_DONE = frozenset({"reconciled"})
_DB_STATUSES_ERROR = frozenset({"failed"})


def db_status_to_canonical(db_status: str) -> IngestionEventStatus:
    s = (db_status or "").strip()
    if s in _DB_STATUSES_QUEUED:
        return "queued"
    if s in _DB_STATUSES_PROCESSING:
        return "processing"
    if s in _DB_STATUSES_DONE:
        return "done"
    if s in _DB_STATUSES_ERROR:
        return "error"
    # Unknown future values: treat as in-flight rather than failing reads.
    return "processing"


def canonical_status_to_db(status: IngestionEventStatus) -> str:
    return {
        "queued": "queued",
        "processing": "processing",
        "done": "reconciled",
        "error": "failed",
    }[status]


def canonical_statuses_to_db_filters(statuses: tuple[IngestionEventStatus, ...]) -> frozenset[str]:
    """Expand canonical filters to the set of legacy DB strings to match."""
    out: set[str] = set()
    for s in statuses:
        if s == "queued":
            out |= _DB_STATUSES_QUEUED
        elif s == "processing":
            out |= _DB_STATUSES_PROCESSING
        elif s == "done":
            out |= _DB_STATUSES_DONE
        elif s == "error":
            out |= _DB_STATUSES_ERROR
    return frozenset(out)
