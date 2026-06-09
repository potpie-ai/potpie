"""Parse a diff-sync cursor — the ISO-8601 "updated since" lower bound.

Diff-sync enumeration differs from backfill: instead of the fixed trailing
window in :mod:`domain.backfill_window`, a diff-sync run passes an explicit
cursor (the timestamp of the last successful graph audit) so the connector
enumerates only source refs changed since then. Connectors fall back to the
backfill window when no cursor is supplied, so the same list tool serves both
the one-shot backfill and the incremental diff-sync paths.

The parser is deliberately lenient: a malformed cursor returns ``None`` (the
caller then falls back to the backfill window) rather than raising, so a bad
value recorded in a history file can never crash an enumeration.
"""

from __future__ import annotations

from datetime import datetime, timezone


def parse_cursor_since(value: str | None) -> datetime | None:
    """Return a timezone-aware UTC cutoff for an ISO-8601 ``value``, or ``None``.

    Accepts ``2026-06-01T00:00:00Z`` or an explicit offset; a naive string is
    assumed UTC. Returns ``None`` for empty/unparseable input so callers can
    fall back to the default backfill window.
    """
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        # ``fromisoformat`` handles ``Z`` natively on 3.11+, but the replace
        # keeps explicit ``Z`` safe across inputs.
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


__all__ = ["parse_cursor_since"]
