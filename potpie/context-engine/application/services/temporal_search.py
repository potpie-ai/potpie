"""Temporal ranking and flags for Graphiti search rows (see docs/context-graph-improvements/01)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Literal

TemporalFlag = Literal["current", "superseded", "planned"]


def _env_temporal_rerank_enabled() -> bool:
    raw = os.getenv("CONTEXT_ENGINE_TEMPORAL_RERANK", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def compute_temporal_flag(
    row: dict[str, Any],
    *,
    as_of: datetime | None,
    now: datetime | None = None,
) -> TemporalFlag:
    """Classify a search row using ``valid_at`` / ``invalid_at`` vs reference time."""
    ref = as_of if as_of is not None else (now or datetime.now(timezone.utc))
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    else:
        ref = ref.astimezone(timezone.utc)

    valid_at = _parse_dt(row.get("valid_at"))
    invalid_at = _parse_dt(row.get("invalid_at"))

    # Fact is valid at ref when valid_at <= ref < invalid_at (invalid_at exclusive).
    if valid_at is not None and ref < valid_at:
        return "planned"
    if invalid_at is not None and ref >= invalid_at:
        return "superseded"
    return "current"


def annotate_search_rows_temporally(
    rows: list[dict[str, Any]],
    *,
    as_of: datetime | None,
    include_invalidated: bool = False,
) -> list[dict[str, Any]]:
    """Add ``temporal_flag`` and optional ``superseded_label``; optionally rerank."""
    if not rows:
        return rows
    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        flag = compute_temporal_flag(r, as_of=as_of, now=now)
        r["temporal_flag"] = flag
        if include_invalidated and flag == "superseded":
            r["superseded_label"] = "[superseded]"
        out.append(r)

    if not _env_temporal_rerank_enabled():
        return out

    def sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
        flag = item.get("temporal_flag", "current")
        # Non-superseded first; then higher valid_at first.
        tier = 1 if flag == "superseded" else 0
        va = _parse_dt(item.get("valid_at"))
        va_ord = -va.timestamp() if va is not None else 0.0
        uuid = str(item.get("uuid") or "")
        return (tier, va_ord, uuid)

    return sorted(out, key=sort_key)
