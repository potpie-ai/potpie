"""Bounded-window knobs for source backfill enumeration.

A backfill is a single ``*.added`` event the reconciliation agent fans out
into many artifacts via connector ``*_list_*`` tools, tracked on its todo
list. The todo list lives in the agent's context window, so the work has to
be *bounded* or it blows the context window, the tool-call budget, and the
agent run timeout. The bound is two knobs every connector's list tools share:

- ``CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS`` (default 365): only enumerate
  artifacts updated within this trailing window. ``0`` / negative disables
  the window (enumerate all history — use with a tight item cap).
- ``CONTEXT_ENGINE_BACKFILL_MAX_ITEMS`` (default 300): hard ceiling on
  enumerated artifacts per list call, sized so one agent run + checkpoint
  resume can drain the list. Connectors clamp any caller-supplied limit to
  this.

Older / overflow artifacts are not lost: they resolve lazily on demand via
the per-item fetch tools, and a follow-up backfill event can advance the
window. These are read live (not cached) so operators can retune without a
redeploy.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_DAYS = 365
_DEFAULT_MAX_ITEMS = 300


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s not an int: %r; using default %d", name, raw, default)
        return default


def backfill_window_days() -> int:
    """Trailing window in days; ``<= 0`` means "no window" (all history)."""
    return _env_int("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", _DEFAULT_WINDOW_DAYS)


def backfill_window_since(now: datetime | None = None) -> datetime | None:
    """UTC cutoff for "updated within the window", or ``None`` for all history."""
    days = backfill_window_days()
    if days <= 0:
        return None
    base = now or datetime.now(timezone.utc)
    return base - timedelta(days=days)


def backfill_max_items() -> int:
    """Hard ceiling on enumerated artifacts per list call (>= 1)."""
    return max(1, _env_int("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", _DEFAULT_MAX_ITEMS))


def clamp_backfill_limit(requested: int | None) -> int:
    """Clamp a caller/agent-supplied limit into ``[1, backfill_max_items()]``."""
    ceiling = backfill_max_items()
    if requested is None:
        return ceiling
    try:
        n = int(requested)
    except (TypeError, ValueError):
        return ceiling
    if n < 1:
        return 1
    return min(n, ceiling)


__all__ = [
    "backfill_window_days",
    "backfill_window_since",
    "backfill_max_items",
    "clamp_backfill_limit",
]
