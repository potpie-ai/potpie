"""Shared Celery enqueue for context-graph backfill (Potpie HTTP surfaces)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.modules.context_graph.tasks import context_graph_backfill_pot
from bootstrap.container import ContextEngineContainer

logger = logging.getLogger(__name__)


def enqueue_backfill_with_container(
    container: ContextEngineContainer,
    db: Session,
    pot_ids_filter: Optional[list[str]],
) -> dict[str, Any]:
    """
    Enqueue backfill for the given pot IDs, or all known IDs from container.pots.

    Expects container.pots to implement known_pot_ids() when filter is None.
    """
    mapping = pot_ids_filter
    pots = container.pots
    if not hasattr(pots, "known_pot_ids"):
        raise RuntimeError("Pot resolution does not support listing IDs")
    ids = mapping or pots.known_pot_ids()  # type: ignore[union-attr]
    results: list[dict[str, Any]] = []
    for pid in ids:
        resolved = container.pots.resolve_pot(pid)
        if not resolved:
            results.append(
                {
                    "status": "skipped",
                    "pot_id": pid,
                    "reason": "unknown_pot_id",
                }
            )
            continue
        try:
            context_graph_backfill_pot.delay(pid)
            results.append({"status": "enqueued", "pot_id": pid})
        except Exception as e:
            logger.warning(
                "Failed to enqueue context graph backfill for %s: %s", pid, e
            )
            results.append(
                {
                    "status": "error",
                    "pot_id": pid,
                    "reason": str(e),
                }
            )
    enqueued = sum(1 for r in results if r.get("status") == "enqueued")
    return {
        "status": "success",
        "results": results,
        "enqueued": enqueued,
        "total_eligible": len(ids),
    }
