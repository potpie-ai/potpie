"""Shared Celery enqueue for context-graph backfill (Potpie HTTP surfaces)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.modules.context_graph.tasks import context_graph_backfill_project
from bootstrap.container import ContextEngineContainer

logger = logging.getLogger(__name__)


def enqueue_backfill_with_container(
    container: ContextEngineContainer,
    db: Session,
    project_ids_filter: Optional[list[str]],
) -> dict[str, Any]:
    """
    Enqueue backfill for the given project IDs, or all known IDs from container.projects.

    Expects container.projects to implement known_project_ids() when filter is None.
    """
    mapping = project_ids_filter
    projects = container.projects
    if not hasattr(projects, "known_project_ids"):
        raise RuntimeError("Project resolution does not support listing IDs")
    ids = mapping or projects.known_project_ids()  # type: ignore[union-attr]
    results: list[dict[str, Any]] = []
    for pid in ids:
        resolved = container.projects.resolve(pid)
        if not resolved:
            results.append(
                {
                    "status": "skipped",
                    "project_id": pid,
                    "reason": "unknown_project_id",
                }
            )
            continue
        try:
            context_graph_backfill_project.delay(pid)
            results.append({"status": "enqueued", "project_id": pid})
        except Exception as e:
            logger.warning(
                "Failed to enqueue context graph backfill for %s: %s", pid, e
            )
            results.append(
                {
                    "status": "error",
                    "project_id": pid,
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
