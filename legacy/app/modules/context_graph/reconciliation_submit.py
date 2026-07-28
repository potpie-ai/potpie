"""Submit agent-reconciliation ingestion events (the current backfill path).

Backfill is no longer a standalone job: an admitted ``agent_reconciliation``
event lands in a batch, and the reconciliation agent (backfill playbooks)
enumerates and seeds the graph in the worker (see
``ContextGraphJobQueuePort.enqueue_batch``).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def submit_agent_reconciliation(
    db: Session, pot_id: str, *, trigger: str, repo_name: str | None = None
) -> dict[str, Any]:
    """Admit one ``agent_reconciliation`` event for ``pot_id``.

    Skips quietly when no context-graph pot exists for the id, and never
    raises — callers treat context-graph seeding as best-effort.
    """
    from app.modules.context_graph.wiring import build_ingestion_server_for_session
    from potpie_context_core.actor import Actor
    from potpie_context_engine.domain.ingestion_event_models import (
        IngestionSubmissionRequest,
    )
    from potpie_context_engine.domain.ingestion_kinds import (
        INGESTION_KIND_AGENT_RECONCILIATION,
    )

    try:
        container = build_ingestion_server_for_session(db)
        resolved = container.pots.resolve_pot(pot_id)
        if resolved is None:
            logger.info(
                "No context-graph pot %s; skipping reconciliation (%s)",
                pot_id,
                trigger,
            )
            return {"status": "skipped", "reason": "unknown_pot", "pot_id": pot_id}
        repo = repo_name or (resolved.repos[0].repo_name if resolved.repos else None)
        request = IngestionSubmissionRequest(
            pot_id=pot_id,
            ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
            source_channel="system",
            source_system="potpie",
            event_type="repository",
            action=trigger,
            repo_name=repo,
            payload={"pot_id": pot_id, "repo_name": repo, "trigger": trigger},
            actor=Actor(
                user_id="system",
                surface="system",
                client_name="potpie-host",
                auth_method="system",
            ),
        )
        receipt = container.ingestion_submission(db).submit(request, sync=False)
        logger.info(
            "Submitted context graph reconciliation for pot %s (%s): event %s",
            pot_id,
            trigger,
            receipt.event_id,
        )
        return {"status": "submitted", "pot_id": pot_id, "event_id": receipt.event_id}
    except Exception as e:
        logger.warning(
            "Reconciliation submission failed for pot %s (%s): %s", pot_id, trigger, e
        )
        return {"status": "error", "pot_id": pot_id, "reason": str(e)}
