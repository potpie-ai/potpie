"""Mount context-engine REST routes inside Potpie (auth + user-scoped projects)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from adapters.inbound.http.api.v1.context.router import (
    IngestPrRequest,
    SyncRequest,
    create_context_router,
)
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.context_graph.sync_enqueue import enqueue_backfill_with_container
from app.modules.context_graph.tasks import context_graph_ingest_pr
from app.modules.context_graph.wiring import build_container_for_user_session
from bootstrap.container import ContextEngineContainer

logger = logging.getLogger(__name__)


def get_context_engine_container(
    db: Session = Depends(get_db),
    user: dict = Depends(AuthService.check_auth),
) -> ContextEngineContainer:
    return build_container_for_user_session(db, user["user_id"])


class PotpieCeleryMutationHandlers:
    """Enqueue backfill / ingest to Celery (same ops model as context-graph UI)."""

    def handle_sync(
        self,
        payload: Optional[SyncRequest],
        container: ContextEngineContainer,
        db: Session,
    ) -> dict[str, Any]:
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is disabled (opt in by unsetting CONTEXT_GRAPH_ENABLED or setting true).",
            )
        mapping = payload.pot_ids if payload and payload.pot_ids else None
        try:
            return enqueue_backfill_with_container(container, db, mapping)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    def handle_ingest_pr(
        self,
        body: IngestPrRequest,
        container: ContextEngineContainer,
        _db: Session,
    ) -> dict[str, Any]:
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is disabled (opt in by unsetting CONTEXT_GRAPH_ENABLED or setting true).",
            )
        resolved = container.pots.resolve_pot(body.pot_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Unknown pot_id")
        try:
            context_graph_ingest_pr.delay(
                body.pot_id,
                body.pr_number,
                is_live_bridge=body.is_live_bridge,
                repo_name=body.repo_name,
            )
        except Exception as e:
            logger.warning(
                "Failed to enqueue ingest PR %s/%s: %s",
                body.pot_id,
                body.pr_number,
                e,
            )
            raise HTTPException(
                status_code=500, detail="Failed to enqueue ingest task"
            ) from e
        return {
            "status": "enqueued",
            "pot_id": body.pot_id,
            "pr_number": body.pr_number,
            "is_live_bridge": body.is_live_bridge,
        }


_potpie_mutations = PotpieCeleryMutationHandlers()

potpie_context_engine_router = create_context_router(
    require_auth=AuthService.check_auth,
    get_container=get_context_engine_container,
    get_db=get_db,
    mutation_handlers=_potpie_mutations,
    enforce_pot_access=True,
)
