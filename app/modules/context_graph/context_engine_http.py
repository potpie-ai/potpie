"""Mount context-engine REST routes inside Potpie (auth + user-scoped context pots)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from adapters.inbound.http.api.v1.context.router import (
    IngestPrRequest,
    SyncRequest,
    create_context_router,
)
from app.core.database import get_db
from app.modules.auth.api_key_deps import get_api_key_user
from app.modules.auth.auth_service import AuthService
from app.modules.context_graph.context_pot_routes import make_pot_router
from app.modules.context_graph.pot_access import api_key_user_id, require_pot_ingest
from app.modules.context_graph.sync_enqueue import enqueue_backfill_with_container
from app.modules.context_graph.wiring import build_container_for_user_session
from bootstrap.container import ContextEngineContainer

logger = logging.getLogger(__name__)


def get_context_engine_container(
    db: Session = Depends(get_db),
    user: dict = Depends(AuthService.check_auth),
) -> ContextEngineContainer:
    return build_container_for_user_session(db, user["user_id"])


def get_context_engine_container_for_api_key(
    user: dict = Depends(get_api_key_user),
    db: Session = Depends(get_db),
) -> ContextEngineContainer:
    """Same container shape as Firebase auth, for /api/v2/context (X-API-Key)."""
    return build_container_for_user_session(db, user["user_id"])


class PotpieContextGraphMutationHandlers:
    """Enqueue backfill / ingest-pr via the container job queue (Celery or Hatchet per wiring)."""

    def handle_sync(
        self,
        payload: SyncRequest | None,
        container: ContextEngineContainer,
        db: Session,
        *,
        auth_user: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is disabled (opt in by unsetting CONTEXT_GRAPH_ENABLED or setting true).",
            )
        mapping = payload.pot_ids if payload and payload.pot_ids else None
        pots = container.pots
        if not hasattr(pots, "known_pot_ids"):
            raise HTTPException(
                status_code=500,
                detail="Pot resolution does not support listing IDs",
            )
        ids = mapping or pots.known_pot_ids()  # type: ignore[union-attr]
        if auth_user is not None:
            uid = api_key_user_id(auth_user)
            for pid in ids:
                require_pot_ingest(db, uid, pid)
        try:
            return enqueue_backfill_with_container(container, db, mapping)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    def handle_ingest_pr(
        self,
        body: IngestPrRequest,
        container: ContextEngineContainer,
        db: Session,
        *,
        auth_user: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is disabled (opt in by unsetting CONTEXT_GRAPH_ENABLED or setting true).",
            )
        if auth_user is not None:
            require_pot_ingest(db, api_key_user_id(auth_user), body.pot_id)
        resolved = container.pots.resolve_pot(body.pot_id)
        if not resolved or not resolved.repos:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Unknown pot_id for this user (create with POST /api/v2/context/pots "
                    "and attach at least one repository)."
                ),
            )
        jobs = container.jobs
        if jobs is None:
            raise HTTPException(
                status_code=500,
                detail="Job queue not configured on context-engine container",
            )
        try:
            jobs.enqueue_ingest_pr(
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
            "repo_name": body.repo_name,
            "is_live_bridge": body.is_live_bridge,
        }


POTPIE_CONTEXT_GRAPH_MUTATIONS = PotpieContextGraphMutationHandlers()

potpie_context_engine_router = create_context_router(
    require_auth=AuthService.check_auth,
    get_container=get_context_engine_container,
    get_db=get_db,
    get_db_optional=get_db,
    mutation_handlers=POTPIE_CONTEXT_GRAPH_MUTATIONS,
    enforce_pot_access=True,
)

# Pots CRUD available at /api/v1/context/pots (Firebase JWT auth)
potpie_context_pot_v1_router = make_pot_router(AuthService.check_auth)
