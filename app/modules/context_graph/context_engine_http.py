"""Mount context-engine REST routes inside Potpie (auth + user-scoped context pots)."""

from __future__ import annotations

import logging

from fastapi import Depends
from sqlalchemy.orm import Session

from adapters.inbound.http.api.v1.context.router import create_context_router
from app.core.database import get_db
from app.modules.auth.api_key_deps import get_api_key_user
from app.modules.auth.auth_service import AuthService
from app.modules.context_graph.context_pot_routes import make_pot_router
from app.modules.context_graph.wiring import build_container_for_user_session
from bootstrap.ingestion_server import IngestionServerContainer

logger = logging.getLogger(__name__)


def get_context_engine_container(
    db: Session = Depends(get_db),
    user: dict = Depends(AuthService.check_auth),
) -> IngestionServerContainer:
    return build_container_for_user_session(db, user["user_id"])


def get_context_engine_container_for_api_key(
    user: dict = Depends(get_api_key_user),
    db: Session = Depends(get_db),
) -> IngestionServerContainer:
    """Same container shape as Firebase auth, for /api/v2/context (X-API-Key)."""
    return build_container_for_user_session(db, user["user_id"])


potpie_context_engine_router = create_context_router(
    require_auth=AuthService.check_auth,
    get_container=get_context_engine_container,
    get_db=get_db,
    get_db_optional=get_db,
)

# Pots CRUD available at /api/v1/context/pots (Firebase JWT auth)
potpie_context_pot_v1_router = make_pot_router(AuthService.check_auth)
