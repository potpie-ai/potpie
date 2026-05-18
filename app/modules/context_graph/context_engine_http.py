"""Mount context-engine REST routes inside Potpie (auth + user-scoped context pots)."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from adapters.inbound.http.api.v1.context.router import create_context_router
from app.core.database import get_db
from app.modules.auth.api_key_deps import get_api_key_user, is_development_mode
from app.modules.auth.api_key_service import APIKeyService
from app.modules.auth.auth_service import AuthService
from app.modules.context_graph.context_pot_routes import make_pot_router
from app.modules.context_graph.wiring import build_container_for_user_session
from app.modules.users.user_service import AsyncUserService
from bootstrap.container import ContextEngineContainer

logger = logging.getLogger(__name__)


async def resolve_user_from_api_key(
    api_key: str,
    db: Session,
    *,
    async_db: AsyncSession | None = None,
    x_user_id: str | None = None,
) -> dict[str, Any] | None:
    """Validate X-API-Key and return user dict (same rules as get_api_key_user)."""
    key = (api_key or "").strip()
    if not key:
        return None

    admin_secret = (os.environ.get("INTERNAL_ADMIN_SECRET") or "").strip()
    if admin_secret and key == admin_secret:
        uid = (x_user_id or "").strip() or os.getenv("defaultUsername", "").strip()
        if not uid:
            return None
        if async_db is None:
            raise ValueError("async_db required for admin secret API key validation")
        user_svc = AsyncUserService(async_db)
        user = await user_svc.get_user_by_uid(uid)
        if not user and is_development_mode():
            user = await user_svc.get_user_by_email("defaultuser@potpie.ai")
        if not user:
            return None
        return {"user_id": user.uid, "email": user.email, "auth_type": "api_key"}

    return await APIKeyService.validate_api_key(key, db)


async def build_container_from_api_key(
    api_key: str,
    db: Session,
    *,
    async_db: AsyncSession | None = None,
    x_user_id: str | None = None,
) -> ContextEngineContainer:
    """Build a user-scoped context engine container from an API key (embedded MCP / scripts)."""
    user = await resolve_user_from_api_key(
        api_key, db, async_db=async_db, x_user_id=x_user_id
    )
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return build_container_for_user_session(db, user["user_id"])


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


potpie_context_engine_router = create_context_router(
    require_auth=AuthService.check_auth,
    get_container=get_context_engine_container,
    get_db=get_db,
    get_db_optional=get_db,
)

# Pots CRUD available at /api/v1/context/pots (Firebase JWT auth)
potpie_context_pot_v1_router = make_pot_router(AuthService.check_auth)
