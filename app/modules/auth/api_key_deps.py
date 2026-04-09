"""FastAPI dependencies for Potpie API key auth (shared by /api/v2 routes and context graph)."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.database import get_async_db, get_db
from app.modules.auth.api_key_service import APIKeyService
from app.modules.users.user_service import AsyncUserService


def is_development_mode() -> bool:
    return os.getenv("isDevelopmentMode", "").strip().lower() == "enabled"


async def get_api_key_user(
    x_api_key: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    async_db: AsyncSession = Depends(get_async_db),
) -> dict:
    """Validate API key and return user dict (user_id, email, auth_type).

    Normal API keys identify one user (no extra headers).

    INTERNAL_ADMIN_SECRET is a shared server secret: it can impersonate any user, so the
    caller must send X-User-Id (users.uid). In development, if the uid is missing we
    fall back to defaultUsername; if lookup still fails we try the dummy dev email
    (defaultuser@potpie.ai) so local CLI matches setup_dummy_user without hand-copying uids.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    admin_secret = (os.environ.get("INTERNAL_ADMIN_SECRET") or "").strip()
    if admin_secret and (x_api_key or "").strip() == admin_secret:
        uid = (x_user_id or "").strip()
        if not uid:
            uid = (os.getenv("defaultUsername") or "").strip()
        user_svc = AsyncUserService(async_db)
        user = await user_svc.get_user_by_uid(uid)
        if not user and is_development_mode():
            user = await user_svc.get_user_by_email("defaultuser@potpie.ai")
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid user_id",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return {"user_id": user.uid, "email": user.email, "auth_type": "api_key"}

    user = await APIKeyService.validate_api_key(x_api_key, db)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return user
