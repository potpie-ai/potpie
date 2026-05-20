"""FastAPI dependencies for Potpie API key auth (shared by /api/v2 routes and context graph)."""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.database import get_async_db, get_db
from app.modules.auth.api_key_service import APIKeyService
from app.modules.users.user_service import AsyncUserService

logger = logging.getLogger(__name__)

# Minimum acceptable INTERNAL_ADMIN_SECRET length. Anything shorter is rejected
# at use time so a weak secret cannot silently grant cross-user impersonation.
_ADMIN_SECRET_MIN_LENGTH = 32


def is_development_mode() -> bool:
    return os.getenv("isDevelopmentMode", "").strip().lower() == "enabled"


def dev_auth_enabled() -> bool:
    """F-11: dev-mode auth bypass requires BOTH `isDevelopmentMode=enabled` AND
    a second-gate env var `POTPIE_ALLOW_DEV_AUTH=1`.

    Why: `isDevelopmentMode` is set in a wide variety of compose/.env templates,
    so a single forgotten env var produces a fully unauthenticated deployment.
    Requiring the second flag forces an explicit opt-in (mirrors the
    `CONTEXT_ENGINE_ALLOW_NO_AUTH` pattern in the context-engine standalone).
    """
    if not is_development_mode():
        return False
    return (
        os.getenv("POTPIE_ALLOW_DEV_AUTH", "").strip().lower() in {"1", "true"}
    )


def _admin_secret_matches(candidate: str, expected: str) -> bool:
    """Constant-time check for the shared admin secret.

    Wraps `hmac.compare_digest` and rejects expected values shorter than the
    minimum allowed length (defence in depth against accidentally weak secrets).
    """
    if not expected or len(expected) < _ADMIN_SECRET_MIN_LENGTH:
        return False
    candidate_b = (candidate or "").encode("utf-8")
    expected_b = expected.encode("utf-8")
    return hmac.compare_digest(candidate_b, expected_b)


async def get_api_key_user(
    request: Request,
    x_api_key: str | None = Header(None),
    x_user_id: str | None = Header(None),
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
    candidate = (x_api_key or "").strip()
    if admin_secret and _admin_secret_matches(candidate, admin_secret):
        uid = (x_user_id or "").strip()
        if not uid:
            uid = os.getenv("defaultUsername", "").strip()
        user_svc = AsyncUserService(async_db)
        user = await user_svc.get_user_by_uid(uid)
        if not user and dev_auth_enabled():
            user = await user_svc.get_user_by_email("defaultuser@potpie.ai")
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid user_id",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        client_ip = request.client.host if request.client else None
        logger.warning(
            "INTERNAL_ADMIN_SECRET used to impersonate user_id=%s ip=%s path=%s",
            user.uid,
            client_ip,
            request.url.path,
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
