"""
Provider-agnostic token management utilities.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from sqlalchemy.orm import Session

from app.modules.users.user_model import User

logger = logging.getLogger(__name__)


class TokenService:
    """
    Provider-agnostic token management service.

    Handles token selection, storage, and refresh behaviour backed by the
    JSONB `provider_info` column on the `User` model. The goal is to keep the
    storage format flexible while providing a consistent API for callers.
    """

    def __init__(self, db: Session):
        self.db = db

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_best_token(
        self,
        user_id: str,
        provider: str = "github",
    ) -> Tuple[Optional[str], str, str]:
        """
        Return the best token available for the given user and provider.

        Priority order:
            1. Valid enhanced token (token_type != 'oauth')
            2. OAuth token (legacy format with no token_type field)
            3. None -> caller can fall back to public token pools

        Returns:
            (token, provider, auth_type)
        """
        try:
            provider_info = self._get_provider_info(user_id)
            if not provider_info:
                logger.warning("No provider_info for user %s", user_id)
                return None, provider, "none"

            access_token = provider_info.get("access_token")
            if not access_token:
                logger.warning(
                    "provider_info missing access_token for user %s", user_id
                )
                return None, provider, "none"

            token_type = provider_info.get("token_type") or "oauth"
            expires_at = provider_info.get("expires_at")

            if expires_at:
                if self._is_expired(expires_at):
                    logger.info(
                        "Token expired for user %s; attempting refresh", user_id
                    )
                    refreshed_token = self.refresh_token(user_id, provider)
                    if refreshed_token:
                        return refreshed_token, provider, token_type
                    logger.warning("Token refresh failed for user %s", user_id)
                    return None, provider, "none"

            logger.info("Using %s token for user %s", token_type, user_id)
            return access_token, provider, token_type
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error getting best token for user %s: %s", user_id, exc)
            return None, provider, "none"

    def generate_app_token(
        self,
        user_id: str,
        provider: str,
        token_payload: Dict[str, Any],
    ) -> bool:
        """
        Persist a newly generated provider-specific application token.
        """
        try:
            user = self._get_user(user_id)
            if not user:
                logger.error("Cannot generate token; user %s not found", user_id)
                return False

            provider_info = (
                deepcopy(user.provider_info)
                if isinstance(user.provider_info, dict)
                else {}
            )

            token = token_payload.get("token")
            if not token:
                logger.error("token_payload missing 'token' for user %s", user_id)
                return False

            expires_value = token_payload.get("expires_at")
            expires_at_iso = self._coerce_iso_datetime(expires_value)

            provider_info.update(
                {
                    "access_token": token,
                    "token_type": "app_user",
                    "expires_at": expires_at_iso,
                    "token_metadata": {
                        "created_via": f"{provider}_app",
                        "provider": provider,
                        "auth_method": token_payload.get(
                            "auth_method", "app_installation"
                        ),
                        "created_at": datetime.utcnow().isoformat(),
                    },
                }
            )

            if "installation_id" in token_payload:
                provider_info["installation_id"] = token_payload["installation_id"]

            # Persist any additional provider-specific fields
            for key, value in token_payload.items():
                if key not in {"token", "expires_at", "installation_id", "auth_method"}:
                    provider_info[key] = value

            user.provider_info = provider_info
            self.db.commit()
            self.db.refresh(user)

            logger.info("Stored app token for user %s (%s)", user_id, provider)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to store app token for user %s: %s", user_id, exc)
            self.db.rollback()
            return False

    def refresh_token(self, user_id: str, provider: str) -> Optional[str]:
        """
        Attempt to refresh an expired token using provider-specific logic.
        """
        try:
            provider_info = self._get_provider_info(user_id)
            if not provider_info:
                return None

            installation_id = provider_info.get("installation_id")
            if not installation_id:
                logger.warning(
                    "Cannot refresh token for user %s - missing installation_id",
                    user_id,
                )
                return None

            if provider == "github":
                from app.modules.code_provider.github.github_service import (
                    GithubService,
                )

                github_service = GithubService(self.db)
                token, expires_at = github_service.generate_github_app_user_token(
                    user_id, installation_id
                )
                if token and expires_at:
                    # generate_github_app_user_token already persists via TokenService
                    return token
                return None

            logger.warning(
                "Token refresh not implemented for provider %s (user %s)",
                provider,
                user_id,
            )
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error refreshing token for user %s: %s", user_id, exc)
            return None

    def get_token_info(self, user_id: str) -> Dict[str, Any]:
        """
        Return a descriptive snapshot of the user's provider token state.
        """
        try:
            provider_info = self._get_provider_info(user_id)
            if not provider_info:
                return {
                    "has_token": False,
                    "token_type": None,
                    "provider": None,
                    "expires_at": None,
                    "is_expired": None,
                }

            token_type = provider_info.get("token_type") or "oauth"
            expires_at = provider_info.get("expires_at")
            is_expired = None
            if expires_at:
                is_expired = self._is_expired(expires_at)

            metadata = provider_info.get("token_metadata") or {}
            provider = metadata.get("provider") or provider_info.get(
                "provider", "github"
            )

            return {
                "has_token": bool(provider_info.get("access_token")),
                "token_type": token_type,
                "provider": provider,
                "provider_id": provider_info.get("providerId", "unknown"),
                "expires_at": expires_at,
                "is_expired": is_expired,
                "installation_id": provider_info.get("installation_id"),
                "metadata": metadata if metadata else None,
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to read token info for user %s: %s", user_id, exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_user(self, user_id: str) -> Optional[User]:
        return self.db.query(User).filter(User.uid == user_id).first()

    def _get_provider_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        user = self._get_user(user_id)
        info = user.provider_info if user else None
        return deepcopy(info) if isinstance(info, dict) else None

    @staticmethod
    def _coerce_iso_datetime(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def _is_expired(expires_at: Any) -> bool:
        if not expires_at:
            return False
        if isinstance(expires_at, datetime):
            return expires_at <= datetime.utcnow()
        if isinstance(expires_at, str):
            try:
                parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                return parsed <= datetime.utcnow()
            except ValueError:
                logger.warning("Invalid expires_at format: %s", expires_at)
                return False
        return False
