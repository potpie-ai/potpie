"""Resolve Potpie API base URL and auth headers for thin CLI / MCP."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from bootstrap.runtime_settings import load_runtime_settings
from adapters.outbound.cli_auth.firebase_session import (
    FirebaseSessionError,
    FirebaseSession,
    id_token_expires_at,
    refresh_id_token,
)
from adapters.outbound.cli_auth.credentials_store import (
    get_potpie_auth_type,
    get_potpie_firebase_api_key,
    get_potpie_firebase_id_token,
    get_potpie_firebase_refresh_token,
    get_stored_api_base_url,
    get_stored_api_key,
    store_potpie_firebase_id_token,
    update_potpie_firebase_refresh_token,
)

_DEFAULT_API_URL = "http://localhost:8001"


@dataclass(frozen=True)
class PotpieAuthConfig:
    mode: str
    headers: dict[str, str]


def resolve_potpie_api_base_url() -> str:
    """Base URL only (no path), no trailing slash."""
    settings = load_runtime_settings()
    u = os.getenv("POTPIE_API_URL", "").strip() or get_stored_api_base_url() or ""
    if not u:
        u = settings.potpie_api_url
    u = u.rstrip("/")
    if not u:
        raise ValueError(
            "Potpie API base URL missing. Set POTPIE_API_URL or run `potpie login`."
        )
    return u


def resolve_potpie_api_key() -> str:
    k = (os.getenv("POTPIE_API_KEY") or get_stored_api_key() or "").strip()
    if not k:
        raise ValueError(
            "Potpie API key missing. Set POTPIE_API_KEY or run `potpie login-api-key <key>`."
        )
    return k


def resolve_potpie_firebase_session(
    refresh_token: str, *, force: bool = False
) -> FirebaseSession:
    """Resolve a Firebase session, reusing a still-valid cached ID token.

    ``force=True`` skips the cache and always exchanges the refresh token for a
    new ID token — used by the 401 re-auth path, where the cached token looked
    valid locally but the server rejected it.
    """
    token = refresh_token.strip()
    if not token:
        raise ValueError("Potpie Firebase refresh token missing.")

    if not force:
        cached_id_token = get_potpie_firebase_id_token()
        if cached_id_token and id_token_expires_at(cached_id_token) > time.time():
            return FirebaseSession(
                id_token=cached_id_token,
                refresh_token=token,
                expires_at=id_token_expires_at(cached_id_token),
            )

    try:
        firebase_api_key = get_potpie_firebase_api_key()
        session = refresh_id_token(
            token,
            firebase_api_key=firebase_api_key or None,
        )
    except FirebaseSessionError as exc:
        raise ValueError(f"Potpie Firebase session refresh failed: {exc}") from exc
    update_potpie_firebase_refresh_token(session.refresh_token)
    store_potpie_firebase_id_token(session.id_token)
    return session


def resolve_potpie_auth_config(*, force_refresh: bool = False) -> PotpieAuthConfig:
    """Resolve the active Potpie auth headers (env key > Firebase session > stored key).

    ``force_refresh=True`` forces a fresh Firebase ID token (no-op for API-key
    auth); pass it as the client's re-auth hook so a 401 retries with a genuinely
    new token rather than re-sending the rejected one.
    """
    api_key = (os.getenv("POTPIE_API_KEY") or "").strip()
    if api_key:
        return PotpieAuthConfig(mode="api_key", headers={"X-API-Key": api_key})

    auth_type = get_potpie_auth_type()
    refresh_token = get_potpie_firebase_refresh_token()
    stored_api_key = get_stored_api_key()

    if auth_type in {"potpie", "firebase_session"} and refresh_token:
        session = resolve_potpie_firebase_session(refresh_token, force=force_refresh)
        return PotpieAuthConfig(
            mode="potpie",
            headers={"Authorization": f"Bearer {session.id_token}"},
        )

    if stored_api_key:
        return PotpieAuthConfig(mode="api_key", headers={"X-API-Key": stored_api_key})

    if refresh_token:
        session = resolve_potpie_firebase_session(refresh_token, force=force_refresh)
        return PotpieAuthConfig(
            mode="potpie",
            headers={"Authorization": f"Bearer {session.id_token}"},
        )

    raise ValueError(
        "Potpie auth missing. Set POTPIE_API_KEY, run `potpie login-api-key <key>`, "
        "or run `potpie login`."
    )
