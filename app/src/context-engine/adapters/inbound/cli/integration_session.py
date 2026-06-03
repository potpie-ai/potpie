"""Ensure integration OAuth tokens are valid, refreshing access tokens when needed."""

from __future__ import annotations

import time
from typing import Any

from adapters.inbound.cli.credentials_store import (
    get_integration_tokens,
    save_integration_tokens,
)
from adapters.inbound.cli.provider_config import Provider
from adapters.inbound.cli.token_exchange import refresh_access_token

REFRESH_BUFFER_SECONDS = 300


def token_needs_refresh(
    expires_at: Any,
    *,
    buffer_seconds: int = REFRESH_BUFFER_SECONDS,
) -> bool:
    """True when the access token is expired or within the refresh buffer."""
    if expires_at is None:
        return False
    try:
        return time.time() >= float(expires_at) - buffer_seconds
    except (TypeError, ValueError):
        return False


def ensure_valid_integration_tokens(provider: Provider) -> dict[str, Any]:
    """Return stored tokens, refreshing OAuth access tokens for Linear when near expiry."""
    tokens = get_integration_tokens(provider)
    access_token = str(tokens.get("access_token") or "").strip()
    if not access_token:
        return tokens
    if not token_needs_refresh(tokens.get("expires_at")):
        return tokens

    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not refresh_token:
        # token_needs_refresh is true but we cannot refresh — do not return stale creds.
        return {}

    refreshed = refresh_access_token("linear", refresh_token=refresh_token)
    merged: dict[str, Any] = {**tokens, **refreshed}
    for field in ("cloud_id", "scope"):
        if not merged.get(field) and tokens.get(field):
            merged[field] = tokens[field]
    save_integration_tokens(provider, merged)
    return merged
