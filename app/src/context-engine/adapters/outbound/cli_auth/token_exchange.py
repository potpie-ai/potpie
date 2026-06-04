"""Exchange OAuth authorization codes for access tokens (Linear CLI integration)."""

from __future__ import annotations

import time
from typing import Any

from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from adapters.outbound.cli_auth.provider_config import (
    OAuthProvider,
    get_client_id,
    get_redirect_uri,
    token_url,
)


def exchange_authorization_code(
    provider: OAuthProvider,
    *,
    code: str,
    code_verifier: str,
    redirect_uri: str | None = None,
    http: HttpClient | None = None,
) -> dict[str, Any]:
    """Exchange an authorization code (PKCE) for Linear OAuth tokens."""
    if provider != "linear":
        raise ValueError(f"OAuth exchange is only supported for Linear, not {provider!r}.")

    client_id = get_client_id(provider)
    if not client_id:
        raise ValueError(
            "Linear OAuth client id is not configured "
            "(set LINEAR_CLIENT_ID in your environment)."
        )
    if not code_verifier:
        raise ValueError("code_verifier is required for PKCE token exchange.")

    payload: dict[str, Any] = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri or get_redirect_uri(),
        "code_verifier": code_verifier,
    }

    owns = http is None
    http = http or AuthHttpClient()
    try:
        response = http.post(
            token_url(provider),
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except AuthHttpError as exc:
        raise RuntimeError(f"Token exchange request failed: {exc}") from exc
    finally:
        if owns:
            http.close()

    if response.status_code != 200:
        raise RuntimeError(
            f"Token exchange failed ({response.status_code}): {response.text}"
        )

    return _tokens_from_oauth_response(response.json())


def refresh_access_token(
    provider: OAuthProvider,
    *,
    refresh_token: str,
    http: HttpClient | None = None,
) -> dict[str, Any]:
    """Exchange a refresh token for a new Linear access token."""
    if provider != "linear":
        raise ValueError(f"OAuth refresh is only supported for Linear, not {provider!r}.")

    refresh_token = refresh_token.strip()
    if not refresh_token:
        raise ValueError("refresh_token is required.")

    client_id = get_client_id(provider)
    if not client_id:
        raise ValueError(
            "Linear OAuth client id is not configured "
            "(set LINEAR_CLIENT_ID in your environment)."
        )

    payload: dict[str, Any] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }

    owns = http is None
    http = http or AuthHttpClient()
    try:
        response = http.post(
            token_url(provider),
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except AuthHttpError as exc:
        raise RuntimeError(f"Token refresh request failed: {exc}") from exc
    finally:
        if owns:
            http.close()

    if response.status_code != 200:
        raise RuntimeError(
            f"Token refresh failed ({response.status_code}): {response.text}"
        )

    tokens = _tokens_from_oauth_response(response.json())
    if not tokens.get("refresh_token"):
        tokens["refresh_token"] = refresh_token
    return tokens


def _tokens_from_oauth_response(data: dict[str, Any]) -> dict[str, Any]:
    expires_in = int(data.get("expires_in") or 3600)
    return {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "token_type": data.get("token_type", "Bearer"),
        "scope": data.get("scope"),
        "expires_in": expires_in,
        "expires_at": time.time() + expires_in,
        "stored_at": time.time(),
    }
