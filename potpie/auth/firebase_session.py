"""Firebase REST session helpers for Potpie CLI auth."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import httpx

from potpie.auth.errors import CliAuthError
from potpie.auth.http import AuthHttpClient, AuthHttpError, HttpClient


class FirebaseSessionError(CliAuthError):
    """Expected Firebase session exchange or refresh failure."""


@dataclass(frozen=True)
class FirebaseSession:
    id_token: str
    refresh_token: str
    expires_at: float


def resolve_firebase_api_key() -> str:
    key = (
        os.getenv("POTPIE_FIREBASE_API_KEY")
        or os.getenv("GOOGLE_IDENTITY_TOOL_KIT_KEY")
        or os.getenv("FIREBASE_API_KEY")
        or os.getenv("NEXT_PUBLIC_FIREBASE_API_KEY")
        or ""
    ).strip()
    if not key:
        raise FirebaseSessionError(
            "Firebase API key missing. Set POTPIE_FIREBASE_API_KEY."
        )
    return key


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def _expires_at(expires_in: str | int | None) -> float:
    try:
        seconds = int(str(expires_in or "3600"))
    except ValueError:
        seconds = 3600
    return time.time() + max(seconds, 1)


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode((payload + padding).encode("ascii"))
        decoded = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def id_token_expires_at(id_token: str) -> float:
    exp = _decode_jwt_payload(id_token).get("exp")
    try:
        return float(exp)
    except (TypeError, ValueError):
        return time.time() + 3600


def exchange_custom_token(
    custom_token: str,
    *,
    firebase_api_key: str | None = None,
    http: HttpClient | None = None,
) -> FirebaseSession:
    token = custom_token.strip()
    if len(token.split(".")) != 3:
        raise FirebaseSessionError("Custom token was not JWT-like.")
    api_key = (firebase_api_key or resolve_firebase_api_key()).strip()
    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
        f"?key={api_key}"
    )
    owns = http is None
    http = http or AuthHttpClient()
    try:
        response = http.post(
            url,
            json={"token": token, "returnSecureToken": True},
            headers={"Content-Type": "application/json"},
        )
    except AuthHttpError as exc:
        raise FirebaseSessionError(
            f"Firebase custom token exchange request failed: {exc}"
        ) from exc
    finally:
        if owns:
            http.close()
    if response.status_code >= 300:
        raise FirebaseSessionError(
            f"Firebase custom token exchange failed: {_json_or_text(response)!r}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise FirebaseSessionError(
            f"Firebase custom token exchange response parsing failed: {exc}"
        ) from exc
    id_token = str(data.get("idToken") or "").strip()
    refresh_token = str(data.get("refreshToken") or "").strip()
    if not id_token or not refresh_token:
        raise FirebaseSessionError("Firebase response missing idToken or refreshToken.")
    return FirebaseSession(
        id_token=id_token,
        refresh_token=refresh_token,
        expires_at=_expires_at(data.get("expiresIn")),
    )


def refresh_id_token(
    refresh_token: str,
    *,
    firebase_api_key: str | None = None,
    http: HttpClient | None = None,
) -> FirebaseSession:
    token = refresh_token.strip()
    if not token:
        raise FirebaseSessionError("Firebase refresh token is missing.")
    api_key = (firebase_api_key or resolve_firebase_api_key()).strip()
    url = f"https://securetoken.googleapis.com/v1/token?key={api_key}"
    body = f"grant_type=refresh_token&refresh_token={quote_plus(token)}"
    owns = http is None
    http = http or AuthHttpClient()
    try:
        response = http.post(
            url,
            content=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except AuthHttpError as exc:
        raise FirebaseSessionError(
            f"Firebase token refresh request failed: {exc}"
        ) from exc
    finally:
        if owns:
            http.close()
    if response.status_code >= 300:
        raise FirebaseSessionError(
            f"Firebase token refresh failed: {_json_or_text(response)!r}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise FirebaseSessionError(
            f"Firebase token refresh response parsing failed: {exc}"
        ) from exc
    id_token = str(data.get("id_token") or "").strip()
    new_refresh_token = str(data.get("refresh_token") or "").strip()
    if not id_token or not new_refresh_token:
        raise FirebaseSessionError(
            "Firebase response missing id_token or refresh_token."
        )
    return FirebaseSession(
        id_token=id_token,
        refresh_token=new_refresh_token,
        expires_at=_expires_at(data.get("expires_in")),
    )
