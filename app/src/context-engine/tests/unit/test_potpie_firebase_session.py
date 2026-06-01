"""Unit tests for Potpie CLI Firebase session helpers."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
import pytest

from adapters.inbound.cli.auth import firebase_session
from adapters.inbound.cli.auth.firebase_session import (
    FirebaseSessionError,
    exchange_custom_token,
    id_token_expires_at,
    refresh_id_token,
)


def test_exchange_custom_token_calls_firebase(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            calls.append({"url": url, "kwargs": kwargs})
            return httpx.Response(
                200,
                json={
                    "idToken": "id-token",
                    "refreshToken": "refresh-token",
                    "expiresIn": "3600",
                },
            )

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    session = exchange_custom_token("header.payload.signature", firebase_api_key="key")

    assert session.id_token == "id-token"
    assert session.refresh_token == "refresh-token"
    assert calls[0]["url"].endswith("accounts:signInWithCustomToken?key=key")
    assert calls[0]["kwargs"]["json"] == {
        "token": "header.payload.signature",
        "returnSecureToken": True,
    }


def test_resolve_firebase_api_key_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "POTPIE_FIREBASE_API_KEY",
        "GOOGLE_IDENTITY_TOOL_KIT_KEY",
        "FIREBASE_API_KEY",
        "NEXT_PUBLIC_FIREBASE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEXT_PUBLIC_FIREBASE_API_KEY", "public-key")

    assert firebase_session.resolve_firebase_api_key() == "public-key"


def test_resolve_firebase_api_key_errors_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "POTPIE_FIREBASE_API_KEY",
        "GOOGLE_IDENTITY_TOOL_KIT_KEY",
        "FIREBASE_API_KEY",
        "NEXT_PUBLIC_FIREBASE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(FirebaseSessionError, match="Firebase API key missing"):
        firebase_session.resolve_firebase_api_key()


def test_exchange_custom_token_rejects_non_jwt_like_token() -> None:
    with pytest.raises(FirebaseSessionError, match="not JWT-like"):
        exchange_custom_token("not-a-jwt", firebase_api_key="key")


def test_exchange_custom_token_errors_on_firebase_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad token"})

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="custom token exchange failed"):
        exchange_custom_token("header.payload.signature", firebase_api_key="key")


def test_exchange_custom_token_requires_id_and_refresh_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, json={"idToken": "id-token"})

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="missing idToken or refreshToken"):
        exchange_custom_token("header.payload.signature", firebase_api_key="key")


def test_refresh_id_token_form_encodes_refresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            calls.append({"url": url, "kwargs": kwargs})
            return httpx.Response(
                200,
                json={
                    "id_token": "new-id-token",
                    "refresh_token": "new-refresh-token",
                    "expires_in": "3600",
                },
            )

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    session = refresh_id_token("refresh+token/with=symbols", firebase_api_key="key")

    assert session.id_token == "new-id-token"
    assert session.refresh_token == "new-refresh-token"
    assert calls[0]["url"].endswith("/token?key=key")
    assert (
        "refresh_token=refresh%2Btoken%2Fwith%3Dsymbols"
        in calls[0]["kwargs"]["content"]
    )


def test_refresh_id_token_errors_on_firebase_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad refresh"})

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="token refresh failed"):
        refresh_id_token("refresh-token", firebase_api_key="key")


def test_refresh_id_token_requires_response_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, json={"id_token": "id-token"})

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="missing id_token or refresh_token"):
        refresh_id_token("refresh-token", firebase_api_key="key")


def test_refresh_id_token_requires_refresh_token() -> None:
    with pytest.raises(FirebaseSessionError, match="refresh token is missing"):
        refresh_id_token(" ", firebase_api_key="key")


def test_id_token_expires_at_reads_jwt_exp() -> None:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": 12345}).encode()).decode()
    token = f"header.{payload.rstrip('=')}.signature"

    assert id_token_expires_at(token) == 12345.0


def test_id_token_expires_at_falls_back_for_invalid_token() -> None:
    before = time.time()
    expires_at = id_token_expires_at("not-a-token")

    assert before + 3590 <= expires_at <= before + 3610
