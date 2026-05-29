"""Unit tests for Potpie CLI Firebase session helpers."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from adapters.inbound.cli.auth.firebase_session import (
    exchange_custom_token,
    refresh_id_token,
)


def test_exchange_custom_token_calls_firebase(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
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

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.firebase_session.httpx.Client",
        FakeClient,
    )
    session = exchange_custom_token("header.payload.signature", firebase_api_key="key")
    assert session.id_token == "id-token"
    assert session.refresh_token == "refresh-token"
    assert calls[0]["kwargs"]["json"] == {
        "token": "header.payload.signature",
        "returnSecureToken": True,
    }


def test_refresh_id_token_form_encodes_refresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
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

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.firebase_session.httpx.Client",
        FakeClient,
    )
    session = refresh_id_token("refresh+token/with=symbols", firebase_api_key="key")
    assert session.id_token == "new-id-token"
    assert session.refresh_token == "new-refresh-token"
    assert "refresh_token=refresh%2Btoken%2Fwith%3Dsymbols" in calls[0]["kwargs"]["content"]
