"""Tests for integration OAuth session refresh."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from adapters.inbound.cli import integration_session as session


def test_ensure_valid_linear_expired_without_refresh_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = {"access_token": "old", "expires_at": 0.0}
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: expired)
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: True)

    result = session.ensure_valid_integration_tokens("linear")

    assert result == {}


def test_ensure_valid_linear_refreshes_when_refresh_token_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {
        "access_token": "old",
        "refresh_token": "rt",
        "expires_at": 0.0,
        "scope": "read",
    }
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: dict(stored))
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: True)
    monkeypatch.setattr(
        session,
        "refresh_access_token",
        lambda _provider, refresh_token: {
            "access_token": "new",
            "refresh_token": refresh_token,
            "expires_at": 9999999999.0,
        },
    )
    saved: list[dict] = []
    monkeypatch.setattr(
        session,
        "save_integration_tokens",
        lambda _provider, tokens: saved.append(tokens),
    )

    result = session.ensure_valid_integration_tokens("linear")

    assert result["access_token"] == "new"
    assert saved and saved[0]["access_token"] == "new"


def test_token_needs_refresh_with_buffer() -> None:
    import time as time_module

    future = time_module.time() + 1000
    assert session.token_needs_refresh(future, buffer_seconds=300) is False
    past = time_module.time() - 10
    assert session.token_needs_refresh(past, buffer_seconds=300) is True
    assert session.token_needs_refresh(None) is False
    assert session.token_needs_refresh("not-a-number") is False


def test_ensure_valid_jira_delegates_to_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        session,
        "get_integration_tokens",
        lambda provider: {"api_token": provider},
    )
    assert session.ensure_valid_integration_tokens("jira") == {"api_token": "jira"}


def test_ensure_valid_non_linear_provider_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session,
        "get_integration_tokens",
        lambda provider: {"provider": provider},
    )
    assert session.ensure_valid_integration_tokens("github") == {"provider": "github"}


def test_ensure_valid_linear_empty_access_token_returns_stored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {"access_token": "  ", "refresh_token": "rt", "expires_at": 0.0}
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: dict(stored))
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: True)

    result = session.ensure_valid_integration_tokens("linear")

    assert result == stored


def test_ensure_valid_linear_refresh_preserves_cloud_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {
        "access_token": "old",
        "refresh_token": "rt",
        "expires_at": 0.0,
        "cloud_id": "cid-1",
        "scope": "read",
    }
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: dict(stored))
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: True)
    monkeypatch.setattr(
        session,
        "refresh_access_token",
        lambda _provider, refresh_token: {
            "access_token": "new",
            "refresh_token": refresh_token,
            "expires_at": 9999999999.0,
        },
    )
    saved: list[dict] = []
    monkeypatch.setattr(
        session,
        "save_integration_tokens",
        lambda _provider, tokens: saved.append(tokens),
    )

    result = session.ensure_valid_integration_tokens("linear")

    assert result["cloud_id"] == "cid-1"
    assert saved and saved[0]["cloud_id"] == "cid-1"


def test_ensure_valid_linear_returns_tokens_when_not_due(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {"access_token": "ok", "expires_at": 9999999999.0}
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: dict(stored))
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: False)

    with patch.object(session, "refresh_access_token") as refresh:
        result = session.ensure_valid_integration_tokens("linear")

    assert result == stored
    refresh.assert_not_called()
