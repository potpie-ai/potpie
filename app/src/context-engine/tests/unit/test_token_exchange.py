"""Tests for OAuth token exchange and refresh."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.inbound.cli import token_exchange as tx


@pytest.fixture(autouse=True)
def _linear_client_id_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_CLIENT_ID", "test-linear-client-id")


def test_exchange_authorization_code_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "access_token": "access",
        "refresh_token": "refresh",
        "token_type": "Bearer",
        "scope": "read",
        "expires_in": 3600,
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    with patch("adapters.inbound.cli.token_exchange.httpx.Client", return_value=client):
        tokens = tx.exchange_authorization_code(
            "linear",
            code="auth-code",
            code_verifier="verifier",
            redirect_uri="http://localhost:8080/callback",
        )

    assert tokens["access_token"] == "access"
    assert tokens["refresh_token"] == "refresh"
    assert tokens["expires_at"] is not None


def test_exchange_authorization_code_rejects_non_linear() -> None:
    with pytest.raises(ValueError, match="only supported for Linear"):
        tx.exchange_authorization_code(
            "github",  # type: ignore[arg-type]
            code="c",
            code_verifier="v",
        )


def test_exchange_authorization_code_requires_verifier() -> None:
    with pytest.raises(ValueError, match="code_verifier is required"):
        tx.exchange_authorization_code("linear", code="c", code_verifier="")


def test_exchange_authorization_code_http_error() -> None:
    response = MagicMock()
    response.status_code = 400
    response.text = "bad request"
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    with patch("adapters.inbound.cli.token_exchange.httpx.Client", return_value=client):
        with pytest.raises(RuntimeError, match="Token exchange failed"):
            tx.exchange_authorization_code(
                "linear",
                code="c",
                code_verifier="v",
            )


def test_refresh_access_token_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "access_token": "new-access",
        "token_type": "Bearer",
        "expires_in": 1800,
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    with patch("adapters.inbound.cli.token_exchange.httpx.Client", return_value=client):
        tokens = tx.refresh_access_token("linear", refresh_token="rt")

    assert tokens["access_token"] == "new-access"
    assert tokens["refresh_token"] == "rt"


def test_refresh_access_token_requires_token() -> None:
    with pytest.raises(ValueError, match="refresh_token is required"):
        tx.refresh_access_token("linear", refresh_token="  ")


def test_exchange_missing_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tx, "get_client_id", lambda _p: "")
    with pytest.raises(ValueError, match="client id is not configured"):
        tx.exchange_authorization_code("linear", code="c", code_verifier="v")


def test_refresh_missing_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tx, "get_client_id", lambda _p: "")
    with pytest.raises(ValueError, match="client id is not configured"):
        tx.refresh_access_token("linear", refresh_token="rt")


def test_refresh_rejects_non_linear() -> None:
    with pytest.raises(ValueError, match="only supported for Linear"):
        tx.refresh_access_token("github", refresh_token="rt")  # type: ignore[arg-type]


def test_refresh_http_error() -> None:
    response = MagicMock()
    response.status_code = 401
    response.text = "unauthorized"
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    with patch("adapters.inbound.cli.token_exchange.httpx.Client", return_value=client):
        with pytest.raises(RuntimeError, match="Token refresh failed"):
            tx.refresh_access_token("linear", refresh_token="rt")
