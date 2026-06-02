"""Tests for integration verify probes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from adapters.inbound.cli.integration_verify import (
    _verify_linear,
    verify_integration_access,
)


def test_verify_linear_transport_error_returns_false() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.side_effect = httpx.ConnectError("offline")
    with patch("adapters.inbound.cli.integration_verify.httpx.Client", return_value=client):
        ok, message = _verify_linear("token")
    assert ok is False
    assert message == "Linear API request failed"


def test_verify_linear_non_json_body_returns_false() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    with patch("adapters.inbound.cli.integration_verify.httpx.Client", return_value=client):
        ok, message = _verify_linear("token")
    assert ok is False
    assert "non-JSON" in message


def test_verify_integration_access_linear_no_token() -> None:
    ok, message = verify_integration_access("linear", {})
    assert ok is False
    assert message == "not authenticated"


def test_verify_linear_success_without_org() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {"viewer": {"id": "v1", "name": "Ada", "email": "a@example.com"}}
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    with patch("adapters.inbound.cli.integration_verify.httpx.Client", return_value=client):
        ok, message = _verify_linear("token")
    assert ok is True
    assert "Ada" in message


def test_verify_integration_access_linear_expired() -> None:
    ok, message = verify_integration_access(
        "linear",
        {"access_token": "tok", "expires_at": 1.0},
    )
    assert ok is False
    assert "expired" in message


def test_verify_linear_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {
            "viewer": {
                "id": "v1",
                "name": "Ada",
                "email": "a@example.com",
                "organization": {"name": "Acme"},
            }
        }
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    with patch("adapters.inbound.cli.integration_verify.httpx.Client", return_value=client):
        ok, message = _verify_linear("token")
    assert ok is True
    assert "Ada" in message
    assert "Acme" in message


def test_verify_integration_access_linear_ignores_invalid_expires_at() -> None:
    with patch(
        "adapters.inbound.cli.integration_verify._verify_linear",
        return_value=(True, "ok (Ada)"),
    ) as verify:
        ok, message = verify_integration_access(
            "linear",
            {"access_token": "tok", "expires_at": "not-a-number"},
        )
    verify.assert_called_once_with("tok")
    assert ok is True
    assert message == "ok (Ada)"


def test_verify_linear_graphql_errors() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"errors": [{"message": "bad token"}]}
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    with patch("adapters.inbound.cli.integration_verify.httpx.Client", return_value=client):
        ok, message = _verify_linear("token")
    assert ok is False
    assert "rejected" in message


def test_verify_linear_non_200_status() -> None:
    response = MagicMock()
    response.status_code = 401
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    with patch("adapters.inbound.cli.integration_verify.httpx.Client", return_value=client):
        ok, message = _verify_linear("token")
    assert ok is False
    assert "HTTP 401" in message


def test_verify_integration_access_unknown_provider() -> None:
    ok, message = verify_integration_access("github", {})  # type: ignore[arg-type]
    assert ok is False
    assert "unknown provider" in message
