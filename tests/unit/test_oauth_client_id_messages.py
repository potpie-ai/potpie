"""Tests for OAuth client ID user-facing error messages."""

from __future__ import annotations

from potpie.auth.oauth_client_id_messages import (
    missing_github_client_id_message,
    missing_linear_client_id_message,
)


def test_missing_linear_client_id_message() -> None:
    message = missing_linear_client_id_message()
    assert "LINEAR_CLIENT_ID" in message
    assert ".env.template" in message
    assert "github.com/potpie-ai/potpie/issues" in message


def test_missing_github_client_id_message() -> None:
    message = missing_github_client_id_message()
    assert "POTPIE_GITHUB_CLIENT_ID" in message
    assert ".env.template" in message
    assert "github.com/potpie-ai/potpie/issues" in message
