"""Tests for CLI OAuth provider configuration."""

from __future__ import annotations

import pytest

from adapters.inbound.cli.provider_config import (
    LINEAR_TOKEN_URL,
    authorization_url,
    get_callback_host,
    get_callback_path,
    get_callback_port,
    get_client_id,
    get_client_secret,
    get_redirect_uri,
    get_scopes,
    token_url,
)


def test_get_client_id_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
    assert get_client_id("linear") == ""


def test_get_client_id_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_CLIENT_ID", "override-client-id")
    assert get_client_id("linear") == "override-client-id"


def test_redirect_and_callback_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI",
        "http://127.0.0.1:9001/custom/callback",
    )
    assert get_redirect_uri() == "http://127.0.0.1:9001/custom/callback"
    assert get_callback_host() == "127.0.0.1"
    assert get_callback_path() == "/custom/callback"
    assert get_callback_port() == 9001


def test_get_callback_port_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    monkeypatch.setenv("POTPIE_CLI_OAUTH_REDIRECT_URI", "http://localhost:8080/callback")
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "9999")
    assert get_callback_port() == 9999


def test_linear_oauth_urls() -> None:
    assert authorization_url("linear") == "https://linear.app/oauth/authorize"
    assert token_url("linear") == LINEAR_TOKEN_URL


def test_get_scopes_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_OAUTH_SCOPE", raising=False)
    assert get_scopes("linear") == "read"
    monkeypatch.setenv("LINEAR_OAUTH_SCOPE", "read,write")
    assert get_scopes("linear") == "read,write"


def test_invalid_redirect_uri_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POTPIE_CLI_OAUTH_REDIRECT_URI", "https://example.com/cb")
    with pytest.raises(ValueError, match="localhost"):
        get_callback_port()


def test_get_client_secret_linear_returns_empty() -> None:
    assert get_client_secret("linear") == ""


def test_get_callback_port_invalid_env_falls_back_to_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    monkeypatch.setenv("POTPIE_CLI_OAUTH_REDIRECT_URI", "http://localhost:8080/callback")
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "not-a-port")
    assert get_callback_port() == 8080


def test_redirect_uri_hostname_not_localhost_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI",
        "http://evil.example.com:8080/callback",
    )
    with pytest.raises(ValueError, match="localhost"):
        get_callback_host()


def test_unsupported_oauth_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported OAuth provider"):
        authorization_url("github")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported OAuth provider"):
        token_url("github")  # type: ignore[arg-type]
