"""Integration profile metadata for credentials.json."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.inbound.cli import credentials_store as cs
from adapters.inbound.cli.integration_profile import (
    build_atlassian_integration_record,
    build_linear_integration_record,
    fetch_linear_viewer,
)


def test_build_linear_integration_record_includes_account_and_scopes() -> None:
    tokens = {
        "access_token": "lin-token",
        "refresh_token": "lin-refresh",
        "scope": "read,write",
        "expires_at": 999.0,
        "token_type": "Bearer",
    }
    viewer = {
        "account": {"id": "u1", "name": "Ada", "email": "ada@example.com"},
        "organization": {"id": "o1", "name": "Acme"},
    }
    with patch(
        "adapters.inbound.cli.integration_profile.fetch_linear_viewer",
        return_value=viewer,
    ):
        record = build_linear_integration_record(tokens)
    assert record["account"]["email"] == "ada@example.com"
    assert record["organization"]["name"] == "Acme"
    assert record["scopes"] == ["read", "write"]
    assert record["metadata"]["auth_flow"] == "pkce"
    assert record["created_at"]
    assert record["updated_at"]


def test_build_linear_integration_record_preserves_account_on_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = {
        "account": {"id": "u1", "name": "Ada", "email": "ada@example.com"},
        "created_at": "2020-01-01T00:00:00+00:00",
    }
    tokens = {"access_token": "new-token", "expires_at": 123.0}
    with patch(
        "adapters.inbound.cli.integration_profile.fetch_linear_viewer",
        return_value={},
    ):
        record = build_linear_integration_record(tokens, existing=prior)
    assert record["account"]["email"] == "ada@example.com"
    assert record["created_at"] == "2020-01-01T00:00:00+00:00"


def test_build_atlassian_integration_record_nested_and_flat() -> None:
    record = build_atlassian_integration_record(
        {
            "email": "user@example.com",
            "cloud_id": "cid",
            "site_url": "https://acme.atlassian.net",
            "site_name": "acme",
        }
    )
    assert record["account"]["email"] == "user@example.com"
    assert record["site"]["site_url"] == "https://acme.atlassian.net"
    assert record["email"] == "user@example.com"
    assert record["site_url"] == "https://acme.atlassian.net"


def test_save_integration_tokens_writes_account_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    monkeypatch.setattr(cs, "_store_keychain_secret", lambda *a, **k: None)
    viewer = {
        "account": {"id": "u1", "name": "Ada", "email": "ada@example.com"},
        "organization": {"name": "Acme"},
    }
    with patch(
        "adapters.inbound.cli.integration_profile.fetch_linear_viewer",
        return_value=viewer,
    ):
        cs.save_integration_tokens(
            "linear",
            {"access_token": "tok", "scope": "read", "expires_at": 1.0},
        )
    payload = cs.read_credentials()
    linear = payload["integrations"]["linear"]
    assert linear["account"]["email"] == "ada@example.com"
    assert linear["organization"]["name"] == "Acme"


def test_get_integration_status_reads_legacy_atlassian_flat_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    monkeypatch.setattr(cs, "_load_keychain_secret", lambda *a, **k: "legacy-token")
    monkeypatch.setattr(cs, "_store_keychain_secret", lambda *a, **k: None)
    cs._write_payload(
        {
            "integrations": {
                "atlassian": {
                    "provider": "atlassian",
                    "auth_type": "api_token",
                    "email": "legacy@example.com",
                    "site_url": "https://legacy.atlassian.net",
                    "site_name": "legacy",
                    "cloud_id": "cloud-1",
                }
            }
        }
    )
    status = cs.get_integration_status("atlassian")
    assert status["authenticated"] is True
    assert status["email"] == "legacy@example.com"
    assert status["site_url"] == "https://legacy.atlassian.net"


def test_fetch_linear_viewer_parses_graphql_response() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {
            "viewer": {
                "id": "v1",
                "name": "Nihit",
                "email": "n@example.com",
                "organization": {"id": "org1", "name": "Potpie"},
            }
        }
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    with patch("adapters.inbound.cli.integration_profile.httpx.Client", return_value=client):
        profile = fetch_linear_viewer("token")
    assert profile["account"]["name"] == "Nihit"
    assert profile["organization"]["name"] == "Potpie"
