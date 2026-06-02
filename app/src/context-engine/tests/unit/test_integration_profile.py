"""Integration profile metadata for credentials.json."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.inbound.cli import credentials_store as cs
from adapters.inbound.cli.integration_profile import (
    build_atlassian_integration_record,
    build_product_integration_record,
)


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


def test_build_product_integration_record_sets_provider() -> None:
    record = build_product_integration_record(
        "jira",
        {
            "email": "user@example.com",
            "cloud_id": "cid",
            "site_url": "https://acme.atlassian.net",
            "site_name": "acme",
        },
    )
    assert record["provider"] == "jira"
    assert record["account"]["email"] == "user@example.com"


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


def test_clear_jira_credentials_removes_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cred_path = tmp_path / "credentials.json"
    deleted: list[str] = []
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    monkeypatch.setattr(
        cs,
        "_delete_keychain_secret",
        lambda _label, name: deleted.append(name),
    )
    monkeypatch.setattr(cs, "_load_keychain_secret", lambda *_a, **_k: "legacy-token")
    monkeypatch.setattr(cs, "_store_keychain_secret", lambda *_a, **_k: None)
    cs._write_payload(
        {
            "integrations": {
                "atlassian": {
                    "provider": "atlassian",
                    "auth_type": "api_token",
                    "email": "legacy@example.com",
                    "site_url": "https://legacy.atlassian.net",
                    "cloud_id": "cloud-1",
                }
            }
        }
    )
    assert cs.get_jira_credentials().get("api_token") == "legacy-token"

    cs.clear_jira_credentials()

    assert cs.get_jira_credentials() == {}
    assert cs._JIRA_TOKEN_SECRET in deleted
    assert cs._ATLASSIAN_LEGACY_TOKEN_SECRET in deleted
    assert "atlassian" not in cs.read_credentials().get("integrations", {})
    assert "jira" not in cs.read_credentials().get("integrations", {})


def test_atlassian_account_from_entry_email_only() -> None:
    from adapters.inbound.cli.integration_profile import atlassian_account_from_entry

    assert atlassian_account_from_entry({"email": "u@example.com"}) == {
        "email": "u@example.com"
    }


