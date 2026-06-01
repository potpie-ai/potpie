"""Tests for Atlassian CLI read commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.inbound.cli import credentials_store as cs
from adapters.inbound.cli.atlassian_read import (
    AtlassianReadError,
    fetch_confluence_spaces_sample,
    fetch_jira_issues_sample,
)


def _mock_keychain(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    secrets: dict[str, str] = {}

    def _store(_label: str, username: str, value: str) -> None:
        secrets[username] = value

    def _load(_label: str, username: str) -> str | None:
        return secrets.get(username)

    monkeypatch.setattr(cs, "_store_keychain_secret", _store)
    monkeypatch.setattr(cs, "_load_keychain_secret", _load)
    return secrets


def test_fetch_jira_issues_sample(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    _mock_keychain(monkeypatch)
    cs.save_jira_credentials(
        {
            "email": "user@example.com",
            "api_token": "secret",
            "cloud_id": "cid-1",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
            "workspaces": {"jira_project": "ENG"},
        }
    )

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "issues": [
            {
                "key": "ENG-1",
                "fields": {
                    "summary": "Fix bug",
                    "status": {"name": "Done"},
                    "project": {"key": "ENG"},
                    "updated": "2026-01-01T00:00:00.000+0000",
                    "description": "Done",
                },
            }
        ]
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.return_value = response
    client.post.return_value = response

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        issues = fetch_jira_issues_sample(limit=5)

    assert len(issues) == 1
    assert issues[0]["key"] == "ENG-1"
    assert issues[0]["summary"] == "Fix bug"


def test_fetch_confluence_spaces_sample(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    _mock_keychain(monkeypatch)
    cs.save_confluence_credentials(
        {
            "email": "user@example.com",
            "api_token": "secret",
            "cloud_id": "cid-1",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        }
    )

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "results": [{"key": "ENG", "name": "Engineering", "type": "global", "_links": {}}]
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.return_value = response

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        spaces = fetch_confluence_spaces_sample(limit=5)

    assert len(spaces) == 1
    assert spaces[0]["key"] == "ENG"


def test_fetch_jira_requires_login(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    with pytest.raises(AtlassianReadError, match="not connected"):
        fetch_jira_issues_sample()


def test_jira_and_confluence_credentials_are_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    _mock_keychain(monkeypatch)
    cs.save_jira_credentials(
        {
            "email": "jira@example.com",
            "api_token": "jira-secret",
            "cloud_id": "cid-j",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        }
    )
    assert cs.get_jira_credentials()["email"] == "jira@example.com"
    assert not cs.get_confluence_credentials()
    cs.save_confluence_credentials(
        {
            "email": "wiki@example.com",
            "api_token": "wiki-secret",
            "cloud_id": "cid-c",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        }
    )
    assert cs.get_confluence_credentials()["email"] == "wiki@example.com"
    cs.clear_jira_credentials()
    assert not cs.get_jira_credentials()
    assert cs.get_confluence_credentials()["email"] == "wiki@example.com"
