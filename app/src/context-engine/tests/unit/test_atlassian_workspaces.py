"""Tests for Atlassian workspace list/use helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.inbound.cli import credentials_store as cs
from adapters.inbound.cli.atlassian_read import (
    fetch_confluence_pages_in_space,
    fetch_jira_issues_in_project,
    fetch_jira_projects,
)


def _save_creds(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cred_path = tmp_path / "credentials.json"
    secrets: dict[str, str] = {}

    def _store(_label: str, username: str, value: str) -> None:
        secrets[username] = value

    def _load(_label: str, username: str) -> str | None:
        return secrets.get(username)

    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    monkeypatch.setattr(cs, "_store_keychain_secret", _store)
    monkeypatch.setattr(cs, "_load_keychain_secret", _load)
    cs.save_jira_credentials(
        {
            "email": "user@example.com",
            "api_token": "secret",
            "cloud_id": "cid-1",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        }
    )


def test_fetch_jira_issues_in_project(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _save_creds(monkeypatch, tmp_path)
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "issues": [
            {
                "key": "ENG-9",
                "fields": {
                    "summary": "Ship CLI",
                    "status": {"name": "In Progress"},
                    "project": {"key": "ENG"},
                    "description": {"type": "doc", "content": [{"type": "text", "text": "Details"}]},
                "assignee": {"displayName": "Ada"},
                "priority": {"name": "High"},
                "issuetype": {"name": "Task"},
                "created": "2026-04-01T09:00:00.000+0000",
                "updated": "2026-05-01T12:30:00.000+0000",
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
        issues = fetch_jira_issues_in_project("ENG", limit=5)

    assert issues[0]["key"] == "ENG-9"
    assert issues[0]["created"] == "2026-04-01 09:00:00"
    assert issues[0]["updated"] == "2026-05-01 12:30:00"
    assert issues[0]["description"] == "Details"
    assert "browse/ENG-9" in (issues[0].get("url") or "")


def test_save_workspace_prefs(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _save_creds(monkeypatch, tmp_path)
    cs.save_confluence_credentials(
        {
            "email": "user@example.com",
            "api_token": "secret2",
            "cloud_id": "cid-1",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        }
    )
    cs.save_jira_workspace_prefs(project_key="ENG")
    cs.save_confluence_workspace_prefs(space_key="DOCS")
    jira = cs.get_jira_credentials()
    conf = cs.get_confluence_credentials()
    assert jira.get("workspaces", {}).get("jira_project") == "ENG"
    assert conf.get("workspaces", {}).get("confluence_space") == "DOCS"


def test_fetch_confluence_pages_in_space(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _save_creds(monkeypatch, tmp_path)
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
        "results": [
            {
                "id": "1",
                "title": "Runbook",
                "status": "current",
                "body": {"storage": {"value": "<p>Steps here</p>"}},
                "history": {
                    "createdDate": "2026-04-01T10:00:00.000Z",
                    "createdBy": {"displayName": "Ada"},
                    "lastUpdated": {
                        "when": "2026-05-22T08:45:40.682Z",
                        "friendlyWhen": "May 22, 2026",
                        "by": {"displayName": "Nihit"},
                    },
                },
                "version": {"when": "2026-05-01"},
                "_links": {"webui": "/spaces/DOCS/pages/1"},
            }
        ]
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.return_value = response

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        pages = fetch_confluence_pages_in_space("DOCS", limit=5)

    assert pages[0]["title"] == "Runbook"
    assert pages[0]["updated"] == "May 22, 2026"
    assert pages[0]["updated_by"] == "Nihit"
    assert pages[0]["created"] == "2026-04-01T10:00:00.000Z"
    assert pages[0]["created_by"] == "Ada"
    assert "Steps here" in (pages[0].get("excerpt") or "")


def test_fetch_jira_projects(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _save_creds(monkeypatch, tmp_path)
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "values": [{"key": "ENG", "name": "Engineering", "projectTypeKey": "software"}]
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.return_value = response

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        projects = fetch_jira_projects(limit=5)

    assert projects[0]["key"] == "ENG"
    assert "browse/ENG" in (projects[0].get("url") or "")
