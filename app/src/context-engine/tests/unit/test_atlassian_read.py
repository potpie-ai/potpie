"""Tests for Atlassian CLI read commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.inbound.cli import credentials_store as cs
import httpx

from adapters.inbound.cli.atlassian_read import (
    AtlassianReadError,
    _cloud_id_from_credentials,
    _get_json,
    _post_json,
    _site_url_from_credentials,
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


def test_get_json_raises_atlassian_read_error_on_http_error() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.side_effect = httpx.ConnectError("connection refused")

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        with pytest.raises(AtlassianReadError, match="jira GET failed") as exc_info:
            _get_json(
                email="user@example.com",
                api_token="secret",
                product="jira",
                cloud_id="cid-1",
                site_url="https://team.atlassian.net",
                path="/rest/api/3/myself",
                site_first=True,
            )

    assert "/rest/api/3/myself" in str(exc_info.value)
    assert "team.atlassian.net" in str(exc_info.value)


def test_post_json_raises_atlassian_read_error_on_http_error() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.side_effect = httpx.ConnectError("connection refused")

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        with pytest.raises(AtlassianReadError, match="jira POST failed") as exc_info:
            _post_json(
                email="user@example.com",
                api_token="secret",
                product="jira",
                cloud_id="cid-1",
                site_url="https://team.atlassian.net",
                path="/rest/api/3/search",
                body={"jql": "project = ENG"},
            )

    assert "/rest/api/3/search" in str(exc_info.value)


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
        "results": [
            {
                "key": "ENG",
                "name": "Engineering",
                "type": "global",
                "_links": {"webui": "/spaces/ENG"},
            }
        ]
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.return_value = response

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        spaces = fetch_confluence_spaces_sample(limit=5)

    assert len(spaces) == 1
    assert spaces[0]["key"] == "ENG"
    assert "spaces/ENG" in (spaces[0].get("url") or "")


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


def test_post_json_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"issues": []}
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        data = _post_json(
            email="user@example.com",
            api_token="secret",
            product="jira",
            cloud_id="cid-1",
            site_url="https://team.atlassian.net",
            path="/rest/api/3/search",
            body={"jql": "order by created DESC"},
        )

    assert data == {"issues": []}


def test_get_json_confluence_wiki_path() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"results": []}
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.return_value = response

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        data = _get_json(
            email="user@example.com",
            api_token="secret",
            product="confluence",
            cloud_id="cid-1",
            site_url="https://team.atlassian.net",
            path="/rest/api/content",
        )

    assert "results" in data
    assert client.get.call_count >= 1


def test_get_json_http_error_status() -> None:
    response = MagicMock()
    response.status_code = 500
    response.text = "server error"
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.return_value = response

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        with pytest.raises(AtlassianReadError, match="HTTP 500"):
            _get_json(
                email="user@example.com",
                api_token="secret",
                product="jira",
                cloud_id="cid-1",
                site_url="https://team.atlassian.net",
                path="/rest/api/3/myself",
            )


def test_get_json_returns_list_payload_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = [{"id": "1"}]
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.return_value = response

    with patch("adapters.inbound.cli.atlassian_read.httpx.Client", return_value=client):
        data = _get_json(
            email="user@example.com",
            api_token="secret",
            product="jira",
            cloud_id="cid-1",
            site_url="https://team.atlassian.net",
            path="/rest/api/3/myself",
        )

    assert data == {"data": [{"id": "1"}]}


def test_prompt_workspace_interactive_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import adapters.inbound.cli.atlassian_read as atlassian_read

    prompts = iter(["2"])
    monkeypatch.setattr(atlassian_read.typer, "prompt", lambda *args, **kwargs: next(prompts))
    printed: list[str] = []
    monkeypatch.setattr(
        atlassian_read,
        "print_plain_line",
        lambda message, **kwargs: printed.append(message),
    )

    picked = atlassian_read._prompt_workspace(
        [{"key": "ENG", "name": "Engineering"}, {"key": "OPS", "name": "Ops"}],
        label="Jira project",
    )

    assert picked["key"] == "OPS"


def test_fetch_jira_issues_sample_uses_saved_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _mock_keychain(monkeypatch)
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "c1",
        }
    )
    cs.save_jira_workspace_prefs(project_key="ENG")
    monkeypatch.setattr(
        "adapters.inbound.cli.atlassian_read.fetch_jira_issues_in_project",
        lambda key, limit: [{"key": f"{key}-1"}],
    )

    rows = fetch_jira_issues_sample(limit=5)
    assert rows == [{"key": "ENG-1"}]


def test_credential_helpers_require_cloud_id_and_site_url() -> None:
    with pytest.raises(AtlassianReadError, match="Missing cloud_id"):
        _cloud_id_from_credentials({})
    with pytest.raises(AtlassianReadError, match="Missing site_url"):
        _site_url_from_credentials({"cloud_id": "c1"})
