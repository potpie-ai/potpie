"""Unit tests for the outbound GitLab client modules."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from adapters.outbound.cli_auth.gitlab_client import (
    GitLabAuthErrorKind,
    gitlab_auth_headers,
    instance_host,
    normalize_instance_url,
    parse_user_profile,
    verify_instance_access,
    verify_read_api_scope,
)
from adapters.outbound.cli_auth.integration_profile import (
    build_gitlab_integration_record,
    gitlab_account_from_entry,
)
from adapters.outbound.cli_auth.integration_verify import verify_integration_access
from adapters.outbound.cli_auth.provider_config import (
    GITLAB_DEFAULT_INSTANCE,
    gitlab_api_base_url,
    gitlab_pat_page_url,
)
from tests._auth_fakes import FakeAuthHttpClient

pytestmark = pytest.mark.unit


# --- normalize_instance_url ---


def test_normalize_bare_hostname() -> None:
    assert normalize_instance_url("gitlab.corp.com") == "https://gitlab.corp.com"


def test_normalize_with_scheme() -> None:
    assert normalize_instance_url("http://git.local:8080") == "https://git.local:8080"


def test_normalize_with_scheme_allow_http() -> None:
    assert (
        normalize_instance_url("http://git.local:8080", allow_http=True)
        == "http://git.local:8080"
    )


def test_normalize_strips_trailing_slash() -> None:
    assert normalize_instance_url("https://gitlab.com/") == "https://gitlab.com"


def test_normalize_empty_returns_empty() -> None:
    assert normalize_instance_url("") == ""


def test_normalize_preserves_port() -> None:
    assert (
        normalize_instance_url("https://git.corp.com:8443")
        == "https://git.corp.com:8443"
    )


# --- instance_host ---


def test_instance_host_extracts_netloc() -> None:
    assert instance_host("https://gitlab.corp.com") == "gitlab.corp.com"


def test_instance_host_with_port() -> None:
    assert instance_host("https://git.local:8443") == "git.local:8443"


def test_instance_host_empty() -> None:
    assert instance_host("") == ""


# --- gitlab_auth_headers ---


def test_auth_headers_include_private_token() -> None:
    headers = gitlab_auth_headers("glpat-abc123")
    assert headers["PRIVATE-TOKEN"] == "glpat-abc123"
    assert headers["Accept"] == "application/json"


def test_auth_headers_strip_whitespace() -> None:
    headers = gitlab_auth_headers("  glpat-abc123  ")
    assert headers["PRIVATE-TOKEN"] == "glpat-abc123"


# --- provider_config constants ---


def test_gitlab_pat_page_url_default() -> None:
    url = gitlab_pat_page_url(GITLAB_DEFAULT_INSTANCE)
    assert url == "https://gitlab.com/-/user_settings/personal_access_tokens"


def test_gitlab_pat_page_url_custom() -> None:
    url = gitlab_pat_page_url("https://git.corp.com")
    assert url == "https://git.corp.com/-/user_settings/personal_access_tokens"


def test_gitlab_api_base_url_default() -> None:
    assert gitlab_api_base_url(GITLAB_DEFAULT_INSTANCE) == "https://gitlab.com/api/v4"


def test_gitlab_api_base_url_custom() -> None:
    assert (
        gitlab_api_base_url("https://git.corp.com:8443")
        == "https://git.corp.com:8443/api/v4"
    )


# --- parse_user_profile ---


def test_parse_user_profile_full() -> None:
    data = {
        "id": 42,
        "username": "jane",
        "name": "Jane Doe",
        "email": "jane@corp.com",
        "web_url": "https://gitlab.corp.com/jane",
    }
    account = parse_user_profile(data)
    assert account["id"] == "42"
    assert account["username"] == "jane"
    assert account["name"] == "Jane Doe"
    assert account["email"] == "jane@corp.com"
    assert account["web_url"] == "https://gitlab.corp.com/jane"


def test_parse_user_profile_minimal() -> None:
    account = parse_user_profile({"id": 1, "username": "bot"})
    assert account["id"] == "1"
    assert account["username"] == "bot"
    assert "email" not in account
    assert "name" not in account


# --- verify_instance_access ---


def test_verify_instance_access_success() -> None:
    fake = FakeAuthHttpClient(
        [
            httpx.Response(200, json={"id": 42, "username": "jane"}),
        ]
    )
    ok, error_kind, data = verify_instance_access(
        "https://gitlab.com",
        "glpat-test",
        http=fake,
    )
    assert ok is True
    assert error_kind is None
    assert data["username"] == "jane"


def test_verify_instance_access_unauthorized() -> None:
    fake = FakeAuthHttpClient([httpx.Response(401)])
    ok, error_kind, data = verify_instance_access(
        "https://gitlab.com",
        "bad-token",
        http=fake,
    )
    assert ok is False
    assert error_kind == GitLabAuthErrorKind.INVALID_CREDENTIALS
    assert data == {}


def test_verify_instance_access_forbidden() -> None:
    fake = FakeAuthHttpClient([httpx.Response(403)])
    ok, error_kind, _ = verify_instance_access(
        "https://gitlab.com",
        "scoped-token",
        http=fake,
    )
    assert ok is False
    assert error_kind == GitLabAuthErrorKind.INSUFFICIENT_SCOPES


def test_verify_instance_access_unreachable() -> None:
    from adapters.outbound.cli_auth.http import AuthHttpError

    def _raise(*a, **kw):
        raise AuthHttpError("connection refused")

    fake = FakeAuthHttpClient(handler=_raise)
    ok, error_kind, _ = verify_instance_access(
        "https://gitlab.corp.com",
        "tok",
        http=fake,
    )
    assert ok is False
    assert error_kind == GitLabAuthErrorKind.INSTANCE_UNREACHABLE


# --- verify_read_api_scope ---


def test_verify_read_api_scope_success() -> None:
    fake = FakeAuthHttpClient([httpx.Response(200, json=[])])
    ok, error = verify_read_api_scope("https://gitlab.com", "glpat-test", http=fake)
    assert ok is True
    assert error is None


def test_verify_read_api_scope_forbidden() -> None:
    fake = FakeAuthHttpClient([httpx.Response(403)])
    ok, error = verify_read_api_scope("https://gitlab.com", "glpat-test", http=fake)
    assert ok is False
    assert error == GitLabAuthErrorKind.INSUFFICIENT_SCOPES


# --- build_gitlab_integration_record ---


def test_build_gitlab_integration_record() -> None:
    creds: dict[str, Any] = {
        "instance_url": "https://gitlab.corp.com",
        "instance_host": "gitlab.corp.com",
        "stored_at": 1710000000.0,
    }
    account: dict[str, Any] = {"id": "42", "username": "jane", "name": "Jane Doe"}
    record = build_gitlab_integration_record(creds, account=account)
    assert record["provider"] == "gitlab"
    assert record["provider_host"] == "gitlab.corp.com"
    assert record["auth_type"] == "personal_access_token"
    assert record["instance_url"] == "https://gitlab.corp.com"
    assert record["account"]["username"] == "jane"
    assert record["metadata"]["auth_flow"] == "personal_access_token"


def test_build_gitlab_integration_record_preserves_workspaces() -> None:
    creds: dict[str, Any] = {
        "instance_url": "https://gitlab.com",
        "instance_host": "gitlab.com",
        "workspaces": {"default_project": "acme/api"},
    }
    record = build_gitlab_integration_record(creds)
    assert record["workspaces"]["default_project"] == "acme/api"


# --- gitlab_account_from_entry ---


def test_gitlab_account_from_entry_returns_account() -> None:
    entry = {"account": {"username": "jane", "id": "42"}}
    assert gitlab_account_from_entry(entry) == {"username": "jane", "id": "42"}


def test_gitlab_account_from_entry_empty() -> None:
    assert gitlab_account_from_entry({}) == {}


# --- verify_integration_access for gitlab ---


def test_verify_integration_access_gitlab_ok() -> None:
    fake = FakeAuthHttpClient(
        [
            httpx.Response(200, json={"id": 42, "username": "jane", "name": "Jane"}),
            httpx.Response(200, json=[]),
        ]
    )
    ok, msg = verify_integration_access(
        "gitlab",
        {
            "personal_access_token": "glpat-test",
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
        },
        http=fake,
    )
    assert ok is True
    assert "jane" in msg


def test_verify_integration_access_gitlab_no_token() -> None:
    ok, msg = verify_integration_access("gitlab", {})
    assert ok is False
    assert "not authenticated" in msg


def test_verify_integration_access_gitlab_unauthorized() -> None:
    fake = FakeAuthHttpClient([httpx.Response(401)])
    ok, msg = verify_integration_access(
        "gitlab",
        {"personal_access_token": "bad", "instance_url": "https://gitlab.com"},
        http=fake,
    )
    assert ok is False
    assert "invalid" in msg.lower()
