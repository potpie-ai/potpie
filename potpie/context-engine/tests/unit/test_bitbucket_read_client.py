"""Unit tests for Bitbucket Cloud read/list helpers."""

from __future__ import annotations

import httpx
import pytest

from adapters.outbound.cli_auth.bitbucket_read_client import (
    BitbucketReadError,
    fetch_bitbucket_pull_requests,
    fetch_bitbucket_repositories,
    fetch_bitbucket_workspaces,
    load_bitbucket_read_credentials,
)
from adapters.outbound.cli_auth.http import AuthHttpError
from adapters.outbound.cli_auth.provider_config import BITBUCKET_USER_WORKSPACES_PATH
from tests._auth_fakes import FakeAuthHttpClient

CREDS = {"email": "user@example.com", "api_token": "bb-token"}


def test_load_bitbucket_read_credentials_requires_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "adapters.outbound.cli_auth.bitbucket_read_client.get_bitbucket_credentials",
        lambda: {},
    )
    with pytest.raises(BitbucketReadError, match="not connected"):
        load_bitbucket_read_credentials()


def test_fetch_workspaces_parses_nested_workspace_rows() -> None:
    http = FakeAuthHttpClient(
        responses=[
            httpx.Response(
                200,
                json={
                    "values": [
                        {
                            "workspace": {
                                "uuid": "{ws}",
                                "slug": "potpie",
                                "name": "Potpie",
                            }
                        }
                    ]
                },
            )
        ],
    )
    rows = fetch_bitbucket_workspaces(credentials=CREDS, http=http)
    assert rows == [
        {
            "id": "{ws}",
            "key": "potpie",
            "name": "Potpie",
            "type": "workspace",
        }
    ]


def test_fetch_workspaces_falls_back_to_slug_extractor() -> None:
    http = FakeAuthHttpClient(
        responses=[httpx.Response(200, json={"values": [{"slug": "legacy-ws"}]})],
    )
    rows = fetch_bitbucket_workspaces(credentials=CREDS, http=http)
    assert rows == [
        {"id": "", "key": "legacy-ws", "name": "legacy-ws", "type": "workspace"}
    ]


def test_fetch_repositories_maps_repository_fields() -> None:
    http = FakeAuthHttpClient(
        responses=[
            httpx.Response(
                200,
                json={
                    "values": [
                        {
                            "uuid": "{repo}",
                            "slug": "backend",
                            "name": "Backend",
                            "links": {"html": {"href": "https://bitbucket.org/potpie/backend"}},
                        }
                    ]
                },
            )
        ],
    )
    rows = fetch_bitbucket_repositories("potpie", credentials=CREDS, http=http)
    assert rows == [
        {
            "id": "{repo}",
            "key": "backend",
            "name": "Backend",
            "type": "repository",
            "url": "https://bitbucket.org/potpie/backend",
        }
    ]


def test_fetch_repositories_requires_workspace_key() -> None:
    with pytest.raises(BitbucketReadError, match="workspace key"):
        fetch_bitbucket_repositories("  ", credentials=CREDS, http=FakeAuthHttpClient())


def test_fetch_pull_requests_maps_pr_fields() -> None:
    http = FakeAuthHttpClient(
        responses=[
            httpx.Response(
                200,
                json={
                    "values": [
                        {
                            "id": 42,
                            "title": "Fix auth",
                            "state": "OPEN",
                            "updated_on": "2026-01-01T00:00:00Z",
                            "author": {"display_name": "Ada"},
                            "links": {
                                "html": {
                                    "href": "https://bitbucket.org/potpie/backend/pull-requests/42"
                                }
                            },
                        }
                    ]
                },
            )
        ],
    )
    rows = fetch_bitbucket_pull_requests(
        "potpie",
        "backend",
        credentials=CREDS,
        http=http,
    )
    assert rows == [
        {
            "id": 42,
            "title": "Fix auth",
            "state": "OPEN",
            "author": "Ada",
            "updated": "2026-01-01T00:00:00Z",
            "url": "https://bitbucket.org/potpie/backend/pull-requests/42",
        }
    ]


def test_fetch_pull_requests_requires_workspace_and_repo() -> None:
    with pytest.raises(BitbucketReadError, match="workspace and repository"):
        fetch_bitbucket_pull_requests("potpie", "", credentials=CREDS, http=FakeAuthHttpClient())


def test_get_json_maps_http_status_errors() -> None:
    cases = [
        (401, "authentication failed"),
        (403, "missing required read scopes"),
        (500, "status 500"),
    ]
    for status_code, message in cases:
        http = FakeAuthHttpClient(responses=[httpx.Response(status_code)])
        with pytest.raises(BitbucketReadError, match=message):
            fetch_bitbucket_workspaces(credentials=CREDS, http=http)


def test_get_json_invalid_json_raises() -> None:
    http = FakeAuthHttpClient(responses=[httpx.Response(200, content=b"not-json")])
    with pytest.raises(BitbucketReadError, match="invalid JSON"):
        fetch_bitbucket_workspaces(credentials=CREDS, http=http)


def test_get_json_unexpected_payload_raises() -> None:
    http = FakeAuthHttpClient(responses=[httpx.Response(200, json=["not", "dict"])])
    with pytest.raises(BitbucketReadError, match="unexpected response"):
        fetch_bitbucket_workspaces(credentials=CREDS, http=http)


def test_get_json_auth_http_error_wraps_failure() -> None:
    class BrokenHttp(FakeAuthHttpClient):
        def get(self, url: str, **kwargs: object) -> httpx.Response:
            raise AuthHttpError("network down")

    with pytest.raises(BitbucketReadError, match="request failed"):
        fetch_bitbucket_workspaces(credentials=CREDS, http=BrokenHttp())


def test_fetch_workspaces_uses_injected_http_without_closing() -> None:
    http = FakeAuthHttpClient(responses=[httpx.Response(200, json={"values": []})])
    fetch_bitbucket_workspaces(credentials=CREDS, http=http)
    assert http.closed is False
    assert http.calls[0][1].endswith(BITBUCKET_USER_WORKSPACES_PATH)
