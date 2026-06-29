"""Unit tests for Bitbucket Cloud token verification."""

from __future__ import annotations

import httpx
import pytest

from adapters.outbound.cli_auth.bitbucket_client import (
    BitbucketVerifyResult,
    verify_bitbucket_api_token,
    workspace_slugs_from_list_payload,
)
from adapters.outbound.cli_auth.http import AuthHttpError
from adapters.outbound.cli_auth.provider_config import (
    BITBUCKET_API_BASE,
    BITBUCKET_USER_WORKSPACES_PATH,
)
from tests._auth_fakes import FakeAuthHttpClient


def _user_ok(display_name: str = "Ada Lovelace") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "display_name": display_name,
            "nickname": "ada",
            "username": "ada",
        },
    )


def _workspaces_ok(slug: str = "potpie") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "values": [
                {
                    "workspace": {
                        "slug": slug,
                        "uuid": "{ws-uuid}",
                        "name": "Potpie",
                    }
                }
            ]
        },
    )


def _repos_ok() -> httpx.Response:
    return httpx.Response(200, json={"values": [{"slug": "backend"}]})


def test_workspace_slugs_from_nested_and_legacy_payloads() -> None:
    nested = workspace_slugs_from_list_payload(
        {"values": [{"workspace": {"slug": "team-a"}}]}
    )
    legacy = workspace_slugs_from_list_payload(
        {"values": [{"slug": "team-b", "username": "team-b"}]}
    )
    assert nested == ["team-a"]
    assert legacy == ["team-b"]
    assert workspace_slugs_from_list_payload({"values": "bad"}) == []
    assert workspace_slugs_from_list_payload({"values": [None, "x"]}) == []


def test_verify_rejects_empty_credentials() -> None:
    result = verify_bitbucket_api_token(" ", " ", http=FakeAuthHttpClient())
    assert result == BitbucketVerifyResult(
        ok=False,
        email="",
        error_kind="invalid_credentials",
    )


def test_verify_success_does_not_close_injected_http_client() -> None:
    http = FakeAuthHttpClient(
        responses=[_user_ok(), _workspaces_ok(), _repos_ok()],
    )
    result = verify_bitbucket_api_token(
        "user@example.com",
        "bb-token",
        http=http,
    )
    assert result.ok is True
    assert result.email == "user@example.com"
    assert result.display_name == "Ada Lovelace"
    assert http.closed is False
    assert len(http.calls) == 3
    assert http.calls[1][1].endswith(BITBUCKET_USER_WORKSPACES_PATH)


def test_verify_success_without_workspace_skips_repository_probe() -> None:
    http = FakeAuthHttpClient(
        responses=[_user_ok(), httpx.Response(200, json={"values": []})],
    )
    result = verify_bitbucket_api_token(
        "user@example.com",
        "bb-token",
        http=http,
    )
    assert result.ok is True
    assert len(http.calls) == 2


def test_verify_user_endpoint_401_invalid_credentials() -> None:
    http = FakeAuthHttpClient(
        responses=[
            httpx.Response(
                401,
                json={"error": {"message": "Unauthorized"}},
            )
        ],
    )
    result = verify_bitbucket_api_token("user@example.com", "bad", http=http)
    assert result.ok is False
    assert result.error_kind == "invalid_credentials"


def test_verify_workspaces_403_insufficient_scopes() -> None:
    http = FakeAuthHttpClient(
        responses=[
            _user_ok(),
            httpx.Response(
                403,
                json={"error": {"message": "Insufficient scope"}},
            ),
        ],
    )
    result = verify_bitbucket_api_token("user@example.com", "bb-token", http=http)
    assert result.ok is False
    assert result.error_kind == "insufficient_scopes"


def test_verify_workspaces_400_classified_as_insufficient_scopes() -> None:
    http = FakeAuthHttpClient(
        responses=[
            _user_ok(),
            httpx.Response(
                400,
                json={"error": {"message": "Token missing scopes"}},
            ),
        ],
    )
    result = verify_bitbucket_api_token("user@example.com", "bb-token", http=http)
    assert result.ok is False
    assert result.error_kind == "insufficient_scopes"


def test_verify_change_2770_message_maps_to_unknown() -> None:
    http = FakeAuthHttpClient(
        responses=[
            _user_ok(),
            httpx.Response(
                410,
                json={"error": {"message": "Deprecated endpoint CHANGE-2770"}},
            ),
        ],
    )
    result = verify_bitbucket_api_token("user@example.com", "bb-token", http=http)
    assert result.ok is False
    assert result.error_kind == "unknown"


def test_verify_invalid_user_json_returns_unknown() -> None:
    http = FakeAuthHttpClient(
        responses=[httpx.Response(200, content=b"not-json")],
    )
    result = verify_bitbucket_api_token("user@example.com", "bb-token", http=http)
    assert result.ok is False
    assert result.error_kind == "unknown"


def test_verify_invalid_workspaces_payload_returns_unknown() -> None:
    http = FakeAuthHttpClient(
        responses=[_user_ok(), httpx.Response(200, json=["not", "a", "dict"])],
    )
    result = verify_bitbucket_api_token("user@example.com", "bb-token", http=http)
    assert result.ok is False
    assert result.error_kind == "unknown"


def test_verify_repository_probe_failure() -> None:
    http = FakeAuthHttpClient(
        responses=[
            _user_ok(),
            _workspaces_ok("potpie"),
            httpx.Response(403, json={"error": {"message": "Forbidden"}}),
        ],
    )
    result = verify_bitbucket_api_token("user@example.com", "bb-token", http=http)
    assert result.ok is False
    assert result.error_kind == "insufficient_scopes"
    assert http.calls[2][1] == f"{BITBUCKET_API_BASE}/repositories/potpie"


def test_verify_repository_invalid_json_returns_unknown() -> None:
    http = FakeAuthHttpClient(
        responses=[
            _user_ok(),
            _workspaces_ok("potpie"),
            httpx.Response(200, content=b"not-json"),
        ],
    )
    result = verify_bitbucket_api_token("user@example.com", "bb-token", http=http)
    assert result.ok is False
    assert result.error_kind == "unknown"


def test_verify_auth_http_error_returns_unknown() -> None:
    class BrokenHttp(FakeAuthHttpClient):
        def get(self, url: str, **kwargs: object) -> httpx.Response:
            raise AuthHttpError("network down")

    result = verify_bitbucket_api_token(
        "user@example.com",
        "bb-token",
        http=BrokenHttp(),
    )
    assert result.ok is False
    assert result.error_kind == "unknown"


def test_verify_uses_nickname_when_display_name_missing() -> None:
    http = FakeAuthHttpClient(
        responses=[
            httpx.Response(200, json={"nickname": "nick-only"}),
            httpx.Response(200, json={"values": []}),
        ],
    )
    result = verify_bitbucket_api_token("user@example.com", "bb-token", http=http)
    assert result.ok is True
    assert result.display_name == "nick-only"
