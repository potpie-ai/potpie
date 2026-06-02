"""HTTP error handling for Atlassian CLI auth probes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import typer

from adapters.inbound.cli import atlassian_auth
from adapters.inbound.cli.atlassian_auth import (
    AtlassianAuthErrorKind,
    AtlassianVerifyResult,
    _auth_failure_message,
    _classify_gateway_status,
    _finalize_selected_site,
    _parse_gateway_probe_success,
    fetch_cloud_id_for_site,
    normalize_site_url,
    run_atlassian_api_token_auth,
    site_url_from_subdomain,
    verify_gateway_product,
)


def test_normalize_site_url_and_subdomain() -> None:
    assert normalize_site_url("team.atlassian.net") == "https://team.atlassian.net"
    assert normalize_site_url("") == ""
    assert site_url_from_subdomain("myteam") == "https://myteam.atlassian.net"
    assert site_url_from_subdomain("bad slug!") == ""


def test_verify_gateway_product_empty_cloud_id() -> None:
    result = verify_gateway_product("user@example.com", "tok", "", "jira")
    assert result.ok is False
    assert result.error_kind == AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED


def test_parse_gateway_probe_success_variants() -> None:
    jira_values = _parse_gateway_probe_success(
        {"values": [{"name": "Project A", "key": "PA"}]},
        product="jira",
        path="/rest/api/3/project/search",
    )
    assert jira_values == "Project A"

    conf_results = _parse_gateway_probe_success(
        {"results": [{"name": "Docs", "key": "DOCS"}]},
        product="confluence",
        path="/wiki/rest/api/space",
    )
    assert conf_results == "Docs"

    conf_user = _parse_gateway_probe_success(
        {"displayName": "Wiki User"},
        product="confluence",
        path="/wiki/rest/api/user/current",
    )
    assert conf_user == "Wiki User"

    assert atlassian_auth._parse_profile_name(None, product="jira") == ""
    assert atlassian_auth._parse_profile_name({"username": "wiki"}, product="confluence") == "wiki"


def test_verify_gateway_product_confluence_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.content = b'{"displayName":"Wiki"}'
    response.json.return_value = {"displayName": "Wiki"}
    with patch("adapters.inbound.cli.atlassian_auth.httpx.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.get.return_value = response

        result = verify_gateway_product(
            "user@example.com",
            "secret-token",
            "cloud-id-1",
            "confluence",
        )

    assert result.ok is True
    assert result.display_name == "Wiki"


def test_classify_gateway_status() -> None:
    assert _classify_gateway_status(401) == AtlassianAuthErrorKind.INVALID_CREDENTIALS
    assert _classify_gateway_status(403) == AtlassianAuthErrorKind.INSUFFICIENT_SCOPES
    assert _classify_gateway_status(404) == AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED
    assert _classify_gateway_status(500) == AtlassianAuthErrorKind.UNKNOWN


def test_parse_gateway_probe_success_jira_myself() -> None:
    name = _parse_gateway_probe_success(
        {"displayName": "Ada"},
        product="jira",
        path="/rest/api/3/myself",
    )
    assert name == "Ada"


def test_verify_gateway_product_insufficient_scopes_on_403() -> None:
    response = MagicMock()
    response.status_code = 403
    response.content = b""
    with patch("adapters.inbound.cli.atlassian_auth.httpx.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.get.return_value = response

        result = verify_gateway_product(
            "user@example.com",
            "secret-token",
            "cloud-id-1",
            "jira",
        )

    assert result.ok is False
    assert result.error_kind == AtlassianAuthErrorKind.INSUFFICIENT_SCOPES


def test_verify_gateway_product_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.content = b'{"displayName":"Ada"}'
    response.json.return_value = {"displayName": "Ada"}
    with patch("adapters.inbound.cli.atlassian_auth.httpx.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.get.return_value = response

        result = verify_gateway_product(
            "user@example.com",
            "secret-token",
            "cloud-id-1",
            "jira",
        )

    assert result.ok is True
    assert result.display_name == "Ada"


def test_fetch_cloud_id_for_site_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"cloudId": "cloud-xyz"}
    with patch("adapters.inbound.cli.atlassian_auth.httpx.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.get.return_value = response

        assert fetch_cloud_id_for_site("https://team.atlassian.net") == "cloud-xyz"


def test_auth_failure_message_other_error_kinds() -> None:
    site_msg = _auth_failure_message(
        "jira",
        error_kind=AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED,
    )
    assert "Could not resolve cloud ID" in site_msg
    denied_msg = _auth_failure_message(
        "confluence",
        error_kind=AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED,
    )
    assert "cannot access Confluence" in denied_msg
    default_msg = _auth_failure_message("jira", error_kind=None)
    assert "Check the site subdomain" in default_msg


def test_finalize_selected_site_success() -> None:
    site = {
        "site_url": "https://team.atlassian.net",
        "site_name": "Team",
        "cloud_id": "cloud-1",
    }
    with patch(
        "adapters.inbound.cli.atlassian_auth.verify_gateway_product",
        return_value=AtlassianVerifyResult(ok=True, display_name="Ada"),
    ):
        finalized, err = _finalize_selected_site(
            "user@example.com",
            "token",
            site,
            "jira",
        )
    assert err is None
    assert finalized is not None
    assert finalized["cloud_id"]


def test_finalize_selected_site_gateway_failure() -> None:
    site = {"site_url": "https://team.atlassian.net", "cloud_id": "c1"}
    with patch(
        "adapters.inbound.cli.atlassian_auth.verify_gateway_product",
        return_value=AtlassianVerifyResult(
            ok=False,
            error_kind=AtlassianAuthErrorKind.INVALID_CREDENTIALS,
        ),
    ):
        finalized, err = _finalize_selected_site(
            "user@example.com",
            "token",
            site,
            "jira",
        )
    assert finalized is None
    assert err == AtlassianAuthErrorKind.INVALID_CREDENTIALS


def test_auth_failure_message_invalid_credentials() -> None:
    msg = _auth_failure_message(
        "jira",
        error_kind=AtlassianAuthErrorKind.INVALID_CREDENTIALS,
    )
    assert "Invalid email or API token" in msg


def test_auth_failure_message_insufficient_scopes_per_product() -> None:
    jira_msg = _auth_failure_message(
        "jira",
        error_kind=AtlassianAuthErrorKind.INSUFFICIENT_SCOPES,
    )
    confluence_msg = _auth_failure_message(
        "confluence",
        error_kind=AtlassianAuthErrorKind.INSUFFICIENT_SCOPES,
    )
    jira_lines = jira_msg.splitlines()
    confluence_lines = confluence_msg.splitlines()
    assert "read:jira-work at minimum" in jira_lines[1]
    assert "read:confluence-content at minimum" in confluence_lines[1]


def test_fetch_cloud_id_for_site_returns_empty_on_http_error() -> None:
    with patch("adapters.inbound.cli.atlassian_auth.httpx.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.get.side_effect = httpx.ConnectError("connection refused")

        assert fetch_cloud_id_for_site("https://team.atlassian.net") == ""


def test_verify_gateway_product_returns_unknown_on_http_error() -> None:
    with patch("adapters.inbound.cli.atlassian_auth.httpx.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.get.side_effect = httpx.ConnectError("connection refused")

        result = verify_gateway_product(
            "user@example.com",
            "secret-token",
            "cloud-id-1",
            "jira",
        )

    assert result.ok is False
    assert result.error_kind == AtlassianAuthErrorKind.UNKNOWN
    assert result.http_status is None


def test_run_atlassian_auth_rejects_non_tty_without_supplied_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atlassian_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        atlassian_auth,
        "_get_product_credentials",
        lambda _product: {},
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        atlassian_auth,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    def _fail_prompt(*args: object, **kwargs: object) -> str:
        raise AssertionError("typer.prompt must not run without a TTY")

    monkeypatch.setattr(atlassian_auth.typer, "prompt", _fail_prompt)

    with pytest.raises(typer.Exit):
        run_atlassian_api_token_auth("jira", force=True, as_json=True)

    assert captured
    assert "requires a terminal" in captured[0][0].lower()


def test_run_atlassian_auth_success_with_supplied_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(atlassian_auth, "credentials_path", lambda: tmp_path / "creds.json")
    monkeypatch.setattr(atlassian_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        atlassian_auth,
        "_get_product_credentials",
        lambda _product: {},
    )
    monkeypatch.setattr(
        atlassian_auth,
        "_save_product_credentials",
        lambda product, payload: None,
    )
    site = {
        "cloud_id": "cloud-1",
        "site_url": "https://myteam.atlassian.net",
        "site_name": "myteam",
    }
    monkeypatch.setattr(
        atlassian_auth,
        "_resolve_site_from_subdomain",
        lambda _sub: (site, None),
    )
    monkeypatch.setattr(
        atlassian_auth,
        "_finalize_selected_site",
        lambda *_args: (site, None),
    )
    printed: list[dict] = []
    monkeypatch.setattr(
        atlassian_auth,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )

    run_atlassian_api_token_auth(
        "jira",
        force=True,
        as_json=True,
        email="user@example.com",
        api_token="secret-token",
        site_subdomain="myteam",
    )

    assert printed
    assert printed[-1].get("ok") is True
    assert printed[-1].get("provider") == "jira"


def test_run_atlassian_auth_already_connected_skips_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atlassian_auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        atlassian_auth,
        "_get_product_credentials",
        lambda _product: {
            "api_token": "tok",
            "site_url": "https://team.atlassian.net",
        },
    )
    printed: list[dict] = []
    monkeypatch.setattr(
        atlassian_auth,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )

    run_atlassian_api_token_auth("jira", as_json=True)

    assert printed
    assert printed[-1].get("already_connected") is True


def test_finalize_selected_site_fetches_cloud_id() -> None:
    site = {"site_url": "https://team.atlassian.net", "site_name": "Team"}
    with (
        patch(
            "adapters.inbound.cli.atlassian_auth.fetch_cloud_id_for_site",
            return_value="cloud-fetched",
        ),
        patch(
            "adapters.inbound.cli.atlassian_auth.verify_gateway_product",
            return_value=AtlassianVerifyResult(ok=True, display_name="Ada"),
        ),
    ):
        finalized, err = _finalize_selected_site(
            "user@example.com",
            "token",
            site,
            "jira",
        )
    assert err is None
    assert finalized is not None
    assert finalized["cloud_id"] == "cloud-fetched"


def test_verify_site_with_api_token_success() -> None:
    from adapters.inbound.cli.atlassian_auth import verify_site_with_api_token

    with (
        patch(
            "adapters.inbound.cli.atlassian_auth.fetch_cloud_id_for_site",
            return_value="cloud-1",
        ),
        patch(
            "adapters.inbound.cli.atlassian_auth.verify_gateway_product",
            return_value=AtlassianVerifyResult(ok=True, display_name="Team Site"),
        ),
    ):
        site = verify_site_with_api_token(
            "user@example.com",
            "token",
            "https://team.atlassian.net",
            "jira",
        )
    assert site is not None
    assert site["cloud_id"] == "cloud-1"


def test_discover_sites_skips_candidates_without_cloud_id() -> None:
    with (
        patch(
            "adapters.inbound.cli.atlassian_auth.collect_site_candidates",
            return_value=[{"site_url": "https://team.atlassian.net", "cloud_id": ""}],
        ),
        patch(
            "adapters.inbound.cli.atlassian_auth.verify_gateway_product",
        ) as verify,
    ):
        found = atlassian_auth.discover_sites_with_api_token(
            "user@example.com",
            "token",
            "jira",
        )
    verify.assert_not_called()
    assert found == []


def test_resolve_site_from_subdomain_failures() -> None:
    site, err = atlassian_auth._resolve_site_from_subdomain("")
    assert site is None
    assert err == AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED

    site2, err2 = atlassian_auth._resolve_site_from_subdomain("bad slug!")
    assert site2 is None
    assert err2 == AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED

    with patch(
        "adapters.inbound.cli.atlassian_auth.fetch_cloud_id_for_site",
        return_value="",
    ):
        site3, err3 = atlassian_auth._resolve_site_from_subdomain("myteam")
    assert site3 is None
    assert err3 == AtlassianAuthErrorKind.SITE_DISCOVERY_FAILED


def test_finalize_atlassian_site_unscoped() -> None:
    site = {
        "site_url": "https://team.atlassian.net",
        "site_name": "Team",
        "cloud_id": "cloud-1",
    }
    with (
        patch(
            "adapters.inbound.cli.atlassian_auth._finalize_selected_site",
            return_value=(site, None),
        ),
        patch(
            "adapters.inbound.cli.atlassian_auth.verify_gateway_product",
            return_value=AtlassianVerifyResult(ok=True, display_name="Wiki"),
        ),
    ):
        finalized, err = atlassian_auth._finalize_atlassian_site_unscoped(
            "user@example.com",
            "token",
            site,
        )
    assert err is None
    assert finalized == site


def test_run_atlassian_auth_supplied_credentials_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atlassian_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(atlassian_auth, "_get_product_credentials", lambda _p: {})
    site = {
        "site_url": "https://team.atlassian.net",
        "site_name": "team",
        "cloud_id": "cloud-1",
    }
    monkeypatch.setattr(
        atlassian_auth,
        "_resolve_site_from_subdomain",
        lambda _sub: (site, None),
    )
    monkeypatch.setattr(
        atlassian_auth,
        "_finalize_selected_site",
        lambda *_args: (site, None),
    )
    saved: list[str] = []
    monkeypatch.setattr(
        atlassian_auth,
        "_save_product_credentials",
        lambda product, _payload: saved.append(product),
    )
    printed: list[dict] = []
    monkeypatch.setattr(
        atlassian_auth,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )

    run_atlassian_api_token_auth(
        "confluence",
        force=True,
        as_json=True,
        email="user@example.com",
        api_token="secret-token",
        site_subdomain="team",
    )

    assert saved == ["confluence"]
    assert printed[-1].get("ok") is True


def test_run_atlassian_auth_emits_site_discovery_error_on_tenant_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atlassian_auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(atlassian_auth, "webbrowser", MagicMock())
    monkeypatch.setattr(
        atlassian_auth,
        "_get_product_credentials",
        lambda _product: {},
    )

    prompts = iter(["user@example.com", "api-token-secret", "myteam"])
    monkeypatch.setattr(
        atlassian_auth.typer,
        "prompt",
        lambda *args, **kwargs: next(prompts),
    )

    captured: list[tuple[str, str]] = []

    def _capture_error(title: str, message: str, **kwargs: object) -> None:
        captured.append((title, message))

    monkeypatch.setattr(atlassian_auth, "emit_error", _capture_error)

    with patch("adapters.inbound.cli.atlassian_auth.httpx.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client
        client.get.side_effect = httpx.ConnectError("connection refused")

        with pytest.raises(typer.Exit):
            run_atlassian_api_token_auth("jira", force=True)

    assert captured
    title, message = captured[0]
    assert "authentication failed" in title.lower()
    assert "Could not resolve cloud ID for the site" in message
