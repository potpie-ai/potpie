"""Unit tests for Atlassian API token CLI modules."""

from __future__ import annotations

from __future__ import annotations
from unittest.mock import MagicMock, patch

from adapters.outbound.cli_auth.http import AuthHttpError
import pytest
import typer
from adapters.inbound.cli.auth import atlassian_auth
from adapters.outbound.cli_auth.atlassian_client import (
    AtlassianAuthErrorKind,
    AtlassianVerifyResult,
    _classify_gateway_status,
    _finalize_selected_site,
    _parse_gateway_probe_success,
    fetch_cloud_id_for_site,
    normalize_site_url,
    site_url_from_subdomain,
    verify_gateway_product,
)
from adapters.inbound.cli.auth.atlassian_auth import (
    _auth_failure_message,
    run_atlassian_api_token_auth,
)
from adapters.outbound.cli_auth.atlassian_client import (
    _fetch_accessible_resources,
    _parse_accessible_resources,
    discover_sites_with_api_token,
    fetch_accessible_resources,
)
from adapters.inbound.cli.auth.atlassian_read import _auth_header_variants
from adapters.outbound.cli_auth import credentials_store as cs
from adapters.inbound.cli.auth.atlassian_read import (
    AtlassianReadError,
    _cloud_id_from_credentials,
    _get_json,
    _post_json,
    _site_url_from_credentials,
    fetch_confluence_spaces_sample,
    fetch_jira_issues_sample,
)
from adapters.inbound.cli.auth.atlassian_read import (
    fetch_confluence_pages_in_space,
    fetch_jira_issues_in_project,
    fetch_jira_projects,
    run_confluence_use_flow,
    run_jira_use_flow,
)
# --- test_atlassian_auth_http.py ---


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
    assert (
        atlassian_auth._parse_profile_name({"username": "wiki"}, product="confluence")
        == "wiki"
    )


def test_verify_gateway_product_confluence_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.content = b'{"displayName":"Wiki"}'
    response.json.return_value = {"displayName": "Wiki"}
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
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
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
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
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get.return_value = response

        result = verify_gateway_product(
            "user@example.com",
            "secret-token",
            "cloud-id-1",
            "jira",
        )

    assert result.ok is True
    assert result.display_name == "Ada"
    assert result.succeeded_scheme == "basic"


def test_verify_gateway_product_bearer_after_basic_401() -> None:
    basic = MagicMock()
    basic.status_code = 401
    bearer = MagicMock()
    bearer.status_code = 200
    bearer.content = b'{"displayName":"Bearer User"}'
    bearer.json.return_value = {"displayName": "Bearer User"}
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get.side_effect = [basic, bearer]

        result = verify_gateway_product(
            "user@example.com",
            "secret-token",
            "cloud-id-1",
            "jira",
        )

    assert result.ok is True
    assert result.succeeded_scheme == "bearer"
    assert result.display_name == "Bearer User"


def test_fetch_cloud_id_for_site_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"cloudId": "cloud-xyz"}
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
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
        "adapters.outbound.cli_auth.atlassian_client.verify_gateway_product",
        return_value=AtlassianVerifyResult(
            ok=True,
            display_name="Ada",
            succeeded_scheme="bearer",
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
    assert finalized["cloud_id"]
    assert finalized["token_style"] == "bearer"


def test_finalize_selected_site_gateway_failure() -> None:
    site = {"site_url": "https://team.atlassian.net", "cloud_id": "c1"}
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.verify_gateway_product",
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
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get.side_effect = AuthHttpError("connection refused")

        assert fetch_cloud_id_for_site("https://team.atlassian.net") == ""


def test_verify_gateway_product_returns_unknown_on_http_error() -> None:
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get.side_effect = AuthHttpError("connection refused")

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
    monkeypatch.setattr(
        atlassian_auth, "credentials_path", lambda: tmp_path / "creds.json"
    )
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
            "adapters.outbound.cli_auth.atlassian_client.fetch_cloud_id_for_site",
            return_value="cloud-fetched",
        ),
        patch(
            "adapters.outbound.cli_auth.atlassian_client.verify_gateway_product",
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
    from adapters.inbound.cli.auth.atlassian_auth import verify_site_with_api_token

    with (
        patch(
            "adapters.outbound.cli_auth.atlassian_client.fetch_cloud_id_for_site",
            return_value="cloud-1",
        ),
        patch(
            "adapters.outbound.cli_auth.atlassian_client.verify_gateway_product",
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
            "adapters.outbound.cli_auth.atlassian_client.collect_site_candidates",
            return_value=[{"site_url": "https://team.atlassian.net", "cloud_id": ""}],
        ),
        patch(
            "adapters.outbound.cli_auth.atlassian_client.verify_gateway_product",
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
        "adapters.outbound.cli_auth.atlassian_client.fetch_cloud_id_for_site",
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
            "adapters.outbound.cli_auth.atlassian_client._finalize_selected_site",
            return_value=(site, None),
        ),
        patch(
            "adapters.outbound.cli_auth.atlassian_client.verify_gateway_product",
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
        "token_style": "bearer",
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
    saved: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        atlassian_auth,
        "_save_product_credentials",
        lambda product, payload: saved.append((product, payload)),
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

    assert saved and saved[0][0] == "confluence"
    assert saved[0][1]["token_style"] == "bearer"
    assert printed[-1].get("token_style") == "bearer"
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

    prompts = iter(["api-token-secret", "user@example.com", "myteam"])
    monkeypatch.setattr(
        atlassian_auth.typer,
        "prompt",
        lambda *args, **kwargs: next(prompts),
    )
    monkeypatch.setattr(
        atlassian_auth.typer,
        "confirm",
        lambda *args, **kwargs: True,
    )

    captured: list[tuple[str, str]] = []

    def _capture_error(title: str, message: str, **kwargs: object) -> None:
        captured.append((title, message))

    monkeypatch.setattr(atlassian_auth, "emit_error", _capture_error)

    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get.side_effect = AuthHttpError("connection refused")

        with pytest.raises(typer.Exit):
            run_atlassian_api_token_auth("jira", force=True)

    assert captured
    title, message = captured[0]
    assert "authentication failed" in title.lower()
    assert "Could not resolve cloud ID for the site" in message


@pytest.mark.parametrize("product", ["jira", "confluence"])
def test_run_atlassian_auth_opens_token_page_after_enter(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    product: str,
) -> None:
    monkeypatch.setattr(atlassian_auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        atlassian_auth,
        "_get_product_credentials",
        lambda _product: {},
    )
    site = {
        "cloud_id": "cloud-1",
        "site_url": "https://team.atlassian.net",
        "site_name": "Team",
    }
    monkeypatch.setattr(
        atlassian_auth,
        "_prompt_and_resolve_site",
        lambda: (site, None),
    )
    monkeypatch.setattr(
        atlassian_auth,
        "_finalize_selected_site",
        lambda *_args: (site, None),
    )
    monkeypatch.setattr(
        atlassian_auth,
        "_save_product_credentials",
        lambda *_args: None,
    )
    opened_urls: list[str] = []
    monkeypatch.setattr(
        atlassian_auth.webbrowser,
        "open",
        lambda url, **_kwargs: opened_urls.append(url) or True,
    )
    confirms: list[str] = []

    def _confirm(prompt: str, **kwargs: object) -> bool:
        confirms.append(prompt)
        return True

    monkeypatch.setattr(atlassian_auth.typer, "confirm", _confirm)
    prompts: list[str] = []
    prompt_values = iter(["api-token-secret", "user@example.com"])

    def _prompt(label: str, **_kwargs: object) -> str:
        prompts.append(label)
        return next(prompt_values)

    monkeypatch.setattr(atlassian_auth.typer, "prompt", _prompt)

    run_atlassian_api_token_auth(product, force=True)

    out = capsys.readouterr().out
    assert "Jira login — Atlassian API token" in out or "Confluence login — Atlassian API token" in out
    assert "  • One token works for both Jira and Confluence" in out
    assert "Press Enter to open the page" in out
    assert "Opening id.atlassian.com ..." in out
    assert "Opening Atlassian in 10 seconds..." not in out
    assert opened_urls == [atlassian_auth.ATLASSIAN_API_TOKEN_PAGE]
    assert confirms == ["Press Enter to continue"]
    assert prompts == ["Enter your API token", "Enter your Atlassian email"]


def test_run_atlassian_auth_target_accepts_either_product(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(atlassian_auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        atlassian_auth,
        "_get_product_credentials",
        lambda _product: {},
    )
    site = {
        "cloud_id": "cloud-1",
        "site_url": "https://team.atlassian.net",
        "site_name": "Team",
    }
    monkeypatch.setattr(
        atlassian_auth,
        "_prompt_and_resolve_site",
        lambda: (site, None),
    )

    finalize_calls: list[str] = []

    def _finalize(
        _email: str,
        _api_token: str,
        selected_site: dict[str, object],
        product: str,
    ) -> tuple[dict[str, object] | None, AtlassianAuthErrorKind | None]:
        finalize_calls.append(product)
        if product == "confluence":
            return {**selected_site, "token_style": "classic"}, None
        return None, AtlassianAuthErrorKind.PRODUCT_ACCESS_DENIED

    saved: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(atlassian_auth, "_finalize_selected_site", _finalize)
    monkeypatch.setattr(
        atlassian_auth,
        "_save_product_credentials",
        lambda product, payload: saved.append((product, payload)),
    )
    monkeypatch.setattr(atlassian_auth.typer, "confirm", lambda *_a, **_k: False)
    prompt_values = iter(["api-token-secret", "user@example.com"])
    monkeypatch.setattr(
        atlassian_auth.typer,
        "prompt",
        lambda *_a, **_k: next(prompt_values),
    )

    run_atlassian_api_token_auth("atlassian", force=True)

    out = capsys.readouterr().out
    assert "Atlassian login — Atlassian API token" in out
    assert finalize_calls == ["jira", "confluence"]
    assert saved[0][0] == "atlassian"
    assert saved[0][1]["api_token"] == "api-token-secret"


@pytest.mark.parametrize("product", ["jira", "confluence"])
def test_run_atlassian_auth_skips_browser_when_confirm_declined(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    product: str,
) -> None:
    monkeypatch.setattr(atlassian_auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        atlassian_auth,
        "_get_product_credentials",
        lambda _product: {},
    )
    site = {
        "cloud_id": "cloud-1",
        "site_url": "https://team.atlassian.net",
        "site_name": "Team",
    }
    monkeypatch.setattr(
        atlassian_auth,
        "_prompt_and_resolve_site",
        lambda: (site, None),
    )
    monkeypatch.setattr(
        atlassian_auth,
        "_finalize_selected_site",
        lambda *_args: (site, None),
    )
    monkeypatch.setattr(
        atlassian_auth,
        "_save_product_credentials",
        lambda *_args: None,
    )
    opened_urls: list[str] = []
    monkeypatch.setattr(
        atlassian_auth.webbrowser,
        "open",
        lambda url, **_kwargs: opened_urls.append(url) or True,
    )
    monkeypatch.setattr(
        atlassian_auth.typer,
        "confirm",
        lambda *_args, **_kwargs: False,
    )
    prompt_values = iter(["api-token-secret", "user@example.com"])
    prompts: list[str] = []

    def _prompt(label: str, **_kwargs: object) -> str:
        prompts.append(label)
        return next(prompt_values)

    monkeypatch.setattr(atlassian_auth.typer, "prompt", _prompt)

    run_atlassian_api_token_auth(product, force=True)

    out = capsys.readouterr().out
    assert "Opening id.atlassian.com ..." not in out
    assert opened_urls == []
    assert prompts == ["Enter your API token", "Enter your Atlassian email"]


# --- test_atlassian_auth_discovery.py ---


def test_parse_accessible_resources() -> None:
    data = [
        {"id": "cloud-1", "url": "https://team.atlassian.net", "name": "Team"},
        {"invalid": True},
    ]
    sites = _parse_accessible_resources(data)
    assert len(sites) == 1
    assert sites[0]["cloud_id"] == "cloud-1"
    assert sites[0]["site_url"] == "https://team.atlassian.net"


def test_fetch_accessible_resources_bearer_fallback() -> None:
    basic = MagicMock()
    basic.status_code = 401
    bearer = MagicMock()
    bearer.status_code = 200
    bearer.json.return_value = [
        {"id": "c2", "url": "https://other.atlassian.net", "name": "Other"},
    ]
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        client.get.side_effect = [basic, bearer]
        sites = _fetch_accessible_resources("user@example.com", "token")
    assert len(sites) == 1
    assert sites[0]["cloud_id"] == "c2"


def test_fetch_accessible_resources_basic_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = [
        {"id": "c1", "url": "https://team.atlassian.net", "name": "Team"},
    ]
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        client.get.return_value = response
        sites = _fetch_accessible_resources("user@example.com", "token")
    assert len(sites) == 1


def test_fetch_accessible_resources_http_error_tries_next_scheme() -> None:
    bearer = MagicMock()
    bearer.status_code = 200
    bearer.json.return_value = [
        {"id": "c2", "url": "https://other.atlassian.net", "name": "Other"},
    ]
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        client.get.side_effect = [AuthHttpError("down"), bearer]
        sites = _fetch_accessible_resources("user@example.com", "token")
    assert len(sites) == 1
    assert sites[0]["cloud_id"] == "c2"


def test_fetch_accessible_resources_invalid_json_tries_next_scheme() -> None:
    bad_json = MagicMock()
    bad_json.status_code = 200
    bad_json.json.side_effect = ValueError("not json")
    good = MagicMock()
    good.status_code = 200
    good.json.return_value = [
        {"id": "c1", "url": "https://team.atlassian.net", "name": "Team"},
    ]
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.AuthHttpClient"
    ) as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        client.get.side_effect = [bad_json, good]
        sites = _fetch_accessible_resources("user@example.com", "token")
    assert len(sites) == 1


def test_discover_sites_with_api_token_filters_by_gateway() -> None:
    from adapters.inbound.cli.auth.atlassian_auth import AtlassianVerifyResult

    candidates = [
        {
            "cloud_id": "c1",
            "site_url": "https://team.atlassian.net",
            "site_name": "Team",
        }
    ]
    with (
        patch(
            "adapters.outbound.cli_auth.atlassian_client.collect_site_candidates",
            return_value=candidates,
        ),
        patch(
            "adapters.outbound.cli_auth.atlassian_client.verify_gateway_product",
            return_value=AtlassianVerifyResult(ok=True, display_name="Ada"),
        ),
    ):
        sites = discover_sites_with_api_token("u@example.com", "tok", "jira")
    assert len(sites) == 1
    assert sites[0]["cloud_id"] == "c1"


def test_collect_login_site_candidates_merges_resources_and_email_hints() -> None:
    from adapters.inbound.cli.auth.atlassian_auth import collect_login_site_candidates

    with (
        patch(
            "adapters.outbound.cli_auth.atlassian_client._fetch_accessible_resources",
            return_value=[
                {
                    "cloud_id": "c1",
                    "site_url": "https://team.atlassian.net",
                    "site_name": "Team",
                }
            ],
        ),
        patch(
            "adapters.outbound.cli_auth.atlassian_client.fetch_cloud_id_for_site",
            return_value="c1",
        ),
    ):
        sites = collect_login_site_candidates(
            "user@team.com",
            "token",
            include_email_hints=True,
        )
    assert sites
    assert any(s["source"] == "accessible-resources" for s in sites)


def test_fetch_accessible_resources_alias() -> None:
    with patch(
        "adapters.outbound.cli_auth.atlassian_client.discover_sites_with_api_token",
        return_value=[],
    ) as discover:
        fetch_accessible_resources("u@example.com", "tok")
    discover.assert_called_once_with("u@example.com", "tok", "jira")


# --- test_atlassian_auth_headers.py ---


def test_tenant_base_uses_basic_only() -> None:
    headers = _auth_header_variants(
        "user@example.com",
        "secret-token",
        base="https://potpie-team.atlassian.net",
    )
    assert len(headers) == 1
    assert headers[0]["Authorization"].startswith("Basic ")


def test_gateway_base_allows_bearer_fallback() -> None:
    headers = _auth_header_variants(
        "user@example.com",
        "secret-token",
        base="https://api.atlassian.com/ex/jira/cloud-id",
    )
    assert len(headers) == 2
    assert headers[0]["Authorization"].startswith("Basic ")
    assert headers[1]["Authorization"].startswith("Bearer ")


# --- test_atlassian_read.py ---


def _mock_keychain(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    secrets: dict[str, str] = {}

    def _store(_label: str, username: str, value: str) -> None:
        secrets[username] = value

    def _load(_label: str, username: str) -> str | None:
        return secrets.get(username)

    monkeypatch.setattr(cs, "_store_file_secret", _store)
    monkeypatch.setattr(cs, "_load_file_secret", _load)
    return secrets


def test_get_json_raises_atlassian_read_error_on_http_error() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.get.side_effect = AuthHttpError("connection refused")

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
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
    assert "cid-1" in str(exc_info.value)


def test_get_json_retries_after_transport_error_on_first_variant() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"issues": []}
    client.get.side_effect = [AuthHttpError("down"), ok]

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
        data = _get_json(
            email="user@example.com",
            api_token="secret",
            product="jira",
            cloud_id="cid-1",
            site_url="https://team.atlassian.net",
            path="/rest/api/3/search",
            site_first=True,
        )

    assert data == {"issues": []}


def test_post_json_raises_atlassian_read_error_on_http_error() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.side_effect = AuthHttpError("connection refused")

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
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

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
        issues = fetch_jira_issues_sample(limit=5)

    assert len(issues) == 1
    assert issues[0]["key"] == "ENG-1"
    assert issues[0]["summary"] == "Fix bug"


def test_fetch_confluence_spaces_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
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

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
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

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
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

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
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

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
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

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
        data = _get_json(
            email="user@example.com",
            api_token="secret",
            product="jira",
            cloud_id="cid-1",
            site_url="https://team.atlassian.net",
            path="/rest/api/3/myself",
        )

    assert data == {"data": [{"id": "1"}]}


def test_prompt_workspace_interactive_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_read as atlassian_read

    prompts = iter(["2"])
    monkeypatch.setattr(
        atlassian_read.typer, "prompt", lambda *args, **kwargs: next(prompts)
    )
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
        "adapters.outbound.cli_auth.atlassian_read_client.fetch_jira_issues_in_project",
        lambda key, limit: [{"key": f"{key}-1"}],
    )

    rows = fetch_jira_issues_sample(limit=5)
    assert rows == [{"key": "ENG-1"}]


def test_credential_helpers_require_cloud_id_and_site_url() -> None:
    with pytest.raises(AtlassianReadError, match="Missing cloud_id"):
        _cloud_id_from_credentials({})
    with pytest.raises(AtlassianReadError, match="Missing site_url"):
        _site_url_from_credentials({"cloud_id": "c1"})


# --- test_atlassian_workspaces.py ---


def _save_creds(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cred_path = tmp_path / "credentials.json"
    secrets: dict[str, str] = {}

    def _store(_label: str, username: str, value: str) -> None:
        secrets[username] = value

    def _load(_label: str, username: str) -> str | None:
        return secrets.get(username)

    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    monkeypatch.setattr(cs, "_store_file_secret", _store)
    monkeypatch.setattr(cs, "_load_file_secret", _load)
    cs.save_jira_credentials(
        {
            "email": "user@example.com",
            "api_token": "secret",
            "cloud_id": "cid-1",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        }
    )


def test_fetch_jira_issues_in_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
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
                    "description": {
                        "type": "doc",
                        "content": [{"type": "text", "text": "Details"}],
                    },
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

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
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


def test_fetch_confluence_pages_in_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
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

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
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

    with patch(
        "adapters.outbound.cli_auth.atlassian_read_client.AuthHttpClient",
        return_value=client,
    ):
        projects = fetch_jira_projects(limit=5)

    assert projects[0]["key"] == "ENG"
    assert "browse/ENG" in (projects[0].get("url") or "")


def test_jira_use_flow_saves_prefs_only_after_successful_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _save_creds(monkeypatch, tmp_path)
    saved: list[str] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_read.save_jira_workspace_prefs",
        lambda *, project_key: saved.append(project_key),
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_read.fetch_jira_projects",
        lambda **kwargs: [{"key": "ENG", "name": "Engineering"}],
    )

    def _fail_fetch(*args: object, **kwargs: object) -> list[dict]:
        raise AtlassianReadError("jira read failed")

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_read.fetch_jira_issues_in_project",
        _fail_fetch,
    )

    with pytest.raises(AtlassianReadError):
        run_jira_use_flow(workspace_key="ENG", limit=5)

    assert saved == []

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_read.fetch_jira_issues_in_project",
        lambda *args, **kwargs: [{"key": "ENG-1"}],
    )
    result = run_jira_use_flow(workspace_key="ENG", limit=5)
    assert saved == ["ENG"]
    assert result["workspace_key"] == "ENG"


def test_jira_use_flow_requires_key_when_non_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_read as atlassian_read

    monkeypatch.setattr(atlassian_read.sys.stdin, "isatty", lambda: False)
    with pytest.raises(AtlassianReadError, match="Interactive workspace"):
        run_jira_use_flow()


def test_confluence_use_flow_requires_key_when_non_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_read as atlassian_read

    monkeypatch.setattr(atlassian_read.sys.stdin, "isatty", lambda: False)
    with pytest.raises(AtlassianReadError, match="Interactive workspace"):
        run_confluence_use_flow()


def test_jira_use_flow_interactive_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import adapters.inbound.cli.auth.atlassian_read as atlassian_read

    _save_creds(monkeypatch, tmp_path)
    monkeypatch.setattr(atlassian_read.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        atlassian_read,
        "fetch_jira_projects",
        lambda: [{"key": "ENG", "name": "Engineering"}],
    )
    monkeypatch.setattr(
        atlassian_read,
        "_prompt_workspace",
        lambda items, label: items[0],
    )
    monkeypatch.setattr(
        atlassian_read,
        "fetch_jira_issues_in_project",
        lambda key, limit: [{"key": f"{key}-1"}],
    )

    result = run_jira_use_flow(limit=3)
    assert result["workspace_key"] == "ENG"


def test_confluence_use_flow_saves_prefs_only_after_successful_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
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
    saved: list[str] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_read.save_confluence_workspace_prefs",
        lambda *, space_key: saved.append(space_key),
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_read.fetch_confluence_spaces_sample",
        lambda **kwargs: [{"key": "DOCS", "name": "Docs"}],
    )

    def _fail_fetch(*args: object, **kwargs: object) -> list[dict]:
        raise AtlassianReadError("confluence read failed")

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_read.fetch_confluence_pages_in_space",
        _fail_fetch,
    )

    with pytest.raises(AtlassianReadError):
        run_confluence_use_flow(workspace_key="DOCS", limit=5)

    assert saved == []

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_read.fetch_confluence_pages_in_space",
        lambda *args, **kwargs: [{"title": "Page"}],
    )
    result = run_confluence_use_flow(workspace_key="DOCS", limit=5)
    assert saved == ["DOCS"]
    assert result["workspace_key"] == "DOCS"
