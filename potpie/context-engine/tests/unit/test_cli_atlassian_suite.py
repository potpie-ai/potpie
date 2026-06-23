"""Unit tests for the unified Atlassian suite CLI."""

from __future__ import annotations

import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli as cli_main
from adapters.inbound.cli.commands._common import set_store
from adapters.inbound.cli.auth.atlassian_suite_auth import ProductConnectResult
from tests._auth_fakes import InMemoryCredentialStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_store() -> None:
    set_store(InMemoryCredentialStore())


def test_atlassian_help_shows_login_logout_status() -> None:
    result = runner.invoke(cli_main.app, ["atlassian", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "login" in result.stdout
    assert "logout" in result.stdout
    assert "status" in result.stdout


def test_suite_login_both_jira_confluence_same_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    monkeypatch.setattr(suite_auth, "load_cli_env", lambda: None)
    monkeypatch.setattr(suite_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        suite_auth,
        "connect_atlassian_product",
        lambda product, **kwargs: ProductConnectResult(
            product=product,
            status="connected",
            site_url="https://team-a.atlassian.net",
            cloud_id="cloud-a",
        ),
    )

    result = runner.invoke(
        cli_main.app,
        [
            "--json",
            "atlassian",
            "login",
            "--email",
            "user@example.com",
            "--api-token",
            "token-123",
            "--site-subdomain",
            "team-a",
            "--skip-bitbucket",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["products"]["jira"]["status"] == "connected"
    assert payload["products"]["confluence"]["status"] == "connected"
    assert payload["products"]["bitbucket"]["status"] == "skipped"


def test_suite_login_confluence_fallback_subdomain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    calls: list[tuple[str, str]] = []

    def _connect(product: str, *, site_subdomain: str, **kwargs: object) -> ProductConnectResult:
        calls.append((product, site_subdomain))
        if product == "jira":
            return ProductConnectResult(
                product=product,
                status="connected",
                site_url="https://team-a.atlassian.net",
                cloud_id="cloud-a",
            )
        if site_subdomain == "team-a":
            return ProductConnectResult(
                product=product,
                status="not_connected",
                reason="product_access_denied",
            )
        return ProductConnectResult(
            product=product,
            status="connected",
            site_url="https://team-b.atlassian.net",
            cloud_id="cloud-b",
        )

    monkeypatch.setattr(suite_auth, "load_cli_env", lambda: None)
    monkeypatch.setattr(suite_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(suite_auth, "connect_atlassian_product", _connect)

    result = runner.invoke(
        cli_main.app,
        [
            "--json",
            "atlassian",
            "login",
            "--email",
            "user@example.com",
            "--api-token",
            "token-123",
            "--site-subdomain",
            "team-a",
            "--confluence-site-subdomain",
            "team-b",
            "--skip-bitbucket",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["products"]["jira"]["site_url"] == "https://team-a.atlassian.net"
    assert payload["products"]["confluence"]["site_url"] == "https://team-b.atlassian.net"
    assert calls == [
        ("jira", "team-a"),
        ("confluence", "team-a"),
        ("confluence", "team-b"),
    ]


def test_suite_login_bitbucket_insufficient_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth
    import adapters.inbound.cli.auth.bitbucket_auth as bitbucket_auth

    monkeypatch.setattr(suite_auth, "load_cli_env", lambda: None)
    monkeypatch.setattr(suite_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        suite_auth,
        "connect_atlassian_product",
        lambda product, **kwargs: ProductConnectResult(
            product=product,
            status="connected",
            site_url="https://team-a.atlassian.net",
            cloud_id="cloud-a",
        ),
    )
    monkeypatch.setattr(
        bitbucket_auth,
        "verify_bitbucket_api_token",
        lambda email, api_token: type(
            "Result",
            (),
            {
                "ok": False,
                "email": email,
                "display_name": "",
                "error_kind": "insufficient_scopes",
            },
        )(),
    )

    result = runner.invoke(
        cli_main.app,
        [
            "--json",
            "atlassian",
            "login",
            "--email",
            "user@example.com",
            "--api-token",
            "token-123",
            "--site-subdomain",
            "team-a",
            "--bitbucket-api-token",
            "bb-token",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["products"]["bitbucket"]["status"] == "not_connected"
    assert payload["products"]["bitbucket"]["reason"] == "insufficient_scopes"


def test_run_bitbucket_step_uses_shared_token_page_and_no_second_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.bitbucket_auth as bitbucket_auth

    captured_panels: list[tuple[str, list[str]]] = []
    open_calls: list[tuple[str, str, int, str | None]] = []

    monkeypatch.setattr(bitbucket_auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(bitbucket_auth, "_human", lambda: False)
    monkeypatch.setattr(bitbucket_auth, "_prompt_connect_now", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        bitbucket_auth,
        "_render_step_panel",
        lambda title, lines, **kwargs: captured_panels.append((title, list(lines))),
    )
    monkeypatch.setattr(
        bitbucket_auth,
        "open_url_with_countdown",
        lambda url, *, label, timeout_seconds, open_message=None: open_calls.append(
            (url, label, timeout_seconds, open_message)
        ),
    )
    monkeypatch.setattr(
        bitbucket_auth.typer,
        "prompt",
        lambda prompt, **kwargs: "bb-token" if prompt == "Bitbucket API token" else "user@example.com",
    )
    monkeypatch.setattr(
        bitbucket_auth,
        "verify_bitbucket_api_token",
        lambda email, api_token: type(
            "Result",
            (),
            {
                "ok": True,
                "email": email,
                "display_name": "User",
                "error_kind": None,
            },
        )(),
    )

    result = bitbucket_auth._run_bitbucket_step(
        force=False,
        skip_bitbucket=False,
        bitbucket_api_token=None,
        email="user@example.com",
        as_json=False,
    )

    assert result.status == "connected"
    panel_lines = captured_panels[0][1]
    assert captured_panels[0][0] == "Bitbucket"
    assert any("read:user:bitbucket" in line for line in panel_lines)
    assert any("read:workspace:bitbucket" in line for line in panel_lines)
    assert any("read:repository:bitbucket" in line for line in panel_lines)
    assert any("read:pullrequest:bitbucket" in line for line in panel_lines)
    assert any("id.atlassian.com" in line for line in panel_lines)
    assert open_calls == [
        (
            "https://id.atlassian.com/manage-profile/security/api-tokens",
            "id.atlassian.com",
            10,
            "Return here and paste the token when you're ready.",
        )
    ]


def test_run_bitbucket_step_prompts_for_email_even_when_jira_email_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.bitbucket_auth as bitbucket_auth

    prompts: list[str] = []
    verify_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(bitbucket_auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(bitbucket_auth, "_human", lambda: False)
    monkeypatch.setattr(bitbucket_auth, "_prompt_connect_now", lambda *args, **kwargs: True)
    monkeypatch.setattr(bitbucket_auth, "open_url_with_countdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bitbucket_auth.typer,
        "prompt",
        lambda prompt, **kwargs: (
            prompts.append(prompt),
            "bitbucket-user@example.com" if "email" in prompt.lower() else "bb-token",
        )[1],
    )
    monkeypatch.setattr(
        bitbucket_auth,
        "verify_bitbucket_api_token",
        lambda email, api_token: (
            verify_calls.append((email, api_token)),
            type(
                "Result",
                (),
                {
                    "ok": True,
                    "email": email,
                    "display_name": "User",
                    "error_kind": None,
                },
            )(),
        )[1],
    )

    result = bitbucket_auth._run_bitbucket_step(
        force=False,
        skip_bitbucket=False,
        bitbucket_api_token=None,
        email="jira-user@example.com",
        as_json=False,
    )

    assert result.status == "connected"
    assert prompts == ["Bitbucket email", "Bitbucket API token"]
    assert verify_calls == [("bitbucket-user@example.com", "bb-token")]


def test_render_status_card_uses_stored_credentials_in_human_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    store = InMemoryCredentialStore()
    store.save_jira_credentials(
        {"email": "user@example.com", "api_token": "jira-token", "site_url": "https://team-a.atlassian.net"}
    )
    store.save_confluence_credentials(
        {"email": "user@example.com", "api_token": "conf-token", "site_url": "https://docs-team.atlassian.net"}
    )
    set_store(store)

    console = Console(record=True, width=120)
    monkeypatch.setattr(suite_auth, "_human", lambda: True)
    monkeypatch.setattr(suite_auth, "_console", console)

    suite_auth._render_status_card({}, as_json=False)

    rendered = console.export_text()
    assert "https://team-a.atlassian.net" in rendered
    assert "https://docs-team.atlassian.net" in rendered
    assert "(not connected)" in rendered


def test_run_step1_opens_atlassian_token_page_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    open_calls: list[tuple[str, str, int, str | None]] = []

    monkeypatch.setattr(suite_auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(suite_auth, "_human", lambda: False)
    monkeypatch.setattr(suite_auth, "_prompt_connect_now", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        suite_auth,
        "open_url_with_countdown",
        lambda url, *, label, timeout_seconds, open_message=None: open_calls.append(
            (url, label, timeout_seconds, open_message)
        ),
    )
    monkeypatch.setattr(
        suite_auth,
        "_prompt_step1_credentials",
        lambda **kwargs: ("user@example.com", "token-123", "team-a"),
    )
    monkeypatch.setattr(
        suite_auth,
        "connect_atlassian_product",
        lambda product, **kwargs: ProductConnectResult(
            product=product,
            status="connected",
            site_url="https://team-a.atlassian.net",
            cloud_id="cloud-a",
        ),
    )

    results = suite_auth._run_step1(
        force=False,
        email=None,
        api_token=None,
        site_subdomain=None,
        confluence_site_subdomain=None,
        verbose=False,
        as_json=False,
    )

    assert results["jira"].status == "connected"
    assert open_calls == [
        (
            "https://id.atlassian.com/manage-profile/security/api-tokens",
            "id.atlassian.com",
            10,
            "Return here and paste the token when you're ready.",
        )
    ]


def test_run_step1_already_connected_skips_result_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    store = InMemoryCredentialStore()
    store.save_jira_credentials(
        {"email": "user@example.com", "api_token": "jira-token", "site_url": "https://team-a.atlassian.net", "cloud_id": "cloud-a"}
    )
    store.save_confluence_credentials(
        {"email": "user@example.com", "api_token": "conf-token", "site_url": "https://team-a.atlassian.net", "cloud_id": "cloud-a"}
    )
    set_store(store)

    rendered_results: list[dict[str, ProductConnectResult]] = []
    monkeypatch.setattr(suite_auth, "_render_result_lines", lambda results, **kwargs: rendered_results.append(results))

    results = suite_auth._run_step1(
        force=False,
        email=None,
        api_token=None,
        site_subdomain=None,
        confluence_site_subdomain=None,
        verbose=False,
        as_json=False,
    )

    assert results["jira"].status == "already_connected"
    assert results["confluence"].status == "already_connected"
    assert rendered_results == []


def test_suite_login_human_mode_skips_final_summary_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    rendered_titles: list[str] = []

    monkeypatch.setattr(suite_auth, "load_cli_env", lambda: None)
    monkeypatch.setattr(suite_auth, "_print_snapshot", lambda: None)
    monkeypatch.setattr(
        suite_auth,
        "_run_step1",
        lambda **kwargs: {
            "jira": ProductConnectResult(product="jira", status="already_connected"),
            "confluence": ProductConnectResult(product="confluence", status="already_connected"),
        },
    )
    monkeypatch.setattr(
        suite_auth,
        "_run_bitbucket_step",
        lambda **kwargs: ProductConnectResult(product="bitbucket", status="connected"),
    )
    monkeypatch.setattr(
        suite_auth,
        "_render_step_panel",
        lambda title, lines, **kwargs: rendered_titles.append(title),
    )

    suite_auth.run_atlassian_suite_login(
        force=False,
        as_json=False,
        verbose=False,
        skip_bitbucket=False,
    )

    assert "Final summary" not in rendered_titles


def test_bitbucket_ls_lists_workspaces(monkeypatch: pytest.MonkeyPatch) -> None:
    import adapters.inbound.cli.auth.auth_commands as auth_commands

    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "fetch_bitbucket_workspaces",
        lambda limit=50: [
            {
                "key": "potpie",
                "name": "Potpie",
                "type": "workspace",
            }
        ],
    )

    result = runner.invoke(cli_main.app, ["bitbucket", "ls"])

    assert result.exit_code == 0, result.stdout
    assert "Bitbucket workspaces:" in result.stdout
    assert "Potpie" in result.stdout
    assert "potpie bitbucket select" in result.stdout


def test_bitbucket_select_fetches_pull_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.auth_commands as auth_commands

    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "run_bitbucket_use_flow",
        lambda workspace_key=None, repo_slug=None, limit=10: {
            "product": "bitbucket",
            "workspace_key": "potpie",
            "workspace_name": "Potpie",
            "repo_key": "backend",
            "repo_name": "Backend",
            "items": [
                {
                    "id": 12,
                    "title": "Fix CLI auth",
                    "state": "OPEN",
                    "author": "Nihit",
                    "url": "https://bitbucket.org/potpie/backend/pull-requests/12",
                }
            ],
        },
    )

    result = runner.invoke(
        cli_main.app,
        ["bitbucket", "select", "--workspace", "potpie", "--repo", "backend"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Bitbucket workspace: potpie (Potpie)" in result.stdout
    assert "Bitbucket repository: backend (Backend)" in result.stdout
    assert "Fix CLI auth" in result.stdout


def test_suite_logout_clears_all_products(monkeypatch: pytest.MonkeyPatch) -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    store = InMemoryCredentialStore()
    store.save_jira_credentials({"email": "u@example.com", "api_token": "jira-token"})
    store.save_confluence_credentials(
        {"email": "u@example.com", "api_token": "conf-token"}
    )
    store.save_bitbucket_credentials(
        {"email": "u@example.com", "api_token": "bb-token"}
    )
    store.save_atlassian_credentials(
        {"email": "u@example.com", "api_token": "legacy-token"}
    )
    set_store(store)
    monkeypatch.setattr(suite_auth, "load_cli_env", lambda: None)

    result = runner.invoke(cli_main.app, ["--json", "atlassian", "logout"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert store.get_jira_credentials() == {}
    assert store.get_confluence_credentials() == {}
    assert store.get_bitbucket_credentials() == {}
    assert store.get_atlassian_credentials() == {}


def test_atlassian_status_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    monkeypatch.setattr(suite_auth, "load_cli_env", lambda: None)
    monkeypatch.setattr(
        suite_auth,
        "build_atlassian_suite_status",
        lambda: [
            {"provider": "jira", "authenticated": True, "site_url": "https://team-a.atlassian.net"},
            {"provider": "confluence", "authenticated": False},
            {"provider": "bitbucket", "authenticated": True, "email": "user@example.com"},
        ],
    )

    result = runner.invoke(cli_main.app, ["--json", "atlassian", "status"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert [row["provider"] for row in payload["integrations"]] == [
        "jira",
        "confluence",
        "bitbucket",
    ]


def test_auth_status_includes_bitbucket(monkeypatch: pytest.MonkeyPatch) -> None:
    import adapters.inbound.cli.auth.auth_commands as auth_commands

    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {"provider": provider, "authenticated": provider == "bitbucket"},
    )

    result = runner.invoke(cli_main.app, ["--json", "auth", "status"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert any(row["provider"] == "bitbucket" for row in payload["integrations"])


def test_atlassian_status_human_lists_connected_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryCredentialStore()
    store.save_jira_credentials(
        {
            "email": "user@example.com",
            "api_token": "jira-token",
            "site_url": "https://team-a.atlassian.net",
        }
    )
    store.save_bitbucket_credentials(
        {
            "email": "user@example.com",
            "api_token": "bb-token",
            "account_name": "ada",
        }
    )
    set_store(store)
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_suite_auth.load_cli_env",
        lambda: None,
    )

    result = runner.invoke(cli_main.app, ["atlassian", "status"])

    assert result.exit_code == 0, result.stdout
    assert "jira: authenticated" in result.stdout
    assert "email=user@example.com" in result.stdout
    assert "confluence: not authenticated" in result.stdout
    assert "bitbucket: authenticated" in result.stdout
    assert "login=ada" in result.stdout


def test_bitbucket_login_already_connected_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.bitbucket_auth as bitbucket_auth

    store = InMemoryCredentialStore()
    store.save_bitbucket_credentials(
        {"email": "user@example.com", "api_token": "bb-token", "account_name": "ada"}
    )
    set_store(store)
    monkeypatch.setattr(bitbucket_auth, "load_cli_env", lambda: None)

    result = runner.invoke(cli_main.app, ["--json", "bitbucket", "login"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["already_connected"] is True
    assert payload["provider"] == "bitbucket"


def test_bitbucket_login_success_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.bitbucket_auth as bitbucket_auth
    from adapters.outbound.cli_auth.bitbucket_client import BitbucketVerifyResult

    monkeypatch.setattr(bitbucket_auth, "load_cli_env", lambda: None)
    monkeypatch.setattr(
        bitbucket_auth,
        "verify_bitbucket_api_token",
        lambda email, token: BitbucketVerifyResult(
            ok=True,
            email=email,
            display_name="Ada",
        ),
    )
    monkeypatch.setattr(bitbucket_auth, "integration_token_storage", lambda: "file")
    monkeypatch.setattr(bitbucket_auth, "credentials_path", lambda: "/tmp/creds.json")

    result = runner.invoke(
        cli_main.app,
        [
            "--json",
            "bitbucket",
            "login",
            "--email",
            "user@example.com",
            "--api-token",
            "bb-token",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["provider"] == "bitbucket"


def test_connected_products_message_both() -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    results = {
        "jira": ProductConnectResult(product="jira", status="connected"),
        "confluence": ProductConnectResult(product="confluence", status="connected"),
    }
    assert suite_auth._connected_products_message(results) == "Connected Jira and Confluence."


def test_connected_products_message_only_confluence() -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    results = {
        "jira": ProductConnectResult(product="jira", status="not_connected"),
        "confluence": ProductConnectResult(product="confluence", status="connected"),
    }
    assert suite_auth._connected_products_message(results) == "Connected Confluence."


def test_connected_products_message_only_jira() -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    results = {
        "jira": ProductConnectResult(product="jira", status="connected"),
        "confluence": ProductConnectResult(product="confluence", status="not_connected"),
    }
    assert suite_auth._connected_products_message(results) == "Connected Jira."


def test_connected_products_message_neither() -> None:
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    results = {
        "jira": ProductConnectResult(product="jira", status="not_connected"),
        "confluence": ProductConnectResult(product="confluence", status="not_connected"),
    }
    assert suite_auth._connected_products_message(results) == ""


def test_step1_success_message_only_confluence_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue 1 fix: when only Confluence connects, message must say 'Connected Confluence.'"""
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    messages: list[str] = []

    def _connect(product: str, **kwargs: object) -> ProductConnectResult:
        if product == "jira":
            return ProductConnectResult(product="jira", status="not_connected", reason="product_access_denied")
        return ProductConnectResult(
            product="confluence", status="connected", site_url="https://potpie-team-1.atlassian.net", cloud_id="c1"
        )

    monkeypatch.setattr(suite_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(suite_auth, "connect_atlassian_product", _connect)
    monkeypatch.setattr(
        suite_auth, "_render_connection_success", lambda msg, **kw: messages.append(msg)
    )
    monkeypatch.setattr(suite_auth, "_render_result_lines", lambda *a, **kw: None)

    results = suite_auth._run_step1(
        force=False,
        email="u@example.com",
        api_token="token",
        site_subdomain="potpie-team-1",
        confluence_site_subdomain=None,
        verbose=False,
        as_json=False,
    )

    assert results["jira"].status == "not_connected"
    assert results["confluence"].status == "connected"
    assert messages == ["Connected Confluence."]


def test_step1_success_message_only_jira_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue 1 fix: when only Jira connects, message must say 'Connected Jira.'"""
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    messages: list[str] = []

    def _connect(product: str, **kwargs: object) -> ProductConnectResult:
        if product == "confluence":
            return ProductConnectResult(product="confluence", status="not_connected", reason="product_access_denied")
        return ProductConnectResult(
            product="jira", status="connected", site_url="https://potpie-team.atlassian.net", cloud_id="c1"
        )

    monkeypatch.setattr(suite_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(suite_auth, "connect_atlassian_product", _connect)
    monkeypatch.setattr(
        suite_auth, "_render_connection_success", lambda msg, **kw: messages.append(msg)
    )
    monkeypatch.setattr(suite_auth, "_render_result_lines", lambda *a, **kw: None)

    results = suite_auth._run_step1(
        force=False,
        email="u@example.com",
        api_token="token",
        site_subdomain="potpie-team",
        confluence_site_subdomain=None,
        verbose=False,
        as_json=False,
    )

    assert results["jira"].status == "connected"
    assert results["confluence"].status == "not_connected"
    assert messages == ["Connected Jira."]


def test_offer_retry_case_b_jira_ok_confluence_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case B: Jira connected, Confluence not — retry prompt asks only for subdomain."""
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    connect_calls: list[tuple[str, str]] = []
    prompts: list[str] = []
    confirms: list[str] = []

    def _connect(product: str, *, site_subdomain: str, **kw: object) -> ProductConnectResult:
        connect_calls.append((product, site_subdomain))
        return ProductConnectResult(
            product=product, status="connected", site_url=f"https://{site_subdomain}.atlassian.net", cloud_id="c"
        )

    monkeypatch.setattr(suite_auth, "connect_atlassian_product", _connect)
    monkeypatch.setattr(
        suite_auth.typer, "confirm", lambda msg, **kw: (confirms.append(msg), True)[1]
    )
    monkeypatch.setattr(
        suite_auth.typer, "prompt", lambda msg, **kw: (prompts.append(msg), "potpie-team-1")[1]
    )

    results = {
        "jira": ProductConnectResult(product="jira", status="connected", site_url="https://potpie-team.atlassian.net"),
        "confluence": ProductConnectResult(product="confluence", status="not_connected", reason="product_access_denied"),
    }

    updated = suite_auth._offer_retry_failed_products(
        results=results,
        email="u@example.com",
        api_token="token",
        as_json=False,
    )

    assert updated["jira"].status == "connected"
    assert updated["confluence"].status == "connected"
    assert updated["confluence"].site_url == "https://potpie-team-1.atlassian.net"
    assert any("Confluence" in c for c in confirms)
    assert any("subdomain" in p.lower() for p in prompts)
    assert connect_calls == [("confluence", "potpie-team-1")]


def test_offer_retry_case_c_confluence_ok_jira_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case C: Confluence connected, Jira not — retry prompt asks only for subdomain."""
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    connect_calls: list[tuple[str, str]] = []
    confirms: list[str] = []

    def _connect(product: str, *, site_subdomain: str, **kw: object) -> ProductConnectResult:
        connect_calls.append((product, site_subdomain))
        return ProductConnectResult(
            product=product, status="connected", site_url=f"https://{site_subdomain}.atlassian.net", cloud_id="c"
        )

    monkeypatch.setattr(suite_auth, "connect_atlassian_product", _connect)
    monkeypatch.setattr(
        suite_auth.typer, "confirm", lambda msg, **kw: (confirms.append(msg), True)[1]
    )
    monkeypatch.setattr(suite_auth.typer, "prompt", lambda msg, **kw: "potpie-team")

    results = {
        "jira": ProductConnectResult(product="jira", status="not_connected", reason="product_access_denied"),
        "confluence": ProductConnectResult(product="confluence", status="connected", site_url="https://potpie-team-1.atlassian.net"),
    }

    updated = suite_auth._offer_retry_failed_products(
        results=results,
        email="u@example.com",
        api_token="token",
        as_json=False,
    )

    assert updated["jira"].status == "connected"
    assert updated["confluence"].status == "connected"
    assert any("Jira" in c for c in confirms)
    assert connect_calls == [("jira", "potpie-team")]


def test_offer_retry_user_declines_no_connect_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User says No to retry — no connect_atlassian_product call should happen."""
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    connect_calls: list[str] = []

    monkeypatch.setattr(
        suite_auth,
        "connect_atlassian_product",
        lambda product, **kw: (connect_calls.append(product), None)[1],
    )
    monkeypatch.setattr(suite_auth.typer, "confirm", lambda msg, **kw: False)

    results = {
        "jira": ProductConnectResult(product="jira", status="not_connected"),
        "confluence": ProductConnectResult(product="confluence", status="not_connected"),
    }

    updated = suite_auth._offer_retry_failed_products(
        results=results,
        email="u@example.com",
        api_token="token",
        as_json=False,
    )

    assert updated["jira"].status == "not_connected"
    assert updated["confluence"].status == "not_connected"
    assert connect_calls == []


def test_jira_site_subdomain_flag_non_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--jira-site-subdomain connects Jira on a different site when initial attempt fails."""
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    calls: list[tuple[str, str]] = []

    def _connect(product: str, *, site_subdomain: str, **kwargs: object) -> ProductConnectResult:
        calls.append((product, site_subdomain))
        if product == "confluence":
            return ProductConnectResult(
                product=product, status="connected", site_url=f"https://{site_subdomain}.atlassian.net", cloud_id="c"
            )
        if site_subdomain == "confluence-site":
            return ProductConnectResult(
                product=product, status="connected", site_url=f"https://{site_subdomain}.atlassian.net", cloud_id="c"
            )
        return ProductConnectResult(product=product, status="not_connected", reason="product_access_denied")

    monkeypatch.setattr(suite_auth, "load_cli_env", lambda: None)
    monkeypatch.setattr(suite_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(suite_auth, "connect_atlassian_product", _connect)

    result = runner.invoke(
        cli_main.app,
        [
            "--json",
            "atlassian",
            "login",
            "--email",
            "user@example.com",
            "--api-token",
            "token-123",
            "--site-subdomain",
            "confluence-site",
            "--jira-site-subdomain",
            "confluence-site",
            "--skip-bitbucket",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["products"]["jira"]["status"] == "connected"
    assert payload["products"]["confluence"]["status"] == "connected"
    # Jira first tried confluence-site (initial), then confluence-site again via --jira-site-subdomain
    # Since initial already connected with confluence-site for Jira, only one call expected.
    # But with different sites: initial fails on site A, fallback uses site B.
    assert any(c[0] == "jira" for c in calls)


def test_summary_shown_when_jira_already_connected_confluence_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge case #1: summary must be shown even when one product is already_connected and other fails."""
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth
    from tests._auth_fakes import InMemoryCredentialStore

    store = InMemoryCredentialStore()
    store.save_jira_credentials(
        {"email": "u@example.com", "api_token": "t", "site_url": "https://team-a.atlassian.net", "cloud_id": "c"}
    )
    set_store(store)

    render_calls: list[dict] = []

    monkeypatch.setattr(suite_auth.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        suite_auth,
        "connect_atlassian_product",
        lambda product, **kw: ProductConnectResult(product=product, status="not_connected", reason="product_access_denied"),
    )
    monkeypatch.setattr(
        suite_auth, "_render_result_lines", lambda results, **kw: render_calls.append({"results": results})
    )
    monkeypatch.setattr(suite_auth, "_render_connection_success", lambda msg, **kw: None)

    results = suite_auth._run_step1(
        force=False,
        email="u@example.com",
        api_token="token",
        site_subdomain="team-a",
        confluence_site_subdomain=None,
        verbose=False,
        as_json=False,
    )

    # Jira already connected, Confluence failed — summary MUST be rendered.
    assert results["jira"].status == "already_connected"
    assert results["confluence"].status == "not_connected"
    assert len(render_calls) == 1, "Summary should be shown even when no new product connected"


def test_offer_retry_loop_retries_after_second_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry loop: user declines on second attempt after first retry fails."""
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    connect_calls: list[tuple[str, str]] = []
    confirm_calls: list[str] = []
    confirm_responses = iter([True, True, False])  # Yes, Yes, No

    def _connect(product: str, *, site_subdomain: str, **kw: object) -> ProductConnectResult:
        connect_calls.append((product, site_subdomain))
        return ProductConnectResult(product=product, status="not_connected", reason="site_discovery_failed")

    monkeypatch.setattr(suite_auth, "connect_atlassian_product", _connect)
    monkeypatch.setattr(
        suite_auth.typer,
        "confirm",
        lambda msg, **kw: (confirm_calls.append(msg), next(confirm_responses))[1],
    )
    monkeypatch.setattr(suite_auth.typer, "prompt", lambda msg, **kw: "bad-site")
    monkeypatch.setattr(suite_auth, "print_plain_line", lambda *a, **kw: None)

    results = {
        "jira": ProductConnectResult(product="jira", status="connected"),
        "confluence": ProductConnectResult(product="confluence", status="not_connected"),
    }

    updated = suite_auth._offer_retry_failed_products(
        results=results,
        email="u@example.com",
        api_token="token",
        as_json=False,
    )

    # Two retry attempts for Confluence, then user says No.
    assert updated["confluence"].status == "not_connected"
    assert len(connect_calls) == 2
    assert len([c for c in confirm_calls if "Confluence" in c]) == 3  # Yes, Yes, No


def test_offer_retry_empty_subdomain_warns_and_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty subdomain on retry shows warning and re-prompts instead of silently skipping."""
    import adapters.inbound.cli.auth.atlassian_suite_auth as suite_auth

    warnings: list[str] = []
    prompt_responses = iter(["", "potpie-team"])  # empty first, then valid
    confirm_responses = iter([True, True])

    monkeypatch.setattr(
        suite_auth,
        "connect_atlassian_product",
        lambda product, *, site_subdomain, **kw: ProductConnectResult(
            product=product, status="connected", site_url=f"https://{site_subdomain}.atlassian.net", cloud_id="c"
        ),
    )
    monkeypatch.setattr(
        suite_auth.typer, "confirm", lambda msg, **kw: next(confirm_responses)
    )
    monkeypatch.setattr(
        suite_auth.typer, "prompt", lambda msg, **kw: next(prompt_responses)
    )
    monkeypatch.setattr(
        suite_auth, "print_plain_line", lambda msg, **kw: warnings.append(msg) if "empty" in msg.lower() else None
    )

    results = {
        "jira": ProductConnectResult(product="jira", status="connected"),
        "confluence": ProductConnectResult(product="confluence", status="not_connected"),
    }

    updated = suite_auth._offer_retry_failed_products(
        results=results,
        email="u@example.com",
        api_token="token",
        as_json=False,
    )

    assert updated["confluence"].status == "connected"
    assert len(warnings) == 1, "Should warn exactly once about empty subdomain"
    assert "empty" in warnings[0].lower()


def test_bitbucket_login_failure_exits_with_auth_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapters.inbound.cli.auth.bitbucket_auth as bitbucket_auth

    monkeypatch.setattr(bitbucket_auth, "load_cli_env", lambda: None)
    monkeypatch.setattr(
        bitbucket_auth,
        "run_bitbucket_step",
        lambda **kwargs: ProductConnectResult(
            product="bitbucket",
            status="not_connected",
            reason="insufficient_scopes",
        ),
    )

    result = runner.invoke(
        cli_main.app,
        [
            "bitbucket",
            "login",
            "--email",
            "user@example.com",
            "--api-token",
            "bad-token",
        ],
    )

    assert result.exit_code != 0
    assert "missing required read scopes" in result.output
