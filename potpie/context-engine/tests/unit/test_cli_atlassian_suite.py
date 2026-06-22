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
