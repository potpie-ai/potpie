"""Unit tests for GitBucket CLI auth integration."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import typer
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli as cli_main
from adapters.inbound.cli.auth import gitbucket_commands as gb_cmds
from adapters.outbound.cli_auth import credentials_store as cs
from adapters.outbound.cli_auth.gitbucket_client import (
    GitBucketAccount,
    GitBucketClientError,
    gitbucket_api_base,
    gitbucket_token_page_url,
    normalize_gitbucket_host_url,
    verify_gitbucket_token,
)
from adapters.outbound.cli_auth.gitbucket_read_client import (
    GitBucketReadError,
    list_gitbucket_repos,
)
from adapters.outbound.cli_auth.http import AuthHttpError
from adapters.outbound.cli_auth.integration_profile import (
    build_gitbucket_integration_record,
    gitbucket_account_from_entry,
)
from adapters.outbound.cli_auth.integration_verify import verify_integration_access
from adapters.outbound.cli_auth.provider_config import (
    GITBUCKET_API_VERSION,
    GITBUCKET_TOKEN_PAGE_SUFFIX,
)

pytestmark = pytest.mark.unit

runner = CliRunner()


class FakeClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append(("GET", url, kwargs))
        if not self._responses:
            raise RuntimeError(f"No more fake responses for GET {url}")
        return self._responses.pop(0)

    def close(self) -> None:
        return


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gb_cmds, "load_cli_env", lambda: None)


# --- gitbucket_client.py ---


def test_normalize_gitbucket_host_url_variants() -> None:
    assert normalize_gitbucket_host_url("") == ""
    assert normalize_gitbucket_host_url("git.company.com") == "https://git.company.com"
    assert (
        normalize_gitbucket_host_url("https://git.company.com/gitbucket/")
        == "https://git.company.com/gitbucket"
    )
    assert normalize_gitbucket_host_url("http://192.168.1.1:8080/") == (
        "http://192.168.1.1:8080"
    )


def test_verify_gitbucket_token_rejects_remote_http_without_insecure_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_GITBUCKET_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("GITBUCKET_ALLOW_INSECURE_HTTP", raising=False)
    client = FakeClient([httpx.Response(200, json={"login": "alice"})])

    with pytest.raises(GitBucketClientError, match="plain HTTP is only allowed"):
        verify_gitbucket_token("http://192.168.1.1:8080", "secret-token", http=client)

    assert client.calls == []


def test_verify_gitbucket_token_allows_remote_http_with_insecure_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_GITBUCKET_ALLOW_INSECURE_HTTP", "1")
    client = FakeClient(
        [httpx.Response(200, json={"login": "alice", "email": "a@example.com"})]
    )

    account = verify_gitbucket_token(
        "http://192.168.1.1:8080",
        "secret-token",
        http=client,
    )

    assert account.login == "alice"
    assert client.calls


def test_verify_gitbucket_token_rejects_127_prefixed_non_loopback_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_GITBUCKET_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("GITBUCKET_ALLOW_INSECURE_HTTP", raising=False)
    client = FakeClient([httpx.Response(200, json={"login": "alice"})])

    with pytest.raises(GitBucketClientError, match="plain HTTP is only allowed"):
        verify_gitbucket_token("http://127.example.com:8080", "secret-token", http=client)

    assert client.calls == []


def test_gitbucket_api_base_and_token_page_url() -> None:
    host = "https://git.company.com/gitbucket"
    assert gitbucket_api_base(host) == f"{host}/api/{GITBUCKET_API_VERSION}"
    assert (
        gitbucket_token_page_url(host, "alice")
        == f"{host}/alice/{GITBUCKET_TOKEN_PAGE_SUFFIX}"
    )


def test_verify_gitbucket_token_success() -> None:
    client = FakeClient(
        [
            httpx.Response(
                200,
                json={
                    "login": "alice",
                    "email": "alice@example.com",
                    "site_admin": True,
                    "html_url": "http://localhost:8080/alice",
                },
            )
        ]
    )

    account = verify_gitbucket_token(
        "http://localhost:8080",
        "secret-token",
        http=client,
    )

    assert account == GitBucketAccount(
        login="alice",
        email="alice@example.com",
        site_admin=True,
        html_url="http://localhost:8080/alice",
    )
    method, url, kwargs = client.calls[0]
    assert method == "GET"
    assert url == "http://localhost:8080/api/v3/user"
    assert kwargs["headers"]["Authorization"] == "token secret-token"


def test_verify_gitbucket_token_401() -> None:
    client = FakeClient([httpx.Response(401, json={"message": "Requires authentication"})])

    with pytest.raises(GitBucketClientError, match="Authentication failed") as exc:
        verify_gitbucket_token("http://localhost:8080", "bad-token", http=client)

    assert exc.value.status_code == 401


def test_verify_gitbucket_token_404() -> None:
    client = FakeClient([httpx.Response(404)])

    with pytest.raises(GitBucketClientError, match="GitBucket API not found") as exc:
        verify_gitbucket_token("http://localhost:8080", "tok", http=client)

    assert exc.value.status_code == 404


def test_verify_gitbucket_token_non_json_response() -> None:
    client = FakeClient([httpx.Response(200, text="not-json")])

    with pytest.raises(GitBucketClientError, match="non-JSON"):
        verify_gitbucket_token("http://localhost:8080", "tok", http=client)


def test_verify_gitbucket_token_missing_login() -> None:
    client = FakeClient([httpx.Response(200, json={"email": "a@example.com"})])

    with pytest.raises(GitBucketClientError, match="missing 'login'"):
        verify_gitbucket_token("http://localhost:8080", "tok", http=client)


def test_verify_gitbucket_token_transport_error() -> None:
    class FailingClient:
        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            raise AuthHttpError("connection refused")

        def close(self) -> None:
            return

    with pytest.raises(GitBucketClientError, match="Could not reach GitBucket"):
        verify_gitbucket_token("http://localhost:8080", "tok", http=FailingClient())


# --- gitbucket_read_client.py ---


def test_list_gitbucket_repos_with_explicit_credentials() -> None:
    client = FakeClient(
        [
            httpx.Response(
                200,
                json=[
                    {
                        "full_name": "alice/widgets",
                        "name": "widgets",
                        "owner": {"login": "alice"},
                        "private": False,
                        "description": "Demo repo",
                        "default_branch": "main",
                        "html_url": "http://localhost:8080/alice/widgets",
                        "clone_url": "http://localhost:8080/git/alice/widgets.git",
                    }
                ],
            )
        ]
    )

    repos = list_gitbucket_repos(
        host_url="http://localhost:8080",
        token="secret-token",
        limit=10,
        http=client,
    )

    assert len(repos) == 1
    assert repos[0]["full_name"] == "alice/widgets"
    assert repos[0]["default_branch"] == "main"
    _method, url, _kwargs = client.calls[0]
    assert url == "http://localhost:8080/api/v3/user/repos?per_page=10"


def test_list_gitbucket_repos_handles_non_dict_owner() -> None:
    client = FakeClient(
        [
            httpx.Response(
                200,
                json=[
                    {
                        "full_name": "alice/widgets",
                        "name": "widgets",
                        "owner": "alice",
                        "private": False,
                    }
                ],
            )
        ]
    )

    repos = list_gitbucket_repos(
        host_url="http://localhost:8080",
        token="secret-token",
        http=client,
    )

    assert repos == [
        {
            "full_name": "alice/widgets",
            "name": "widgets",
            "owner": "",
            "private": False,
            "description": "",
            "default_branch": "",
            "html_url": "",
            "clone_url": "",
        }
    ]


def test_list_gitbucket_repos_non_list_response_raises() -> None:
    client = FakeClient([httpx.Response(200, json={"message": "unexpected"})])

    with pytest.raises(
        GitBucketReadError,
        match="unexpected response format for /user/repos",
    ):
        list_gitbucket_repos(
            host_url="http://localhost:8080",
            token="secret-token",
            http=client,
        )


def test_list_gitbucket_repos_rejects_insecure_http_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_GITBUCKET_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.delenv("GITBUCKET_ALLOW_INSECURE_HTTP", raising=False)
    client = FakeClient([httpx.Response(200, json=[])])

    with pytest.raises(GitBucketReadError, match="plain HTTP is only allowed"):
        list_gitbucket_repos(
            host_url="http://192.168.1.1:8080",
            token="secret-token",
            http=client,
        )

    assert client.calls == []


def test_list_gitbucket_repos_requires_stored_credentials() -> None:
    with pytest.raises(GitBucketReadError, match="not connected"):
        list_gitbucket_repos()


def test_list_gitbucket_repos_rejects_partial_explicit_credentials() -> None:
    with pytest.raises(
        GitBucketReadError,
        match="host_url and token must be provided together",
    ):
        list_gitbucket_repos(host_url="http://localhost:8080")

    with pytest.raises(
        GitBucketReadError,
        match="host_url and token must be provided together",
    ):
        list_gitbucket_repos(token="secret-token")


def test_list_gitbucket_repos_401() -> None:
    client = FakeClient([httpx.Response(401)])

    with pytest.raises(GitBucketReadError, match="authentication failed"):
        list_gitbucket_repos(
            host_url="http://localhost:8080",
            token="bad-token",
            http=client,
        )


# --- credentials_store.py (GitBucket) ---


def test_gitbucket_credentials_roundtrip() -> None:
    cs.save_gitbucket_credentials(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "email": "alice@example.com",
            "token": "gb-token",
        }
    )

    creds = cs.get_gitbucket_credentials()
    assert creds["token"] == "gb-token"
    assert creds["host_url"] == "http://localhost:8080"
    assert creds["account"]["login"] == "alice"
    assert creds["auth_type"] == "personal_access_token"

    cs.clear_gitbucket_credentials()
    assert cs.get_gitbucket_credentials() == {}


def test_save_gitbucket_credentials_requires_token() -> None:
    with pytest.raises(cs.ProviderCredentialError, match="token is required"):
        cs.save_gitbucket_credentials({"host_url": "http://localhost:8080", "token": "  "})


def test_save_gitbucket_credentials_requires_host_url() -> None:
    with pytest.raises(cs.ProviderCredentialError, match="host URL is required"):
        cs.save_gitbucket_credentials({"token": "gb-token"})


def test_get_integration_tokens_gitbucket() -> None:
    cs.save_gitbucket_credentials(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "token": "gb-token",
        }
    )

    tokens = cs.get_integration_tokens("gitbucket")
    assert tokens["auth_type"] == "personal_access_token"
    assert tokens["token"] == "gb-token"


def test_clear_integration_tokens_gitbucket() -> None:
    cs.save_gitbucket_credentials(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "token": "gb-token",
        }
    )

    cs.clear_integration_tokens("gitbucket")
    assert cs.get_gitbucket_credentials() == {}


def test_get_integration_status_gitbucket_unauthenticated() -> None:
    status = cs.get_integration_status("gitbucket")
    assert status["authenticated"] is False
    assert status["auth_type"] == "personal_access_token"


def test_get_integration_status_gitbucket_authenticated() -> None:
    cs.save_gitbucket_credentials(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "email": "alice@example.com",
            "token": "gb-token",
        }
    )

    status = cs.get_integration_status("gitbucket")
    assert status["authenticated"] is True
    assert status["login"] == "alice"
    assert status["email"] == "alice@example.com"
    assert status["host_url"] == "http://localhost:8080"


def test_list_integration_providers_includes_gitbucket() -> None:
    cs.save_gitbucket_credentials(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "token": "gb-token",
        }
    )

    providers = cs.list_integration_providers()
    assert "gitbucket" in providers


# --- integration_profile.py ---


def test_build_gitbucket_integration_record() -> None:
    record = build_gitbucket_integration_record(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "email": "alice@example.com",
        }
    )
    assert record["provider"] == "gitbucket"
    assert record["auth_type"] == "personal_access_token"
    assert record["account"]["login"] == "alice"


def test_gitbucket_account_from_entry_nested_and_flat() -> None:
    assert gitbucket_account_from_entry(
        {"account": {"login": "alice", "email": "a@example.com"}}
    ) == {"login": "alice", "email": "a@example.com"}
    assert gitbucket_account_from_entry({"login": "bob"}) == {"login": "bob"}


# --- integration_verify.py ---


def test_verify_integration_access_gitbucket_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "adapters.outbound.cli_auth.gitbucket_client.verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(
            login="alice",
            email="alice@example.com",
        ),
    )

    ok, message = verify_integration_access(
        "gitbucket",
        {"host_url": "http://localhost:8080", "token": "gb-token"},
    )

    assert ok is True
    assert "alice" in message
    assert "localhost:8080" in message


def test_verify_integration_access_gitbucket_not_authenticated() -> None:
    ok, message = verify_integration_access("gitbucket", {})
    assert ok is False
    assert message == "not authenticated"


def test_verify_integration_access_gitbucket_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(host: str, token: str, **_: Any) -> GitBucketAccount:
        raise GitBucketClientError("Authentication failed.", status_code=401)

    monkeypatch.setattr(
        "adapters.outbound.cli_auth.gitbucket_client.verify_gitbucket_token",
        _fail,
    )

    ok, message = verify_integration_access(
        "gitbucket",
        {"host_url": "http://localhost:8080", "token": "bad"},
    )

    assert ok is False
    assert "Authentication failed" in message


# --- gitbucket_commands.py ---


def test_run_gitbucket_auth_success_with_supplied_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        gb_cmds,
        "verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(
            login="alice",
            email="alice@example.com",
        ),
    )
    printed: list[dict] = []
    monkeypatch.setattr(
        gb_cmds,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )

    gb_cmds.run_gitbucket_api_token_auth(
        force=True,
        as_json=True,
        host="http://localhost:8080",
        token="gb-token",
    )

    assert printed
    assert printed[-1].get("ok") is True
    assert printed[-1].get("provider") == "gitbucket"
    assert printed[-1].get("login") == "alice"

    creds = cs.get_gitbucket_credentials()
    assert creds["token"] == "gb-token"
    assert creds["host_url"] == "http://localhost:8080"


def test_run_gitbucket_auth_already_connected_skips_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cs.save_gitbucket_credentials(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "token": "gb-token",
        }
    )
    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: True)
    printed: list[dict] = []
    monkeypatch.setattr(
        gb_cmds,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )

    gb_cmds.run_gitbucket_api_token_auth(as_json=True)

    assert printed
    assert printed[-1].get("already_connected") is True
    assert printed[-1].get("login") == "alice"


def test_run_gitbucket_auth_already_connected_uses_top_level_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[dict] = []

    class _Store:
        def get_gitbucket_credentials(self) -> dict[str, Any]:
            return {
                "host_url": "http://localhost:8080",
                "login": "alice",
                "token": "gb-token",
            }

    monkeypatch.setattr(gb_cmds, "get_store", lambda: _Store())
    monkeypatch.setattr(
        gb_cmds,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )

    gb_cmds.run_gitbucket_api_token_auth(as_json=True)

    assert printed[-1].get("login") == "alice"


def test_run_gitbucket_auth_rejects_non_tty_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: False)
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gb_cmds,
        "emit_error",
        lambda title, detail, **_: captured.append((title, detail)),
    )

    with pytest.raises(typer.Exit) as exc:
        gb_cmds.run_gitbucket_api_token_auth(force=True)

    assert exc.value.exit_code == 1
    assert captured
    assert "gitbucket_token" in captured[0][1].lower()


def test_run_gitbucket_auth_verification_failure_exits_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: False)

    def _fail(host: str, token: str, **_: Any) -> GitBucketAccount:
        raise GitBucketClientError("Authentication failed.", status_code=401)

    monkeypatch.setattr(gb_cmds, "verify_gitbucket_token", _fail)
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gb_cmds,
        "emit_error",
        lambda title, detail, **_: captured.append((title, detail)),
    )

    with pytest.raises(typer.Exit) as exc:
        gb_cmds.run_gitbucket_api_token_auth(
            force=True,
            host="http://localhost:8080",
            token="bad-token",
        )

    assert exc.value.exit_code == gb_cmds.EXIT_AUTH
    assert captured
    assert "Authentication failed" in captured[0][1]


def test_gitbucket_login_cli_non_interactive_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITBUCKET_TOKEN", "gb-token")
    monkeypatch.setattr(
        gb_cmds,
        "verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(login="alice", email="a@example.com"),
    )

    result = runner.invoke(
        cli_main.app,
        [
            "gitbucket",
            "login",
            "--host",
            "http://localhost:8080",
            "--force",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Logged in to GitBucket as alice" in result.output
    assert cs.get_gitbucket_credentials()["token"] == "gb-token"


def test_gitbucket_login_cli_non_interactive_via_token_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gb_cmds,
        "verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(login="alice", email="a@example.com"),
    )

    result = runner.invoke(
        cli_main.app,
        [
            "gitbucket",
            "login",
            "--host",
            "http://localhost:8080",
            "--token",
            "gb-token",
            "--force",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Logged in to GitBucket as alice" in result.output
    assert cs.get_gitbucket_credentials()["token"] == "gb-token"


def test_run_gitbucket_auth_non_interactive_via_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_calls = 0

    class _PipedStdin:
        def isatty(self) -> bool:
            return False

        def read(self) -> str:
            nonlocal read_calls
            read_calls += 1
            if read_calls > 1:
                raise AssertionError("stdin.read() must only be called once")
            return "gb-token\n"

    monkeypatch.delenv("GITBUCKET_TOKEN", raising=False)
    monkeypatch.delenv("POTPIE_GITBUCKET_TOKEN", raising=False)
    monkeypatch.setattr(gb_cmds.sys, "stdin", _PipedStdin())
    monkeypatch.setattr(
        gb_cmds,
        "verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(login="alice", email="a@example.com"),
    )

    gb_cmds.run_gitbucket_api_token_auth(
        force=True,
        host="http://localhost:8080",
    )

    assert read_calls == 1
    assert cs.get_gitbucket_credentials()["token"] == "gb-token"


def test_gitbucket_logout_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cs.save_gitbucket_credentials(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "token": "gb-token",
        }
    )

    result = runner.invoke(cli_main.app, ["gitbucket", "logout"])

    assert result.exit_code == 0, result.output
    assert "Logged out successfully" in result.output
    assert cs.get_gitbucket_credentials() == {}


def test_gitbucket_repos_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cs.save_gitbucket_credentials(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "token": "gb-token",
        }
    )
    monkeypatch.setattr(
        gb_cmds,
        "list_gitbucket_repos",
        lambda **_: [
            {
                "full_name": "alice/widgets",
                "private": False,
                "default_branch": "main",
            }
        ],
    )

    result = runner.invoke(cli_main.app, ["gitbucket", "repos"])

    assert result.exit_code == 0, result.output
    assert "alice/widgets" in result.output
    assert "public" in result.output
    assert "main" in result.output


def test_gitbucket_repos_requires_connection() -> None:
    result = runner.invoke(cli_main.app, ["gitbucket", "repos"])

    assert result.exit_code == gb_cmds.EXIT_UNAVAILABLE
    assert "not connected" in result.output.lower()


def test_guard_typer_prompt_maps_click_abort_to_keyboard_interrupt() -> None:
    import click

    def _raise_abort() -> str:
        raise click.Abort()

    with pytest.raises(KeyboardInterrupt):
        gb_cmds._guard_typer_prompt(_raise_abort)


def test_open_token_page_declined_skips_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(gb_cmds, "_guard_typer_prompt", lambda callback: False)
    monkeypatch.setattr(
        gb_cmds,
        "print_plain_line",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(gb_cmds.webbrowser, "open", lambda url, **_: opened.append(url))

    gb_cmds._open_token_page("http://localhost:8080")

    assert opened == []


def test_open_token_page_opens_browser_when_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(gb_cmds, "_guard_typer_prompt", lambda callback: True)
    monkeypatch.setattr(gb_cmds, "_prompt_gitbucket_login", lambda: "alice")
    monkeypatch.setattr(
        gb_cmds,
        "print_plain_line",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        gb_cmds.webbrowser,
        "open",
        lambda url, **_: opened.append(url) or True,
    )

    gb_cmds._open_token_page("http://localhost:8080")

    assert opened == ["http://localhost:8080/alice/_application"]


def test_open_token_page_prints_url_when_browser_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(gb_cmds, "_guard_typer_prompt", lambda callback: True)
    monkeypatch.setattr(gb_cmds, "_prompt_gitbucket_login", lambda: "alice")
    monkeypatch.setattr(
        gb_cmds,
        "print_plain_line",
        lambda message, **_: printed.append(str(message)),
    )
    monkeypatch.setattr(gb_cmds.webbrowser, "open", lambda url, **_: False)

    gb_cmds._open_token_page("http://localhost:8080")

    assert "Could not open a browser. Open this URL:" in printed
    assert "http://localhost:8080/alice/_application" in printed


def test_run_gitbucket_auth_json_mode_emits_token_page_when_browser_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(gb_cmds, "_prompt_gitbucket_login", lambda: "alice")
    monkeypatch.setattr(gb_cmds, "_prompt_token", lambda: "gb-token")
    monkeypatch.setattr(gb_cmds.webbrowser, "open", lambda url, **_: False)
    monkeypatch.setattr(
        gb_cmds,
        "verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(login="alice"),
    )

    gb_cmds.run_gitbucket_api_token_auth(
        force=True,
        as_json=True,
        host="http://localhost:8080",
    )

    out = capsys.readouterr().out
    assert '"action": "open_token_page"' in out
    assert '"token_page_url": "http://localhost:8080/alice/_application"' in out
    assert '"browser_opened": false' in out


def test_run_gitbucket_auth_interactive_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(gb_cmds, "_prompt_host_url", lambda: "http://localhost:8080")
    monkeypatch.setattr(gb_cmds, "_prompt_token", lambda: "gb-token")
    monkeypatch.setattr(gb_cmds, "_open_token_page", lambda _host: None)
    monkeypatch.setattr(
        gb_cmds,
        "verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(login="alice"),
    )
    printed: list[str] = []
    monkeypatch.setattr(
        gb_cmds,
        "print_plain_line",
        lambda message, **_: printed.append(str(message)),
    )

    gb_cmds.run_gitbucket_api_token_auth(force=True)

    assert any("Logged in to GitBucket as alice" in line for line in printed)


def test_run_gitbucket_auth_interactive_flow_reuses_cli_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompted_host = False

    def _unexpected_host_prompt() -> str:
        nonlocal prompted_host
        prompted_host = True
        return "http://should-not-be-used"

    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(gb_cmds, "_prompt_host_url", _unexpected_host_prompt)
    monkeypatch.setattr(gb_cmds, "_prompt_token", lambda: "gb-token")
    monkeypatch.setattr(gb_cmds, "_open_token_page", lambda _host: None)
    monkeypatch.setattr(
        gb_cmds,
        "verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(login="alice"),
    )

    gb_cmds.run_gitbucket_api_token_auth(
        force=True,
        host="http://localhost:8080",
    )

    assert not prompted_host
    assert cs.get_gitbucket_credentials()["host_url"] == "http://localhost:8080"


def test_run_gitbucket_auth_empty_host_in_interactive_mode_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(gb_cmds, "_prompt_host_url", lambda: "   ")
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gb_cmds,
        "emit_error",
        lambda title, detail, **_: captured.append((title, detail)),
    )

    with pytest.raises(typer.Exit) as exc:
        gb_cmds.run_gitbucket_api_token_auth(force=True)

    assert exc.value.exit_code == gb_cmds.EXIT_AUTH
    assert captured
    assert "Host URL must not be empty" in captured[0][1]


def test_run_gitbucket_auth_credential_lookup_error_continues_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: False)

    class _BrokenLookupStore:
        def get_gitbucket_credentials(self) -> dict[str, str]:
            raise cs.ProviderCredentialError("lookup failed")

        def save_gitbucket_credentials(self, credentials: dict[str, Any]) -> None:
            cs.save_gitbucket_credentials(credentials)

    monkeypatch.setattr(gb_cmds, "get_store", lambda: _BrokenLookupStore())
    monkeypatch.setattr(
        gb_cmds,
        "verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(login="alice"),
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gb_cmds,
        "print_plain_line",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        gb_cmds,
        "emit_error",
        lambda title, detail, **_: captured.append((title, detail)),
    )

    gb_cmds.run_gitbucket_api_token_auth(
        force=True,
        host="http://localhost:8080",
        token="gb-token",
    )

    assert captured
    assert "lookup failed" in captured[0][1]
    assert cs.get_gitbucket_credentials()["token"] == "gb-token"


def test_run_gitbucket_auth_storage_failure_exits_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gb_cmds.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        gb_cmds,
        "verify_gitbucket_token",
        lambda host, token, **_: GitBucketAccount(login="alice"),
    )

    class _BrokenStore:
        def get_gitbucket_credentials(self) -> dict[str, str]:
            return {}

        def save_gitbucket_credentials(self, credentials: dict[str, Any]) -> None:
            raise cs.ProviderCredentialError("storage failed")

    monkeypatch.setattr(gb_cmds, "get_store", lambda: _BrokenStore())
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gb_cmds,
        "emit_error",
        lambda title, detail, **_: captured.append((title, detail)),
    )

    with pytest.raises(typer.Exit) as exc:
        gb_cmds.run_gitbucket_api_token_auth(
            force=True,
            host="http://localhost:8080",
            token="gb-token",
        )

    assert exc.value.exit_code == gb_cmds.EXIT_AUTH
    assert captured
    assert "storage failed" in captured[0][1]


def test_gitbucket_logout_json_when_not_authenticated() -> None:
    result = runner.invoke(cli_main.app, ["--json", "gitbucket", "logout"])

    assert result.exit_code == 0, result.output
    assert '"cleared_stale": true' in result.output


def test_gitbucket_logout_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenStore:
        def clear_gitbucket_credentials(self) -> None:
            raise cs.ProviderCredentialError("logout failed")

    monkeypatch.setattr(gb_cmds, "get_store", lambda: _BrokenStore())

    result = runner.invoke(cli_main.app, ["gitbucket", "logout"])

    assert result.exit_code == gb_cmds.EXIT_AUTH
    assert "logout failed" in result.output


def test_gitbucket_repos_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cs.save_gitbucket_credentials(
        {
            "host_url": "http://localhost:8080",
            "login": "alice",
            "token": "gb-token",
        }
    )
    monkeypatch.setattr(
        gb_cmds,
        "list_gitbucket_repos",
        lambda **_: [{"full_name": "alice/widgets", "private": False}],
    )

    result = runner.invoke(cli_main.app, ["--json", "gitbucket", "repos"])

    assert result.exit_code == 0, result.output
    assert '"provider": "gitbucket"' in result.output
    assert "alice/widgets" in result.output


def test_verify_gitbucket_token_unexpected_http_status() -> None:
    client = FakeClient([httpx.Response(500)])

    with pytest.raises(GitBucketClientError, match="HTTP 500") as exc:
        verify_gitbucket_token("http://localhost:8080", "tok", http=client)

    assert exc.value.status_code == 500


def test_verify_gitbucket_token_unexpected_payload_shape() -> None:
    client = FakeClient([httpx.Response(200, json=["not", "a", "dict"])])

    with pytest.raises(GitBucketClientError, match="unexpected response format"):
        verify_gitbucket_token("http://localhost:8080", "tok", http=client)


def test_list_gitbucket_repos_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            raise AuthHttpError("connection refused")

        def close(self) -> None:
            return

    with pytest.raises(GitBucketReadError, match="request failed"):
        list_gitbucket_repos(
            host_url="http://localhost:8080",
            token="gb-token",
            http=FailingClient(),
        )


def test_list_gitbucket_repos_non_json_response() -> None:
    client = FakeClient([httpx.Response(200, text="not-json")])

    with pytest.raises(GitBucketReadError, match="non-JSON"):
        list_gitbucket_repos(
            host_url="http://localhost:8080",
            token="gb-token",
            http=client,
        )


def test_list_gitbucket_repos_http_status_error() -> None:
    client = FakeClient([httpx.Response(500)])

    with pytest.raises(GitBucketReadError, match="HTTP 500"):
        list_gitbucket_repos(
            host_url="http://localhost:8080",
            token="gb-token",
            http=client,
        )


def test_integration_auth_provider_accepts_gitbucket() -> None:
    from adapters.inbound.cli.auth import auth_commands

    assert auth_commands._integration_auth_provider("gitbucket") == "gitbucket"
    assert auth_commands._integration_auth_provider(" GitBucket ") == "gitbucket"


def test_run_integration_login_gitbucket_routes_to_gitbucket_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adapters.inbound.cli.auth import auth_commands

    calls: list[dict[str, object]] = []

    def _fake_gitbucket_login(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        auth_commands,
        "ensure_runtime_environment_loaded",
        lambda: None,
    )
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.gitbucket_commands.run_gitbucket_api_token_auth",
        _fake_gitbucket_login,
    )

    auth_commands.run_integration_login("gitbucket", force=True)

    assert calls == [{"force": True, "as_json": False, "verbose": False}]
