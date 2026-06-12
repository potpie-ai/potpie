from __future__ import annotations

import io
import sys
import types
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli as cli_main
from adapters.outbound.cli_auth import github as gh_auth
from adapters.inbound.cli.auth import github_commands as gh_cmds
from adapters.outbound.cli_auth import credentials_store as cs

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture(autouse=True)
def _default_linux_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs.sys, "platform", "linux")


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    store: dict[tuple[str, str], str] = {}

    def _set_password(service: str, username: str, password: str) -> None:
        store[(service, username)] = password

    def _get_password(service: str, username: str) -> str | None:
        return store.get((service, username))

    def _delete_password(service: str, username: str) -> None:
        store.pop((service, username), None)

    monkeypatch.setattr(cs.keyring, "set_password", _set_password)
    monkeypatch.setattr(cs.keyring, "get_password", _get_password)
    monkeypatch.setattr(cs.keyring, "delete_password", _delete_password)
    return store


@pytest.fixture(autouse=True)
def _github_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # The device flow requires POTPIE_GITHUB_CLIENT_ID (no hardcoded default).
    monkeypatch.setenv("POTPIE_GITHUB_CLIENT_ID", "Iv1.testclientid")
    monkeypatch.setattr(gh_auth, "load_cli_env", lambda: None)


class FakeClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append(("POST", url, kwargs))
        return self._responses.pop(0)

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append(("GET", url, kwargs))
        return self._responses.pop(0)

    def close(self) -> None:
        return


def test_request_device_code_sends_expected_payload() -> None:
    client = FakeClient(
        [
            httpx.Response(
                200,
                json={
                    "device_code": "dev-code",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 900,
                    "interval": 5,
                },
            )
        ]
    )

    result = gh_auth.request_device_code(http=client)

    assert result.device_code == "dev-code"
    method, url, kwargs = client.calls[0]
    assert method == "POST"
    assert url == gh_auth.GITHUB_DEVICE_CODE_URL
    assert kwargs["headers"]["Accept"] == "application/json"
    assert kwargs["data"]["client_id"] == gh_auth.get_github_client_id()
    assert kwargs["data"]["scope"] == "repo read:org read:user user:email"


def test_request_device_code_reads_client_id_from_env_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_GITHUB_CLIENT_ID", "env-client-id")
    client = FakeClient(
        [
            httpx.Response(
                200,
                json={
                    "device_code": "dev-code",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 900,
                    "interval": 5,
                },
            )
        ]
    )

    gh_auth.request_device_code(http=client)

    _method, _url, kwargs = client.calls[0]
    assert kwargs["data"]["client_id"] == "env-client-id"


def test_poll_for_access_token_waits_on_authorization_pending() -> None:
    client = FakeClient(
        [
            httpx.Response(200, json={"error": "authorization_pending"}),
            httpx.Response(
                200,
                json={
                    "access_token": "gho_token",
                    "token_type": "bearer",
                    "scope": "repo read:org read:user",
                },
            ),
        ]
    )
    sleeps: list[int] = []

    token = gh_auth.poll_for_access_token(
        gh_auth.DeviceCode(
            device_code="dev-code",
            user_code="ABCD-EFGH",
            verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
            expires_in=900,
            interval=5,
        ),
        http=client,
        sleep_fn=sleeps.append,
    )

    assert token.access_token == "gho_token"
    assert sleeps == [5, 5]


def test_poll_for_access_token_slow_down_increases_interval() -> None:
    client = FakeClient(
        [
            httpx.Response(200, json={"error": "slow_down"}),
            httpx.Response(
                200,
                json={
                    "access_token": "gho_token",
                    "token_type": "bearer",
                    "scope": "repo",
                },
            ),
        ]
    )
    sleeps: list[int] = []

    gh_auth.poll_for_access_token(
        gh_auth.DeviceCode(
            device_code="dev-code",
            user_code="ABCD-EFGH",
            verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
            expires_in=900,
            interval=5,
        ),
        http=client,
        sleep_fn=sleeps.append,
    )

    assert sleeps == [5, 10]


@pytest.mark.parametrize("error_code", ["expired_token", "access_denied"])
def test_poll_for_access_token_exits_cleanly_on_terminal_errors(
    error_code: str,
) -> None:
    client = FakeClient([httpx.Response(200, json={"error": error_code})])

    with pytest.raises(gh_auth.GitHubDeviceFlowError):
        gh_auth.poll_for_access_token(
            gh_auth.DeviceCode(
                device_code="dev-code",
                user_code="ABCD-EFGH",
                verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
                expires_in=900,
                interval=5,
            ),
            http=client,
            sleep_fn=lambda _seconds: None,
        )


def test_verify_account_parses_user() -> None:
    client = FakeClient(
        [
            httpx.Response(
                200,
                json={
                    "login": "octocat",
                    "id": 1,
                    "name": "The Octocat",
                    "email": "octocat@example.com",
                },
            )
        ]
    )

    account = gh_auth.verify_account("gho_token", http=client)

    assert account.login == "octocat"
    assert account.email == "octocat@example.com"
    assert client.calls[0][2]["headers"]["Authorization"] == "Bearer gho_token"


def test_verify_account_uses_primary_verified_email_when_user_email_is_null() -> None:
    client = FakeClient(
        [
            httpx.Response(
                200,
                json={
                    "login": "octocat",
                    "id": 1,
                    "name": "The Octocat",
                    "email": None,
                },
            ),
            httpx.Response(
                200,
                json=[
                    {
                        "email": "secondary@example.com",
                        "primary": False,
                        "verified": True,
                    },
                    {
                        "email": "octocat@example.com",
                        "primary": True,
                        "verified": True,
                    },
                ],
            ),
        ]
    )

    account = gh_auth.verify_account("gho_token", http=client)

    assert account.email == "octocat@example.com"
    assert client.calls[1][0] == "GET"
    assert client.calls[1][1] == gh_auth.GITHUB_USER_EMAILS_URL
    assert client.calls[1][2]["headers"]["Authorization"] == "Bearer gho_token"


def test_list_user_owned_repositories_uses_owner_affiliation() -> None:
    client = FakeClient(
        [
            httpx.Response(
                200,
                json=[
                    {
                        "id": 1,
                        "name": "widgets",
                        "full_name": "octocat/widgets",
                        "private": True,
                        "html_url": "https://github.com/octocat/widgets",
                        "default_branch": "main",
                        "updated_at": "2026-05-29T00:00:00Z",
                    }
                ],
            )
        ]
    )

    repos = gh_auth.list_user_owned_repositories("gho_token", http=client)

    assert repos == [
        {
            "id": 1,
            "name": "widgets",
            "full_name": "octocat/widgets",
            "private": True,
            "html_url": "https://github.com/octocat/widgets",
            "default_branch": "main",
            "updated_at": "2026-05-29T00:00:00Z",
        }
    ]
    method, url, kwargs = client.calls[0]
    assert method == "GET"
    assert url == gh_auth.GITHUB_USER_REPOS_URL
    assert kwargs["headers"]["Authorization"] == "Bearer gho_token"
    assert kwargs["params"]["affiliation"] == "owner"


def test_github_login_stores_token_only_after_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fake_keyring: dict[tuple[str, str], str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    opened_urls: list[str] = []
    monkeypatch.setattr(
        gh_cmds.webbrowser,
        "open",
        lambda url: opened_urls.append(url) or True,
    )
    waited: list[bool] = []
    monkeypatch.setattr(
        gh_cmds,
        "_wait_for_enter_or_auto_open",
        lambda: waited.append(True),
    )
    monkeypatch.setattr(
        gh_cmds,
        "request_device_code",
        lambda: gh_auth.DeviceCode(
            device_code="dev-code",
            user_code="ABCD-EFGH",
            verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
            expires_in=900,
            interval=5,
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "poll_for_access_token",
        lambda _device: gh_auth.AccessToken(
            access_token="plaintext-token",
            token_type="bearer",
            scopes=["repo", "read:org", "read:user", "user:email"],
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "verify_account",
        lambda _token: gh_auth.GitHubAccount(
            login="octocat", id=1, name="The Octocat", email="octocat@example.com"
        ),
    )

    result = runner.invoke(cli_main.app, ["auth", "github", "login"])

    assert result.exit_code == 0, result.stdout
    stored = cs.get_provider_credentials("github")
    assert stored["access_token"] == "plaintext-token"
    assert stored["account"]["login"] == "octocat"
    assert stored["account"]["email"] == "octocat@example.com"
    secrets = cs._read_integration_secrets_file()
    assert secrets["github_token"] == "plaintext-token"
    assert ("potpie", "github_token") not in fake_keyring
    raw = cs.read_credentials()
    assert "access_token" not in raw["integrations"]["github"]
    assert raw["integrations"]["github"]["account"] == {
        "login": "octocat",
        "id": 1,
        "name": "The Octocat",
        "email": "octocat@example.com",
    }
    assert raw["integrations"]["github"]["token_storage"] == "file"
    assert opened_urls == [gh_auth.GITHUB_VERIFICATION_URI]
    assert "GitHub login requires a one-time verification code." in result.stdout
    assert "Copy this code: ABCD-EFGH" in result.stdout
    assert "Paste the copied code into GitHub to continue." in result.stdout
    assert "Waiting for authorization" in result.stdout
    assert "Logged in to GitHub as octocat (octocat@example.com)" in result.stdout
    assert waited == [True]


def test_github_login_prints_verification_url_when_browser_open_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gh_cmds.webbrowser, "open", lambda _url: False)
    monkeypatch.setattr(
        gh_cmds,
        "_wait_for_enter_or_auto_open",
        lambda: None,
    )
    monkeypatch.setattr(
        gh_cmds,
        "request_device_code",
        lambda: gh_auth.DeviceCode(
            device_code="dev-code",
            user_code="ABCD-EFGH",
            verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
            expires_in=900,
            interval=5,
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "poll_for_access_token",
        lambda _device: gh_auth.AccessToken(
            access_token="plaintext-token",
            token_type="bearer",
            scopes=["repo"],
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "verify_account",
        lambda _token: gh_auth.GitHubAccount(
            login="octocat", id=1, name=None, email=None
        ),
    )

    result = runner.invoke(cli_main.app, ["auth", "github", "login"])

    assert result.exit_code == 0, result.stdout
    assert "Could not open a browser automatically. Open this URL:" in result.stdout
    assert gh_auth.GITHUB_VERIFICATION_URI in result.stdout
    assert "Paste the copied code into GitHub to continue." not in result.stdout


def test_wait_for_enter_or_auto_open_returns_when_enter_is_pressed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_stream = io.StringIO("\n")
    monkeypatch.setattr(
        gh_cmds.select,
        "select",
        lambda _read, _write, _error, _timeout: ([input_stream], [], []),
    )
    monkeypatch.setattr(
        gh_cmds.time,
        "sleep",
        lambda _seconds: pytest.fail("enter should skip the countdown sleep"),
    )

    gh_cmds._wait_for_enter_or_auto_open(seconds=10, input_stream=input_stream)

    out = capsys.readouterr().out
    assert (
        "Copy the code. Press Enter to open now, or GitHub opens in 10s"
        in out
    )
    assert "Opening GitHub now" not in out


def test_wait_for_enter_or_auto_open_times_out_on_same_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sleeps: list[int] = []
    monkeypatch.setattr(
        gh_cmds.select,
        "select",
        lambda _read, _write, _error, _timeout: ([], [], []),
    )
    monkeypatch.setattr(gh_cmds.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh_cmds._wait_for_enter_or_auto_open(seconds=2, input_stream=io.StringIO(""))

    out = capsys.readouterr().out
    assert "\r\033[KCopy the code. Press Enter to open now, or GitHub opens in 2s" in out
    assert "\r\033[KCopy the code. Press Enter to open now, or GitHub opens in 1s" in out
    assert "Opening GitHub now..." not in out
    assert sleeps == []


def test_wait_for_enter_or_auto_open_uses_msvcrt_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    kbhit_calls = {"count": 0}

    def _kbhit() -> bool:
        kbhit_calls["count"] += 1
        return kbhit_calls["count"] == 1

    fake_msvcrt = types.SimpleNamespace(kbhit=_kbhit, getwch=lambda: "\r")
    monkeypatch.setattr(gh_cmds.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(
        gh_cmds,
        "select",
        "select",
        lambda *_args, **_kwargs: pytest.fail("Windows path must not use select"),
    )

    gh_cmds._wait_for_enter_or_auto_open(seconds=10)

    out = capsys.readouterr().out
    assert (
        "Copy the code. Press Enter to open now, or GitHub opens in 10s"
        in out
    )
    assert "Opening GitHub now" not in out


def test_github_login_ctrl_c_at_enter_prompt_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        gh_cmds.webbrowser,
        "open",
        lambda _url: pytest.fail("cancelled login must not open the browser"),
    )

    def _cancel() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(gh_cmds, "_wait_for_enter_or_auto_open", _cancel)
    monkeypatch.setattr(
        gh_cmds,
        "request_device_code",
        lambda: gh_auth.DeviceCode(
            device_code="dev-code",
            user_code="ABCD-EFGH",
            verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
            expires_in=900,
            interval=5,
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "poll_for_access_token",
        lambda _device: pytest.fail("cancelled login must not poll for a token"),
    )

    result = runner.invoke(cli_main.app, ["auth", "github", "login"])

    assert result.exit_code == gh_cmds.EXIT_CANCELLED
    assert "Copy this code: ABCD-EFGH" in result.stdout
    assert "\nGitHub login cancelled." in result.stdout
    assert "Traceback" not in result.stdout
    assert "Abort" not in result.stdout
    assert "NameError" not in result.stdout
    assert cs.get_integration_metadata("github") == {}


def test_github_login_click_abort_from_prompt_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        gh_cmds,
        "_open_github_device_verification",
        lambda *_args: (_ for _ in ()).throw(gh_cmds.Abort()),
    )
    monkeypatch.setattr(
        gh_cmds,
        "request_device_code",
        lambda: gh_auth.DeviceCode(
            device_code="dev-code",
            user_code="ABCD-EFGH",
            verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
            expires_in=900,
            interval=5,
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "poll_for_access_token",
        lambda _device: pytest.fail("cancelled login must not poll for a token"),
    )

    result = runner.invoke(cli_main.app, ["auth", "github", "login"])

    assert result.exit_code == gh_cmds.EXIT_CANCELLED
    assert "\nGitHub login cancelled." in result.stdout
    assert "GitHub login failed" not in result.stdout
    assert "Unexpected error" not in result.stdout
    assert cs.get_integration_metadata("github") == {}


def test_github_login_abort_named_exception_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Abort(Exception):
        pass

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        gh_cmds,
        "_open_github_device_verification",
        lambda *_args: (_ for _ in ()).throw(Abort()),
    )
    monkeypatch.setattr(
        gh_cmds,
        "request_device_code",
        lambda: gh_auth.DeviceCode(
            device_code="dev-code",
            user_code="ABCD-EFGH",
            verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
            expires_in=900,
            interval=5,
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "poll_for_access_token",
        lambda _device: pytest.fail("cancelled login must not poll for a token"),
    )

    result = runner.invoke(cli_main.app, ["auth", "github", "login"])

    assert result.exit_code == gh_cmds.EXIT_CANCELLED
    assert "\nGitHub login cancelled." in result.stdout
    assert "GitHub login failed" not in result.stdout
    assert "Unexpected error" not in result.stdout


def test_github_login_json_mode_does_not_open_browser_or_print_countdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        gh_cmds.webbrowser,
        "open",
        lambda _url: pytest.fail("json mode must not open the browser"),
    )
    monkeypatch.setattr(
        gh_cmds,
        "request_device_code",
        lambda: gh_auth.DeviceCode(
            device_code="dev-code",
            user_code="ABCD-EFGH",
            verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
            expires_in=900,
            interval=5,
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "poll_for_access_token",
        lambda _device: gh_auth.AccessToken(
            access_token="plaintext-token",
            token_type="bearer",
            scopes=["repo"],
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "verify_account",
        lambda _token: gh_auth.GitHubAccount(
            login="octocat", id=1, name=None, email=None
        ),
    )

    result = runner.invoke(cli_main.app, ["--json", "auth", "github", "login"])

    assert result.exit_code == 0, result.stdout
    assert "Opening GitHub in" not in result.stdout
    assert "Copy this code" not in result.stdout
    assert '"provider": "github"' in result.stdout


def test_deprecated_git_login_alias(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gh_cmds, "github_login_impl", lambda: None)
    result = runner.invoke(cli_main.app, ["git", "login"])
    assert result.exit_code == 0, result.stdout


def test_github_logout_clears_github_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fake_keyring: dict[tuple[str, str], str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "plaintext-token",
            "account": {"login": "octocat", "id": 1},
        },
    )
    assert cs._read_integration_secrets_file()["github_token"] == "plaintext-token"

    result = runner.invoke(cli_main.app, ["auth", "github", "logout"])

    assert result.exit_code == 0, result.stdout
    assert cs.get_integration_metadata("github") == {}
    assert "github_token" not in cs._read_integration_secrets_file()
    assert ("potpie", "github_token") not in fake_keyring
    assert "github" not in (cs.read_credentials().get("integrations") or {})


def test_git_login_does_not_store_when_account_verification_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gh_cmds.webbrowser, "open", lambda _url: True)
    monkeypatch.setattr(gh_cmds.typer, "confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        gh_cmds,
        "request_device_code",
        lambda: gh_auth.DeviceCode(
            device_code="dev-code",
            user_code="ABCD-EFGH",
            verification_uri=gh_auth.GITHUB_VERIFICATION_URI,
            expires_in=900,
            interval=5,
        ),
    )
    monkeypatch.setattr(
        gh_cmds,
        "poll_for_access_token",
        lambda _device: gh_auth.AccessToken(
            access_token="plaintext-token",
            token_type="bearer",
            scopes=["repo"],
        ),
    )

    def _fail(_token: str) -> gh_auth.GitHubAccount:
        raise gh_auth.GitHubDeviceFlowError("verification failed")

    monkeypatch.setattr(gh_cmds, "verify_account", _fail)

    result = runner.invoke(cli_main.app, ["auth", "github", "login"])

    assert result.exit_code == 4, result.stdout
    assert cs.get_integration_metadata("github") == {}


def test_github_repos_lists_stored_account_repositories(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "plaintext-token",
        },
    )
    monkeypatch.setattr(
        gh_cmds,
        "list_user_owned_repositories",
        lambda token: [
            {
                "full_name": "octocat/widgets",
                "private": False,
                "default_branch": "main",
            }
        ]
        if token == "plaintext-token"
        else [],
    )

    result = runner.invoke(cli_main.app, ["github", "repos"])

    assert result.exit_code == 0, result.stdout
    assert "octocat/widgets" in result.stdout
    assert "public" in result.stdout
    assert "main" in result.stdout


def test_github_test_repos_deprecated_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "plaintext-token",
        },
    )
    monkeypatch.setattr(
        gh_cmds,
        "list_user_owned_repositories",
        lambda token: [{"full_name": "octocat/widgets", "private": False}],
    )

    result = runner.invoke(cli_main.app, ["github", "test", "repos"])

    assert result.exit_code == 0, result.stdout
    assert "octocat/widgets" in result.stdout


def test_github_repos_requires_stored_github_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    result = runner.invoke(cli_main.app, ["github", "repos"])

    assert result.exit_code == 4
    assert (
        f"GitHub token not found in {cs._integration_secret_store_label()}. "
        "Run: potpie github login"
        in result.output
    )


def test_potpie_login_delegates_to_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    monkeypatch.setattr(
        "adapters.inbound.cli.auth._login_impl.potpie_login_impl",
        lambda: called.append(True),
    )

    result = runner.invoke(cli_main.app, ["login"])

    assert result.exit_code == 0, result.stdout
    assert called == [True]


def test_verify_integration_access_github_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adapters.outbound.cli_auth.integration_verify import verify_integration_access

    class _Account:
        login = "octocat"
        email = "octo@example.com"

    monkeypatch.setattr(
        "adapters.outbound.cli_auth.github.verify_account",
        lambda _token, **_: _Account(),
    )

    ok, message = verify_integration_access(
        "github",
        {"access_token": "gho_test"},
    )
    assert ok is True
    assert "octocat" in message


def test_verify_integration_access_github_no_token() -> None:
    from adapters.outbound.cli_auth.integration_verify import verify_integration_access

    ok, message = verify_integration_access("github", {})
    assert ok is False
    assert message == "not authenticated"
