from __future__ import annotations

from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main
from adapters.inbound.cli.auth import github as gh_auth
from adapters.inbound.cli import credentials_store as cs

pytestmark = pytest.mark.unit

runner = CliRunner()


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

    result = gh_auth.request_device_code(client=client)

    assert result.device_code == "dev-code"
    method, url, kwargs = client.calls[0]
    assert method == "POST"
    assert url == gh_auth.GITHUB_DEVICE_CODE_URL
    assert kwargs["headers"]["Accept"] == "application/json"
    assert kwargs["data"]["client_id"] == gh_auth.GITHUB_CLIENT_ID
    assert kwargs["data"]["scope"] == "repo read:org read:user user:email"


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
        client=client,
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
                json={"access_token": "gho_token", "token_type": "bearer", "scope": "repo"},
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
        client=client,
        sleep_fn=sleeps.append,
    )

    assert sleeps == [5, 10]


@pytest.mark.parametrize("error_code", ["expired_token", "access_denied"])
def test_poll_for_access_token_exits_cleanly_on_terminal_errors(error_code: str) -> None:
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
            client=client,
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

    account = gh_auth.verify_account("gho_token", client=client)

    assert account.login == "octocat"
    assert account.email == "octocat@example.com"
    assert client.calls[0][2]["headers"]["Authorization"] == "Bearer gho_token"


def test_verify_account_uses_primary_verified_email_when_user_email_is_null() -> None:
    client = FakeClient(
        [
            httpx.Response(
                200,
                json={"login": "octocat", "id": 1, "name": "The Octocat", "email": None},
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

    account = gh_auth.verify_account("gho_token", client=client)

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

    repos = gh_auth.list_user_owned_repositories("gho_token", client=client)

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
    success_providers: list[str] = []
    monkeypatch.setattr(
        cli_main.webbrowser,
        "open",
        lambda url: opened_urls.append(url) or True,
    )
    monkeypatch.setattr(
        cli_main,
        "open_cli_success_page",
        lambda *, provider="potpie": success_providers.append(provider),
    )
    monkeypatch.setattr(
        cli_main,
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
        cli_main,
        "poll_for_access_token",
        lambda _device: gh_auth.AccessToken(
            access_token="plaintext-token",
            token_type="bearer",
            scopes=["repo", "read:org", "read:user", "user:email"],
        ),
    )
    monkeypatch.setattr(
        cli_main,
        "verify_account",
        lambda _token: gh_auth.GitHubAccount(
            login="octocat", id=1, name="The Octocat", email="octocat@example.com"
        ),
    )

    result = runner.invoke(cli_main.app, ["auth", "github", "login"])

    assert result.exit_code == 0, result.stdout
    stored = cli_main.get_provider_credentials("github")
    assert stored["access_token"] == "plaintext-token"
    assert stored["account"]["login"] == "octocat"
    assert stored["account"]["email"] == "octocat@example.com"
    assert fake_keyring[("potpie", "github_token")] == "plaintext-token"
    raw = cs.read_credentials()
    assert "access_token" not in raw["integrations"]["github"]
    assert raw["integrations"]["github"]["account"] == {
        "login": "octocat",
        "id": 1,
        "name": "The Octocat",
        "email": "octocat@example.com",
    }
    assert raw["integrations"]["github"]["token_storage"] == "keychain"
    assert opened_urls == [gh_auth.GITHUB_VERIFICATION_URI]
    assert "Enter code: ABCD-EFGH" in result.stdout
    assert "Logged in to GitHub as octocat (octocat@example.com)" in result.stdout
    assert success_providers == ["github"]


def test_deprecated_git_login_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(cli_main, "_github_login_impl", lambda: None)
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
            "token_storage": "keychain",
            "account": {"login": "octocat", "id": 1},
        },
    )
    assert fake_keyring[("potpie", "github_token")] == "plaintext-token"

    result = runner.invoke(cli_main.app, ["auth", "github", "logout"])

    assert result.exit_code == 0, result.stdout
    assert cli_main.get_provider_credentials("github") == {}
    assert ("potpie", "github_token") not in fake_keyring
    assert "github" not in (cs.read_credentials().get("integrations") or {})


def test_git_login_does_not_store_when_account_verification_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        cli_main,
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
        cli_main,
        "poll_for_access_token",
        lambda _device: gh_auth.AccessToken(
            access_token="plaintext-token",
            token_type="bearer",
            scopes=["repo"],
        ),
    )

    def _fail(_token: str) -> gh_auth.GitHubAccount:
        raise gh_auth.GitHubDeviceFlowError("verification failed")

    monkeypatch.setattr(cli_main, "verify_account", _fail)

    result = runner.invoke(cli_main.app, ["auth", "github", "login"])

    assert result.exit_code == 1, result.stdout
    assert cli_main.get_provider_credentials("github") == {}


def test_git_test_repos_lists_stored_account_repositories(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cli_main.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "plaintext-token",
        },
    )
    monkeypatch.setattr(
        cli_main,
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

    result = runner.invoke(cli_main.app, ["github", "test", "repos"])

    assert result.exit_code == 0, result.stdout
    assert "octocat/widgets" in result.stdout
    assert "public" in result.stdout
    assert "main" in result.stdout


def test_git_test_repos_requires_stored_github_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    result = runner.invoke(cli_main.app, ["github", "test", "repos"])

    assert result.exit_code == 1
    assert "GitHub token not found in system keychain. Run: potpie auth github login" in result.output
