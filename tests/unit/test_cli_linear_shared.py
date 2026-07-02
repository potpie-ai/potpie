"""Unit tests for shared CLI plumbing (Linear OAuth PR)."""

from __future__ import annotations

from __future__ import annotations
import typer
import pytest
from typer.testing import CliRunner
from potpie.cli.auth import auth_commands
from potpie.cli import host_cli as cli_main
from unittest.mock import MagicMock, patch
import json
import stat
from pathlib import Path
from potpie.cli.auth import credentials_store as cs
from potpie.cli.auth.integration_profile import (
    build_linear_integration_record,
    fetch_linear_viewer,
)
from potpie.cli.auth.http import AuthHttpError
from potpie.cli.commands._common import set_store
from tests._auth_fakes import InMemoryCredentialStore
from potpie.cli.auth.integration_verify import (
    _verify_linear,
    verify_integration_access,
)
from potpie.cli.auth.provider_config import (
    LINEAR_TOKEN_URL,
    authorization_url,
    get_callback_host,
    get_callback_path,
    get_callback_port,
    get_client_id,
    get_client_secret,
    get_redirect_uri,
    get_scopes,
    token_url,
)
# --- test_auth_commands.py ---

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_xdg_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def test_auth_help_is_wired_into_main_cli() -> None:
    result = runner.invoke(cli_main.app, ["auth", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "Deprecated" in result.stdout


def test_status_routes_to_integration_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    def _integration_status(*, verify: bool = False) -> None:
        called.append(verify)

    monkeypatch.setattr(
        auth_commands,
        "integration_status",
        _integration_status,
    )

    result = runner.invoke(cli_main.app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert called == [False]

    result = runner.invoke(cli_main.app, ["status", "--verify"])
    assert result.exit_code == 0, result.stdout
    assert called == [False, True]


def test_register_provider_app_normalizes_name(monkeypatch) -> None:
    calls: list[tuple[typer.Typer, str]] = []

    def add_typer(provider_app: typer.Typer, *, name: str) -> None:
        calls.append((provider_app, name))

    monkeypatch.setattr(auth_commands.auth_app, "add_typer", add_typer)

    provider_app = typer.Typer()
    auth_commands.register_provider_app(" Example ", provider_app)

    assert calls == [(provider_app, "example")]


def test_register_provider_app_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="provider app name must be non-empty"):
        auth_commands.register_provider_app(" ", typer.Typer())


def test_esc_handles_none_and_markup() -> None:
    from rich.markup import escape

    assert auth_commands._esc(None) == ""
    raw = "[bold]x[/bold]"
    assert auth_commands._esc(raw) == escape(raw)


def test_auth_status_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "auth_type": "oauth" if provider == "linear" else "api_token",
        },
    )
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))

    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0, result.stdout
    assert '"integrations"' in result.stdout
    assert '"linear"' in result.stdout


def test_auth_logout_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    result = runner.invoke(auth_commands.auth_app, ["logout", "unknown"])

    assert result.exit_code == 1
    assert captured
    assert "Unknown provider" in captured[0][0]


def test_flags_delegates_to_common(monkeypatch: pytest.MonkeyPatch) -> None:
    from potpie.cli.commands import _common

    monkeypatch.setattr(_common, "is_json", lambda: True)
    monkeypatch.setattr(_common, "is_verbose", lambda: True)
    assert auth_commands._flags() == (True, True)


def test_token_is_expired_invalid_expires_at() -> None:
    assert auth_commands._token_is_expired("not-a-number") is False
    assert auth_commands._token_is_expired(0.0) is True


def test_auth_logout_not_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    result = runner.invoke(auth_commands.auth_app, ["logout", "linear"])

    assert result.exit_code == 1
    assert captured
    assert "not authenticated" in captured[0][0]


def test_auth_revoke_delegates_to_logout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True},
    )
    set_store(InMemoryCredentialStore())

    result = runner.invoke(auth_commands.auth_app, ["revoke", "linear"])

    assert result.exit_code == 0
    assert '"ok": true' in result.stdout


def test_auth_logout_clear_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from potpie.cli.auth.credentials_store import ProviderCredentialError

    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, True))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True},
    )

    store = InMemoryCredentialStore()

    def _fail_clear(_provider: str) -> None:
        raise ProviderCredentialError("keychain broke")

    store.clear_integration_tokens = _fail_clear  # type: ignore[method-assign]
    set_store(store)

    result = runner.invoke(auth_commands.auth_app, ["logout", "linear"])

    assert result.exit_code == 4


# --- test_auth_commands_cli.py ---

runner = CliRunner()


def _mock_cli(monkeypatch: pytest.MonkeyPatch, *, json_mode: bool = False) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (json_mode, False))


def test_auth_status_verify_token_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cli(monkeypatch, json_mode=True)
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": True,
            "auth_type": "oauth",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: (_ for _ in ()).throw(RuntimeError("refresh broke")),
    )
    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])
    assert result.exit_code == 0
    assert '"verified": false' in result.stdout
    assert "refresh broke" in result.stdout


def test_wait_for_callback_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from potpie.cli.auth.callback_server import OAuthCallbackResult

    monkeypatch.setattr(
        auth_commands,
        "wait_for_oauth_callback",
        lambda **kwargs: OAuthCallbackResult(code="c", state="s"),
    )
    result = auth_commands._wait_for_callback(
        host="localhost", port=8080, path="/callback"
    )
    assert result.code == "c"


def test_linear_login_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        auth_commands,
        "_run_linear_oauth_flow",
        lambda force=False, add=False: called.append((force, add)),
    )
    runner.invoke(auth_commands.linear_app, ["login", "--force"])
    assert called == [(True, False)]


# --- test_auth_commands_helpers.py ---

runner = CliRunner()


def test_build_linear_authorization_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(auth_commands, "get_scopes", lambda _p: "read")
    url = auth_commands._build_linear_authorization_url(
        redirect_uri="http://localhost:8080/callback",
        state="state-1",
        code_challenge="challenge",
    )
    assert "client_id=client-id" in url
    assert "code_challenge=challenge" in url
    assert "linear.app/oauth/authorize" in url


def test_token_is_expired_helper() -> None:
    import time as time_module

    assert auth_commands._token_is_expired(time_module.time() + 3600) is False
    assert auth_commands._token_is_expired(time_module.time() - 10) is True
    assert auth_commands._token_is_expired(None) is False


def test_try_refresh_linear_session_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: {"access_token": "tok", "expires_at": 9999999999.0},
    )
    assert auth_commands._try_refresh_linear_session() is True


def test_try_refresh_linear_session_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )
    assert auth_commands._try_refresh_linear_session() is False


def test_handle_already_connected_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    printed: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )
    auth_commands._handle_already_connected(
        "linear",
        {"auth_type": "oauth", "expires_at": 1.0, "cloud_id": "c1"},
    )
    assert printed[-1].get("already_connected") is True


# --- test_auth_commands_status.py ---

runner = CliRunner()


def test_auth_status_human_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Acme",
            "auth_type": "oauth",
        },
    )
    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0
    assert "linear: authenticated" in result.stdout
    assert "github: not authenticated" in result.stdout


def test_auth_status_includes_github_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "github",
            "login": "octocat",
            "email": "a@b.com",
            "auth_type": "oauth",
        },
    )
    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0
    assert "github: authenticated" in result.stdout
    assert "login=octocat" in result.stdout


def test_linear_ls_lists_workspaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "fetch_linear_workspaces",
        lambda limit=50: [
            {
                "key": "potpie-ai-cli",
                "name": "Potpie AI CLI",
                "type": "workspace",
                "active": True,
            },
        ],
    )

    result = runner.invoke(auth_commands.auth_app, ["linear", "ls"])

    assert result.exit_code == 0
    assert "Potpie AI CLI" in result.stdout
    assert "potpie linear login --add" in result.stdout


def test_linear_select_fetches_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "run_linear_use_flow",
        lambda org_key=None, team_key=None, limit=10: {
            "product": "linear",
            "workspace_key": "potpie-ai-cli",
            "workspace_name": "Potpie AI CLI",
            "team_key": "ENG",
            "team_name": "Engineering",
            "items": [
                {
                    "identifier": "ENG-1",
                    "title": "Fix login",
                    "status": "In Progress",
                    "url": "https://linear.app/issue/ENG-1",
                }
            ],
        },
    )

    result = runner.invoke(
        auth_commands.auth_app,
        ["linear", "select", "--org", "potpie-ai-cli", "--key", "ENG"],
    )

    assert result.exit_code == 0
    assert "ENG-1" in result.stdout
    assert "Fix login" in result.stdout
    assert "Potpie AI CLI" in result.stdout


def test_auth_status_verify_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": True,
            "auth_type": "oauth",
            "expires_at": 9999999999.0,
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _provider: {"access_token": "tok", "expires_at": 9999999999.0},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _provider, _creds: (True, "ok (Ada)"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0, result.stdout
    assert '"verified": true' in result.stdout


def test_auth_status_human_verify_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Acme",
            "expires_at": 12345.0,
            "token_storage": "file",
            "auth_type": "oauth",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: {"access_token": "tok"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (False, "Linear API request failed"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0
    assert "verify failed" in result.stdout
    assert "Linear API request failed" in result.stdout.replace("\n", "")
    assert "expires_at=" in result.stdout


def test_auth_status_human_verify_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "login": "Ada",
            "auth_type": "oauth",
            "expires_at": 9999999999.0,
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: {"access_token": "tok"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (True, "ok (Ada)"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0
    assert "verify=ok (Ada)" in result.stdout


# --- test_credentials_store.py ---


def test_config_dir_respects_xdg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    assert cs.config_dir() == cfg / "potpie"


def test_write_read_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="secret-token", api_base_url="http://localhost:9999")
    path = cs.credentials_path()
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["api_key"] == "secret-token"
    assert data["api_base_url"] == "http://localhost:9999"
    assert cs.get_stored_api_key() == "secret-token"
    assert cs.get_stored_api_base_url() == "http://localhost:9999"


def test_write_preserves_base_url_when_url_not_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="a", api_base_url="http://x")
    cs.write_credentials(api_key="b", api_base_url=None)
    assert cs.get_stored_api_key() == "b"
    assert cs.get_stored_api_base_url() == "http://x"


def test_clear_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="x")
    cs.clear_credentials()
    assert not cs.credentials_path().is_file()


def test_clear_active_pot_id_preserves_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="k")
    cs.set_active_pot_id("11111111-1111-1111-1111-111111111111")
    cs.clear_active_pot_id()
    assert cs.get_active_pot_id() == ""
    assert cs.get_stored_api_key() == "k"
    data = json.loads(cs.credentials_path().read_text(encoding="utf-8"))
    assert "active_pot_id" not in data


def test_clear_active_pot_id_removes_file_when_only_pot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.set_active_pot_id("22222222-2222-2222-2222-222222222222")
    cs.clear_active_pot_id()
    assert not cs.credentials_path().is_file()


def test_register_and_resolve_pot_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    uid = "33333333-3333-3333-3333-333333333333"
    cs.register_pot_alias("My-Workspace", uid)
    assert cs.get_pot_aliases() == {"my-workspace": uid}
    got, err = cs.resolve_cli_pot_ref("my-workspace")
    assert err == ""
    assert got == uid
    got2, err2 = cs.resolve_cli_pot_ref("MY-WORKSPACE")
    assert err2 == ""
    assert got2 == uid


def test_resolve_cli_pot_ref_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    got, err = cs.resolve_cli_pot_ref("nope")
    assert got is None
    assert "Unknown pot" in err
    assert "pot create" in err


def test_clear_pot_scope_state_keeps_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="secret", api_base_url="http://localhost:9")
    cs.register_pot_alias("w", "77777777-7777-7777-7777-777777777777")
    cs.set_active_pot_id("77777777-7777-7777-7777-777777777777")
    cs.clear_pot_scope_state()
    assert cs.get_stored_api_key() == "secret"
    assert cs.get_active_pot_id() == ""
    assert cs.get_pot_aliases() == {}


def test_integration_metadata_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="secret")
    cs.write_integration_metadata(
        "Example",
        {"auth_type": "oauth", "token_storage": "keychain"},
    )
    assert cs.get_stored_api_key() == "secret"
    assert cs.get_integration_metadata("example") == {
        "auth_type": "oauth",
        "token_storage": "keychain",
    }
    assert cs.list_integration_metadata() == {
        "example": {"auth_type": "oauth", "token_storage": "keychain"}
    }


def test_clear_integration_metadata_preserves_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="secret")
    cs.write_integration_metadata("example", {"auth_type": "oauth"})
    cs.clear_integration_metadata("example")
    assert cs.get_integration_metadata("example") == {}
    assert cs.get_stored_api_key() == "secret"


def test_clear_integration_metadata_removes_file_when_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_integration_metadata("example", {"auth_type": "oauth"})
    cs.clear_integration_metadata("example")
    assert not cs.credentials_path().is_file()


def test_integration_metadata_rejects_empty_provider() -> None:
    with pytest.raises(ValueError, match="integration provider must be non-empty"):
        cs.get_integration_metadata(" ")


def test_secure_secret_roundtrip() -> None:
    cs.store_secure_secret("example_access_token", "secret-token")
    assert cs.load_secure_secret("example_access_token") == "secret-token"
    cs.delete_secure_secret("example_access_token")
    assert cs.load_secure_secret("example_access_token") == ""


def test_secure_secret_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="secret name must be non-empty"):
        cs.store_secure_secret(" ", "secret")


def test_secure_secret_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_write(_secrets: dict[str, str]) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(cs, "_write_integration_secrets_file", fail_write)

    with pytest.raises(cs.CredentialStoreError, match="Failed to store Example token"):
        cs.store_secure_secret("example_token", "secret", label="Example token")


def test_secure_secret_missing_read_returns_empty() -> None:
    assert cs.load_secure_secret("example_token", label="Example token") == ""


def test_secure_secret_delete_unexpected_errors_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cs.store_secure_secret("example_token", "secret")

    def fail_write(_secrets: dict[str, str]) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(cs, "_write_integration_secrets_file", fail_write)

    with pytest.raises(cs.CredentialStoreError, match="Failed to remove Example token"):
        cs.delete_secure_secret("example_token", label="Example token")


def test_secure_secret_delete_missing_is_ignored() -> None:
    cs.delete_secure_secret("example_token")


def test_resolve_cli_pot_ref_invalid_stored_uuid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    payload = {"pot_aliases": {"broken": "not-a-uuid"}}
    path = cs.credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    got, err = cs.resolve_cli_pot_ref("broken")
    assert got is None
    assert "not a valid UUID" in err


def test_resolve_cli_pot_ref_uuid_normalizes() -> None:
    s = "550E8400-E29B-41D4-A716-446655440000"
    got, err = cs.resolve_cli_pot_ref(s)
    assert err == ""
    assert got == "550e8400-e29b-41d4-a716-446655440000"


# --- test_credentials_store_integrations.py ---


@pytest.fixture
def keychain() -> dict[tuple[str, str], str]:
    return {}


def _linear_viewer_stub(**org_overrides: object) -> dict[str, object]:
    organization = {
        "id": "org-1",
        "name": "Potpie AI CLI",
        "url_key": "potpie-ai-cli",
        **org_overrides,
    }
    return {
        "account": {"name": "Ada", "email": "ada@example.com"},
        "organization": organization,
    }


def test_linear_integration_tokens_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.setattr(
        "potpie.cli.auth.integration_profile.fetch_linear_viewer",
        lambda _token: _linear_viewer_stub(),
    )
    cs.save_integration_tokens(
        "linear",
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "scope": "read",
            "expires_at": 9999999999.0,
        },
    )
    tokens = cs.get_integration_tokens("linear")
    assert tokens["access_token"] == "access"
    assert tokens["refresh_token"] == "refresh"

    cs.clear_integration_tokens("linear")
    assert cs.get_integration_tokens("linear") == {}


def test_list_integration_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.setattr(
        "potpie.cli.auth.integration_profile.fetch_linear_viewer",
        lambda _token: _linear_viewer_stub(),
    )
    cs.save_integration_tokens(
        "linear",
        {"access_token": "tok", "expires_at": 9999999999.0},
    )
    providers = cs.list_integration_providers()
    assert "linear" in providers


def test_clear_integration_tokens_linear(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.setattr(
        "potpie.cli.auth.integration_profile.fetch_linear_viewer",
        lambda _token: _linear_viewer_stub(),
    )
    cs.save_integration_tokens(
        "linear",
        {"access_token": "a", "refresh_token": "r", "expires_at": 9999999999.0},
    )
    cs.clear_integration_tokens("linear")
    assert cs.get_integration_tokens("linear") == {}
    status = cs.get_integration_status("linear")
    assert status["authenticated"] is False


def test_readable_credentials_path_prefers_legacy_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cfg = tmp_path / "xdg"
    legacy_cfg = tmp_path / "legacy-xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    monkeypatch.setattr(cs, "legacy_config_dir", lambda: legacy_cfg)
    legacy_cfg.mkdir(parents=True)
    legacy_file = legacy_cfg / "credentials.json"
    legacy_file.write_text('{"api_key": "from-legacy"}', encoding="utf-8")

    assert cs.readable_credentials_path() == legacy_file
    assert cs.read_credentials().get("api_key") == "from-legacy"


def test_read_credentials_invalid_json_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    path = tmp_path / "credentials.json"
    path.write_text("{not json", encoding="utf-8")
    assert cs.read_credentials() == {}


def test_get_integration_status_unknown_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    with pytest.raises(ValueError, match="Unknown integration provider"):
        cs.get_integration_status("slack")


def test_get_integration_status_github_unauthenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    status = cs.get_integration_status("github")
    assert status == {"provider": "github", "authenticated": False, "auth_type": "oauth"}


def test_get_integration_status_github_authenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keychain: dict[tuple[str, str], str],
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "gh-token",
            "token_type": "bearer",
            "scopes": ["repo"],
            "account": {"login": "octocat", "id": 1, "name": None, "email": "a@b.com"},
            "created_at": "2026-05-29T00:00:00+00:00",
            "updated_at": "2026-05-29T00:00:00+00:00",
            "expires_at": None,
            "metadata": {"auth_flow": "device"},
        },
    )
    status = cs.get_integration_status("github")
    assert status["authenticated"] is True
    assert status["login"] == "octocat"
    assert status["email"] == "a@b.com"
    assert status["token_storage"] == "file"


def test_linear_status_includes_org_and_scope_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(
        "potpie.cli.auth.integration_profile.fetch_linear_viewer",
        lambda _token: _linear_viewer_stub(name="Potpie", url_key="potpie"),
    )
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_integration_tokens(
        "linear",
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_at": 9999999999.0,
            "scope": "read,write",
            "account": {"name": "Ada", "email": "ada@example.com"},
            "organization": {"name": "Potpie"},
        },
    )
    status = cs.get_integration_status("linear")
    assert status["authenticated"] is True
    assert status["site_name"] == "Potpie"
    assert status["scope"] in (["read", "write"], "read,write")


def test_store_secure_secret_generic_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(_secrets: dict[str, str]) -> None:
        raise OSError("unexpected")

    monkeypatch.setattr(cs, "_write_integration_secrets_file", fail_write)
    with pytest.raises(cs.CredentialStoreError, match="local credentials file"):
        cs.store_secure_secret("name", "secret")


def test_load_secure_secret_missing_returns_empty() -> None:
    assert cs.load_secure_secret("name") == ""


# --- test_integration_profile.py ---


def test_build_linear_integration_record_includes_account_and_scopes() -> None:
    tokens = {
        "access_token": "lin-token",
        "refresh_token": "lin-refresh",
        "scope": "read,write",
        "expires_at": 999.0,
        "token_type": "Bearer",
    }
    viewer = {
        "account": {"id": "u1", "name": "Ada", "email": "ada@example.com"},
        "organization": {"id": "o1", "name": "Acme"},
    }
    with patch(
        "potpie.cli.auth.integration_profile.fetch_linear_viewer",
        return_value=viewer,
    ):
        record = build_linear_integration_record(tokens)
    assert record["account"]["email"] == "ada@example.com"
    assert record["organization"]["name"] == "Acme"
    assert record["scopes"] == ["read", "write"]
    assert record["metadata"]["auth_flow"] == "pkce"
    assert record["created_at"]
    assert record["updated_at"]


def test_build_linear_integration_record_preserves_account_on_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = {
        "account": {"id": "u1", "name": "Ada", "email": "ada@example.com"},
        "created_at": "2020-01-01T00:00:00+00:00",
    }
    tokens = {"access_token": "new-token", "expires_at": 123.0}
    with patch(
        "potpie.cli.auth.integration_profile.fetch_linear_viewer",
        return_value={},
    ):
        record = build_linear_integration_record(tokens, existing=prior)
    assert record["account"]["email"] == "ada@example.com"
    assert record["created_at"] == "2020-01-01T00:00:00+00:00"


def test_save_integration_tokens_writes_account_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    monkeypatch.setattr(cs, "_store_file_secret", lambda *a, **k: None)
    viewer = {
        "account": {"id": "u1", "name": "Ada", "email": "ada@example.com"},
        "organization": {"id": "org-acme", "name": "Acme", "url_key": "acme"},
    }
    with patch(
        "potpie.cli.auth.integration_profile.fetch_linear_viewer",
        return_value=viewer,
    ):
        cs.save_integration_tokens(
            "linear",
            {"access_token": "tok", "scope": "read", "expires_at": 1.0},
        )
    payload = cs.read_credentials()
    linear = payload["integrations"]["linear"]
    assert linear["account"]["email"] == "ada@example.com"
    assert linear["organization"]["name"] == "Acme"
    assert linear["organizations"]["org-acme"]["name"] == "Acme"


def test_get_integration_status_linear_requires_keychain_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    monkeypatch.setattr(cs, "_load_file_secret", lambda *_a, **_k: "")
    cs._write_payload(
        {
            "integrations": {
                "linear": {
                    "provider": "linear",
                    "auth_type": "oauth",
                    "account": {"email": "ada@example.com", "name": "Ada"},
                    "organization": {"name": "Acme"},
                }
            }
        }
    )
    status = cs.get_integration_status("linear")
    assert status["authenticated"] is False
    assert status["auth_type"] == "oauth"


def test_fetch_linear_viewer_non_json_body_returns_empty() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    assert fetch_linear_viewer("token", http=client) == {}


def test_fetch_linear_viewer_non_200_returns_empty() -> None:
    response = MagicMock()
    response.status_code = 500
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    assert fetch_linear_viewer("token", http=client) == {}


def test_fetch_linear_viewer_empty_token_returns_empty() -> None:
    assert fetch_linear_viewer("   ") == {}


def test_fetch_linear_viewer_non_dict_viewer_returns_empty() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": {"viewer": "not-a-dict"}}
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    assert fetch_linear_viewer("token", http=client) == {}


def test_build_linear_integration_record_normalizes_scope_list() -> None:
    record = build_linear_integration_record(
        {
            "access_token": "a",
            "scope": ["read", "write"],
            "account": {"name": "Ada"},
        }
    )
    assert record["scopes"] == ["read", "write"]


def test_fetch_linear_viewer_parses_graphql_response() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {
            "viewer": {
                "id": "v1",
                "name": "Nihit",
                "email": "n@example.com",
                "organization": {"id": "org1", "name": "Potpie"},
            }
        }
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    profile = fetch_linear_viewer("token", http=client)
    assert profile["account"]["name"] == "Nihit"
    assert profile["organization"]["name"] == "Potpie"


# --- test_integration_verify.py ---


def test_verify_linear_transport_error_returns_false() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.side_effect = AuthHttpError("offline")
    ok, message = _verify_linear("token", http=client)
    assert ok is False
    assert message == "Linear API request failed"


def test_verify_linear_non_json_body_returns_false() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is False
    assert "non-JSON" in message


def test_verify_integration_access_linear_no_token() -> None:
    ok, message = verify_integration_access("linear", {})
    assert ok is False
    assert message == "not authenticated"


def test_verify_linear_success_without_org() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {"viewer": {"id": "v1", "name": "Ada", "email": "a@example.com"}}
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is True
    assert "Ada" in message


def test_verify_integration_access_linear_expired() -> None:
    ok, message = verify_integration_access(
        "linear",
        {"access_token": "tok", "expires_at": 1.0},
    )
    assert ok is False
    assert "expired" in message


def test_verify_linear_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {
            "viewer": {
                "id": "v1",
                "name": "Ada",
                "email": "a@example.com",
                "organization": {"name": "Acme"},
            }
        }
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is True
    assert "Ada" in message
    assert "Acme" in message


def test_verify_integration_access_linear_ignores_invalid_expires_at() -> None:
    with patch(
        "potpie.cli.auth.integration_verify._verify_linear",
        return_value=(True, "ok (Ada)"),
    ) as verify:
        ok, message = verify_integration_access(
            "linear",
            {"access_token": "tok", "expires_at": "not-a-number"},
        )
    verify.assert_called_once_with("tok", http=None)
    assert ok is True
    assert message == "ok (Ada)"


def test_verify_linear_graphql_errors() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"errors": [{"message": "bad token"}]}
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is False
    assert "rejected" in message


def test_verify_linear_non_200_status() -> None:
    response = MagicMock()
    response.status_code = 401
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response
    ok, message = _verify_linear("token", http=client)
    assert ok is False
    assert "HTTP 401" in message


def test_verify_integration_access_unknown_provider() -> None:
    ok, message = verify_integration_access("slack", {})  # type: ignore[arg-type]
    assert ok is False
    assert "unknown provider" in message


# --- test_provider_config.py ---


def test_get_client_id_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
    assert get_client_id("linear") == ""


def test_get_client_id_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_CLIENT_ID", "override-client-id")
    assert get_client_id("linear") == "override-client-id"


def test_redirect_and_callback_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI",
        "http://127.0.0.1:9001/custom/callback",
    )
    assert get_redirect_uri() == "http://127.0.0.1:9001/custom/callback"
    assert get_callback_host() == "127.0.0.1"
    assert get_callback_path() == "/custom/callback"
    assert get_callback_port() == 9001


def test_get_callback_port_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI", "http://localhost:8080/callback"
    )
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "9999")
    assert get_callback_port() == 9999


def test_linear_oauth_urls() -> None:
    assert authorization_url("linear") == "https://linear.app/oauth/authorize"
    assert token_url("linear") == LINEAR_TOKEN_URL


def test_get_scopes_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_OAUTH_SCOPE", raising=False)
    assert get_scopes("linear") == "read"
    monkeypatch.setenv("LINEAR_OAUTH_SCOPE", "read,write")
    assert get_scopes("linear") == "read,write"


def test_invalid_redirect_uri_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POTPIE_CLI_OAUTH_REDIRECT_URI", "https://example.com/cb")
    with pytest.raises(ValueError, match="localhost"):
        get_callback_port()


def test_get_client_secret_linear_returns_empty() -> None:
    assert get_client_secret("linear") == ""


def test_get_callback_port_invalid_env_falls_back_to_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI", "http://localhost:8080/callback"
    )
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "not-a-port")
    assert get_callback_port() == 8080


def test_get_callback_port_out_of_range_falls_back_to_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI", "http://localhost:8080/callback"
    )
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "70000")
    assert get_callback_port() == 8080
    monkeypatch.setenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "0")
    assert get_callback_port() == 8080


def test_redirect_uri_hostname_not_localhost_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "POTPIE_CLI_OAUTH_REDIRECT_URI",
        "http://evil.example.com:8080/callback",
    )
    with pytest.raises(ValueError, match="localhost"):
        get_callback_host()


def test_unsupported_oauth_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported OAuth provider"):
        authorization_url("github")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported OAuth provider"):
        token_url("github")  # type: ignore[arg-type]
