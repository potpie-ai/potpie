"""Unit tests for shared CLI plumbing (Atlassian API token PR)."""

from __future__ import annotations

from __future__ import annotations
import typer
import pytest
from typer.testing import CliRunner
from adapters.inbound.cli.auth import auth_commands
from adapters.inbound.cli.commands._common import set_store
from tests._auth_fakes import InMemoryCredentialStore
from adapters.inbound.cli import host_cli as cli_main
from adapters.inbound.cli.auth.atlassian_read import AtlassianReadError
import json
import stat
from pathlib import Path
from keyring.errors import KeyringError
from adapters.outbound.cli_auth import credentials_store as cs
from adapters.outbound.cli_auth.integration_profile import (
    build_atlassian_integration_record,
    build_product_integration_record,
)
from adapters.inbound.cli.auth.atlassian_auth import (
    AtlassianAuthErrorKind,
    AtlassianVerifyResult,
)
from adapters.outbound.cli_auth.integration_verify import (
    _verify_atlassian_product,
    _verify_message_for_kind,
    verify_integration_access,
)
from adapters.outbound.cli_auth.provider_config import (
    ATLASSIAN_API_GATEWAY,
    atlassian_confluence_gateway_url,
    atlassian_jira_gateway_url,
)
# --- test_auth_commands.py ---

runner = CliRunner()


def test_auth_help_is_wired_into_main_cli() -> None:
    result = runner.invoke(cli_main.app, ["auth", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "Deprecated" in result.stdout


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
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "jira",
            "auth_type": "api_token",
        },
    )
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))

    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0, result.stdout
    assert '"integrations"' in result.stdout
    assert '"jira"' in result.stdout


def test_auth_logout_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
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
    from adapters.inbound.cli.commands import _common

    monkeypatch.setattr(_common, "is_json", lambda: True)
    monkeypatch.setattr(_common, "is_verbose", lambda: True)
    assert auth_commands._flags() == (True, True)


def test_auth_logout_not_authenticated_still_clears_stale_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    cleared: list[str] = []
    store = InMemoryCredentialStore()
    store.clear_integration_tokens = lambda provider: cleared.append(provider)  # type: ignore[method-assign]
    set_store(store)

    result = runner.invoke(auth_commands.auth_app, ["logout", "jira"])

    assert result.exit_code == 0
    assert cleared == ["jira"]
    assert "stale" in result.stdout.lower()


def test_auth_logout_wiki_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True},
    )
    cleared: list[str] = []
    store = InMemoryCredentialStore()
    store.clear_integration_tokens = lambda provider: cleared.append(provider)  # type: ignore[method-assign]
    set_store(store)

    result = runner.invoke(auth_commands.auth_app, ["logout", "wiki"])

    assert result.exit_code == 0
    assert cleared == ["confluence"]


def test_auth_revoke_delegates_to_logout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True},
    )
    set_store(InMemoryCredentialStore())

    result = runner.invoke(auth_commands.auth_app, ["revoke", "jira"])

    assert result.exit_code == 0
    assert '"ok": true' in result.stdout


def test_auth_logout_clear_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from adapters.outbound.cli_auth.credentials_store import ProviderCredentialError

    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
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

    result = runner.invoke(auth_commands.auth_app, ["logout", "jira"])

    assert result.exit_code == 4


def test_auth_logout_jira(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _provider: {"authenticated": True},
    )
    cleared: list[str] = []
    store = InMemoryCredentialStore()
    store.clear_integration_tokens = lambda provider: cleared.append(provider)  # type: ignore[method-assign]
    set_store(store)

    result = runner.invoke(auth_commands.auth_app, ["logout", "jira"])

    assert result.exit_code == 0, result.stdout
    assert cleared == ["jira"]
    assert '"ok": true' in result.stdout


# --- test_auth_commands_cli.py ---

runner = CliRunner()


def _mock_cli(monkeypatch: pytest.MonkeyPatch, *, json_mode: bool = False) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (json_mode, False))


def test_print_wiki_row_pages_and_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(
        auth_commands,
        "_print_remote_line",
        lambda message: lines.append(message),
    )

    auth_commands._print_wiki_row(
        {
            "title": "Page [bold]",
            "space": "DOCS",
            "status": "current",
            "created": "2024-01-01",
            "created_by": "Ada",
            "updated": "2024-02-01",
            "updated_by": "Bob",
            "excerpt": "Hello",
            "url": "https://x.com/p",
        },
        pages=True,
    )
    assert any("Page" in line for line in lines)
    assert any("DOCS" in line for line in lines)


def test_run_product_use_result_human(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cli(monkeypatch, json_mode=False)
    lines: list[str] = []
    monkeypatch.setattr(
        auth_commands, "print_plain_line", lambda m, **k: lines.append(m)
    )
    monkeypatch.setattr(auth_commands, "_print_remote_line", lambda m: lines.append(m))

    auth_commands._run_product_use_result(
        {
            "product": "confluence",
            "workspace_key": "DOCS",
            "workspace_name": "Documentation",
            "items": [{"title": "Intro", "space": "DOCS", "url": "https://x.com"}],
        },
        product_label="Confluence",
    )
    assert any("Documentation" in line for line in lines)
    assert any("Intro" in line for line in lines)


def test_auth_status_verify_token_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cli(monkeypatch, json_mode=True)
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": True,
            "auth_type": "api_token",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_tokens",
        lambda _p: (_ for _ in ()).throw(RuntimeError("token load broke")),
    )
    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])
    assert result.exit_code == 0
    assert '"verified": false' in result.stdout
    assert "token load broke" in result.stdout


# --- test_auth_commands_helpers.py ---


def test_handle_already_connected_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    printed: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )
    auth_commands._handle_already_connected(
        "jira",
        {
            "auth_type": "api_token",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        },
    )
    assert printed[-1].get("already_connected") is True


def test_run_product_use_result_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    blobs: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_json_blob",
        lambda payload, **kwargs: blobs.append(payload),
    )

    auth_commands._run_product_use_result(
        {
            "product": "jira",
            "workspace_key": "ENG",
            "workspace_name": "Engineering",
            "items": [{"key": "ENG-1"}],
        },
        product_label="Jira",
    )

    assert blobs[0]["workspace_key"] == "ENG"
    assert blobs[0]["provider"] == "jira"


def test_run_product_use_result_json_maps_wiki_to_confluence_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    blobs: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_json_blob",
        lambda payload, **kwargs: blobs.append(payload),
    )

    auth_commands._run_product_use_result(
        {
            "product": "wiki",
            "workspace_key": "DOCS",
            "workspace_name": "Docs",
            "items": [],
        },
        product_label="Confluence",
    )

    assert blobs[0]["product"] == "wiki"
    assert blobs[0]["provider"] == "confluence"


# --- test_auth_commands_status.py ---

runner = CliRunner()


def test_auth_status_human_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "jira",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Team",
            "site_url": "https://team.atlassian.net",
            "auth_type": "api_token",
        },
    )
    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0
    assert "jira: authenticated" in result.stdout
    assert "confluence: not authenticated" in result.stdout


def test_jira_login_help() -> None:
    result = runner.invoke(auth_commands.auth_app, ["jira", "login", "--help"])
    assert result.exit_code == 0
    assert "api-token" in result.stdout.lower() or "token" in result.stdout.lower()


def test_auth_status_human_verify_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "jira",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Team",
            "site_url": "https://team.atlassian.net",
            "expires_at": 12345.0,
            "cloud_id": "cloud-1",
            "token_storage": "keychain",
            "auth_type": "api_token",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_tokens",
        lambda _p: {"api_token": "tok"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (False, "gateway down"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    # Normalize whitespace: the long status line soft-wraps at the console width.
    out = " ".join(result.stdout.split())
    assert result.exit_code == 0
    assert "verify failed" in out
    assert "gateway down" in out
    assert "expires_at=" in out
    assert "cloud_id=" in out


def test_auth_status_human_verify_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "jira",
            "login": "Ada",
            "email": "ada@example.com",
            "auth_type": "api_token",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_tokens",
        lambda _p: {
            "api_token": "tok",
            "email": "ada@example.com",
            "site_url": "https://x.net",
        },
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


def test_secure_secret_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    stored: dict[tuple[str, str], str] = {}

    def set_password(service: str, username: str, password: str) -> None:
        stored[(service, username)] = password

    def get_password(service: str, username: str) -> str | None:
        return stored.get((service, username))

    def delete_password(service: str, username: str) -> None:
        stored.pop((service, username), None)

    monkeypatch.setattr(cs.keyring, "set_password", set_password)
    monkeypatch.setattr(cs.keyring, "get_password", get_password)
    monkeypatch.setattr(cs.keyring, "delete_password", delete_password)

    cs.store_secure_secret("example_access_token", "secret-token")
    assert cs.load_secure_secret("example_access_token") == "secret-token"
    cs.delete_secure_secret("example_access_token")
    assert cs.load_secure_secret("example_access_token") == ""


def test_secure_secret_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="secret name must be non-empty"):
        cs.store_secure_secret(" ", "secret")


def test_secure_secret_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def set_password(service: str, username: str, password: str) -> None:
        raise KeyringError("backend unavailable")

    monkeypatch.setattr(cs.keyring, "set_password", set_password)

    with pytest.raises(cs.CredentialStoreError, match="Failed to store Example token"):
        cs.store_secure_secret("example_token", "secret", label="Example token")


def test_secure_secret_read_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def get_password(service: str, username: str) -> str | None:
        raise KeyringError("backend unavailable")

    monkeypatch.setattr(cs.keyring, "get_password", get_password)

    with pytest.raises(cs.CredentialStoreError, match="Failed to read Example token"):
        cs.load_secure_secret("example_token", label="Example token")


def test_secure_secret_delete_unexpected_errors_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def delete_password(service: str, username: str) -> None:
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(cs.keyring, "delete_password", delete_password)

    with pytest.raises(cs.CredentialStoreError, match="Failed to remove Example token"):
        cs.delete_secure_secret("example_token", label="Example token")


def test_secure_secret_delete_keyring_error_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def delete_password(service: str, username: str) -> None:
        raise KeyringError("not found")

    monkeypatch.setattr(cs.keyring, "delete_password", delete_password)

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
def keychain(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    stored: dict[tuple[str, str], str] = {}

    def set_password(service: str, username: str, password: str) -> None:
        stored[(service, username)] = password

    def get_password(service: str, username: str) -> str | None:
        return stored.get((service, username))

    def delete_password(service: str, username: str) -> None:
        stored.pop((service, username), None)

    monkeypatch.setattr(cs.keyring, "set_password", set_password)
    monkeypatch.setattr(cs.keyring, "get_password", get_password)
    monkeypatch.setattr(cs.keyring, "delete_password", delete_password)
    return stored


def test_jira_credentials_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_jira_credentials(
        {
            "email": "user@example.com",
            "api_token": "jira-token",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "cloud-1",
        }
    )
    creds = cs.get_jira_credentials()
    assert creds["api_token"] == "jira-token"
    assert creds["email"] == "user@example.com"

    cs.clear_jira_credentials()
    assert cs.get_jira_credentials() == {}
    assert cs._JIRA_TOKEN_SECRET not in {k[1] for k in keychain}


def test_list_integration_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    providers = cs.list_integration_providers()
    assert "jira" in providers


def test_get_atlassian_credentials_prefers_jira(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    creds = cs.get_atlassian_credentials()
    assert creds["api_token"] == "tok"


def test_get_integration_status_jira_unauthenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    status = cs.get_integration_status("jira")
    assert status["authenticated"] is False


def test_confluence_credentials_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_confluence_credentials(
        {
            "email": "wiki@example.com",
            "api_token": "wiki-token",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "cloud-2",
        }
    )
    creds = cs.get_confluence_credentials()
    assert creds["api_token"] == "wiki-token"
    cs.clear_confluence_credentials()
    assert cs.get_confluence_credentials() == {}


def test_get_integration_tokens_jira(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    tokens = cs.get_integration_tokens("jira")
    assert tokens["auth_type"] == "api_token"
    assert tokens["api_token"] == "tok"


def test_get_integration_status_jira_authenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
            "site_name": "X",
        }
    )
    status = cs.get_integration_status("jira")
    assert status["authenticated"] is True
    assert status["email"] == "u@example.com"
    assert status["site_url"] == "https://x.atlassian.net"


def test_save_atlassian_credentials_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_atlassian_credentials(
        {
            "email": "legacy@example.com",
            "api_token": "legacy-tok",
            "site_url": "https://legacy.atlassian.net",
            "cloud_id": "legacy-cloud",
        }
    )
    creds = cs.get_atlassian_credentials()
    assert creds["email"] == "legacy@example.com"
    cs.clear_atlassian_credentials()
    assert cs.get_atlassian_credentials() == {}


def test_clear_atlassian_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    cs.clear_atlassian_credentials()
    assert cs.get_jira_credentials() == {}
    assert cs.get_confluence_credentials() == {}


def test_get_integration_status_confluence_authenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_confluence_credentials(
        {
            "email": "wiki@example.com",
            "api_token": "wiki-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "c2",
        }
    )
    status = cs.get_integration_status("confluence")
    assert status["authenticated"] is True
    assert status["email"] == "wiki@example.com"


def test_save_jira_credentials_requires_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    with pytest.raises(cs.ProviderCredentialError, match="API token is required"):
        cs.save_jira_credentials({"email": "u@example.com", "api_token": "  "})


def test_clear_integration_tokens_confluence(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_confluence_credentials(
        {
            "email": "wiki@example.com",
            "api_token": "wiki-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "c2",
        }
    )
    cs.clear_integration_tokens("confluence")
    assert cs.get_confluence_credentials() == {}


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


def test_store_secure_secret_generic_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def set_password(service: str, username: str, password: str) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cs.keyring, "set_password", set_password)
    with pytest.raises(cs.CredentialStoreError, match="keychain"):
        cs.store_secure_secret("name", "secret")


def test_load_secure_secret_generic_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get_password(service: str, username: str) -> str | None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cs.keyring, "get_password", get_password)
    with pytest.raises(cs.CredentialStoreError, match="keychain"):
        cs.load_secure_secret("name")


def test_clear_integration_tokens_jira(
    monkeypatch: pytest.MonkeyPatch, tmp_path, keychain: dict
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_jira_credentials(
        {
            "email": "u@example.com",
            "api_token": "tok",
            "site_url": "https://x.atlassian.net",
            "cloud_id": "c1",
        }
    )
    cs.clear_integration_tokens("jira")
    assert cs.get_jira_credentials() == {}


# --- test_integration_profile.py ---


def test_build_atlassian_integration_record_nested_and_flat() -> None:
    record = build_atlassian_integration_record(
        {
            "email": "user@example.com",
            "cloud_id": "cid",
            "site_url": "https://acme.atlassian.net",
            "site_name": "acme",
        }
    )
    assert record["account"]["email"] == "user@example.com"
    assert record["site"]["site_url"] == "https://acme.atlassian.net"
    assert record["email"] == "user@example.com"
    assert record["site_url"] == "https://acme.atlassian.net"


def test_build_product_integration_record_sets_provider() -> None:
    record = build_product_integration_record(
        "jira",
        {
            "email": "user@example.com",
            "cloud_id": "cid",
            "site_url": "https://acme.atlassian.net",
            "site_name": "acme",
        },
    )
    assert record["provider"] == "jira"
    assert record["account"]["email"] == "user@example.com"


def test_get_integration_status_reads_legacy_atlassian_flat_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setattr(cs, "credentials_path", lambda: cred_path)
    monkeypatch.setattr(cs, "_load_keychain_secret", lambda *a, **k: "legacy-token")
    monkeypatch.setattr(cs, "_store_keychain_secret", lambda *a, **k: None)
    cs._write_payload(
        {
            "integrations": {
                "atlassian": {
                    "provider": "atlassian",
                    "auth_type": "api_token",
                    "email": "legacy@example.com",
                    "site_url": "https://legacy.atlassian.net",
                    "site_name": "legacy",
                    "cloud_id": "cloud-1",
                }
            }
        }
    )
    status = cs.get_integration_status("atlassian")
    assert status["authenticated"] is True
    assert status["email"] == "legacy@example.com"
    assert status["site_url"] == "https://legacy.atlassian.net"


def test_clear_jira_credentials_preserves_shared_legacy_for_confluence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    keychain: dict,
) -> None:
    """Product logout must not remove legacy storage still used by the other product."""
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_atlassian_credentials(
        {
            "email": "shared@example.com",
            "api_token": "shared-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "cloud-shared",
        }
    )
    cs.save_confluence_credentials(
        {
            "email": "wiki@example.com",
            "api_token": "wiki-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "c2",
        }
    )
    cs.save_jira_credentials(
        {
            "email": "jira@example.com",
            "api_token": "jira-tok",
            "site_url": "https://team.atlassian.net",
            "cloud_id": "c1",
        }
    )

    cs.clear_jira_credentials()

    assert cs.get_confluence_credentials().get("api_token") == "wiki-tok"
    integrations = cs.read_credentials().get("integrations", {})
    assert "atlassian" in integrations
    assert "confluence" in integrations
    assert "jira" not in integrations
    assert cs._JIRA_TOKEN_SECRET not in {k[1] for k in keychain}
    assert cs._ATLASSIAN_LEGACY_TOKEN_SECRET in {k[1] for k in keychain}


def test_clear_atlassian_credentials_removes_shared_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    keychain: dict,
) -> None:
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    cs.save_atlassian_credentials(
        {
            "email": "legacy@example.com",
            "api_token": "legacy-token",
            "site_url": "https://legacy.atlassian.net",
            "cloud_id": "cloud-1",
        }
    )
    assert cs.get_jira_credentials().get("api_token") == "legacy-token"

    cs.clear_atlassian_credentials()

    assert cs.get_jira_credentials() == {}
    assert "atlassian" not in cs.read_credentials().get("integrations", {})


def test_atlassian_account_from_entry_email_only() -> None:
    from adapters.outbound.cli_auth.integration_profile import (
        atlassian_account_from_entry,
    )

    assert atlassian_account_from_entry({"email": "u@example.com"}) == {
        "email": "u@example.com"
    }


# --- test_integration_verify.py ---


def test_verify_message_for_kind() -> None:
    assert "invalid" in _verify_message_for_kind(
        "jira", AtlassianAuthErrorKind.INVALID_CREDENTIALS
    )
    assert "scope" in _verify_message_for_kind(
        "confluence", AtlassianAuthErrorKind.INSUFFICIENT_SCOPES
    )


def test_verify_integration_access_unknown_provider() -> None:
    ok, message = verify_integration_access("slack", {})  # type: ignore[arg-type]
    assert ok is False
    assert "unknown provider" in message


def test_verify_atlassian_product_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "adapters.outbound.cli_auth.integration_verify.fetch_cloud_id_for_site",
        lambda _url: "cloud-1",
    )
    monkeypatch.setattr(
        "adapters.outbound.cli_auth.integration_verify.verify_gateway_product",
        lambda *args, **kwargs: AtlassianVerifyResult(
            ok=True,
            display_name="Ada",
            error_kind=None,
        ),
    )
    ok, message = _verify_atlassian_product(
        "jira",
        {
            "email": "ada@example.com",
            "api_token": "tok",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        },
    )
    assert ok is True
    assert "Ada" in message


def test_verify_integration_access_jira(monkeypatch) -> None:
    monkeypatch.setattr(
        "adapters.outbound.cli_auth.integration_verify._verify_atlassian_product",
        lambda _p, _c: (True, "ok (Ada @ team)"),
    )
    ok, message = verify_integration_access(
        "jira",
        {"email": "a@example.com", "api_token": "t", "site_url": "https://x.net"},
    )
    assert ok is True
    assert "Ada" in message


# --- test_provider_config.py ---


def test_atlassian_gateway_urls() -> None:
    assert atlassian_jira_gateway_url("abc") == (f"{ATLASSIAN_API_GATEWAY}/ex/jira/abc")
    assert atlassian_confluence_gateway_url("abc") == (
        f"{ATLASSIAN_API_GATEWAY}/ex/confluence/abc"
    )
