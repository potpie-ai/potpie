"""credentials_store persistence."""

import json
import stat
from pathlib import Path

import pytest

from adapters.outbound.cli_auth import credentials_store as cs


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


def test_write_payload_uses_private_temp_file_before_atomic_replace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    original_replace = cs.os.replace
    seen: dict[str, object] = {}

    def assert_secure_replace(src: str | Path, dst: str | Path) -> None:
        source = Path(src)
        destination = Path(dst)
        seen["called"] = True
        seen["source_mode"] = stat.S_IMODE(source.stat().st_mode)
        seen["destination_exists_before_replace"] = destination.exists()
        original_replace(src, dst)

    monkeypatch.setattr(cs.os, "replace", assert_secure_replace)

    cs.write_credentials(api_key="secret-token")

    assert seen == {
        "called": True,
        "source_mode": 0o600,
        "destination_exists_before_replace": False,
    }
    assert stat.S_IMODE(cs.credentials_path().stat().st_mode) == 0o600


def test_write_preserves_base_url_when_url_not_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="a", api_base_url="http://x")
    cs.write_credentials(api_key="b", api_base_url=None)
    assert cs.get_stored_api_key() == "b"
    assert cs.get_stored_api_base_url() == "http://x"


def test_write_api_base_url_does_not_touch_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="legacy-file-key", api_base_url="http://old")

    cs.write_api_base_url("https://api.example.com/")

    data = cs.read_credentials()
    assert data["api_key"] == "legacy-file-key"
    assert data["api_base_url"] == "https://api.example.com"


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


def test_bitbucket_credentials_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    cs.save_bitbucket_credentials(
        {
            "email": "user@example.com",
            "api_token": "bb-token",
            "account_name": "Bitbucket User",
        }
    )

    creds = cs.get_bitbucket_credentials()
    status = cs.get_integration_status("bitbucket")

    assert creds["api_token"] == "bb-token"
    assert status["authenticated"] is True
    assert status["email"] == "user@example.com"
    assert status["login"] == "Bitbucket User"
    assert cs._read_integration_secrets_file()["bitbucket_api_token"] == "bb-token"


def test_clear_atlassian_suite_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.save_jira_credentials({"email": "u@example.com", "api_token": "jira-token"})
    cs.save_confluence_credentials(
        {"email": "u@example.com", "api_token": "conf-token"}
    )
    cs.save_bitbucket_credentials({"email": "u@example.com", "api_token": "bb-token"})
    cs.save_atlassian_credentials(
        {"email": "u@example.com", "api_token": "legacy-token"}
    )

    cs.clear_atlassian_suite_credentials()

    assert cs.get_jira_credentials() == {}
    assert cs.get_confluence_credentials() == {}
    assert cs.get_bitbucket_credentials() == {}
    assert cs.get_atlassian_credentials() == {}


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
        {"auth_type": "oauth", "token_storage": "file"},
    )
    assert cs.get_stored_api_key() == "secret"
    assert cs.get_integration_metadata("example") == {
        "auth_type": "oauth",
        "token_storage": "file",
    }
    assert cs.list_integration_metadata() == {
        "example": {"auth_type": "oauth", "token_storage": "file"}
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


def test_secure_secret_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
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


def test_secure_secret_missing_read_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert cs.load_secure_secret("example_token", label="Example token") == ""


def test_secure_secret_delete_unexpected_errors_are_wrapped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.store_secure_secret("example_token", "secret")

    def fail_write(_secrets: dict[str, str]) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(cs, "_write_integration_secrets_file", fail_write)

    with pytest.raises(cs.CredentialStoreError, match="Failed to remove Example token"):
        cs.delete_secure_secret("example_token", label="Example token")


def test_secure_secret_delete_missing_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.delete_secure_secret("example_token")


def test_resolve_cli_pot_ref_uuid_normalizes() -> None:
    s = "550E8400-E29B-41D4-A716-446655440000"
    got, err = cs.resolve_cli_pot_ref(s)
    assert err == ""
    assert got == "550e8400-e29b-41d4-a716-446655440000"


def test_store_potpie_api_key_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(
        api_key="legacy-file-key", api_base_url="http://localhost:8000"
    )

    cs.store_potpie_api_key("sk-chain-key", created_at="2026-05-29T12:00:00+00:00")

    secrets = json.loads(cs.integration_secrets_path().read_text(encoding="utf-8"))
    assert secrets["potpie_api_key"] == "sk-chain-key"
    assert cs.get_stored_api_key() == "sk-chain-key"

    data = cs.read_credentials()
    assert data["api_key"] == "legacy-file-key"
    assert data["api_base_url"] == "http://localhost:8000"
    assert data["integrations"]["potpie"] == {
        "auth_type": "api_key",
        "token_storage": "file",
        "created_at": "2026-05-29T12:00:00+00:00",
    }
    assert "sk-chain-key" not in json.dumps(data)


def test_store_potpie_firebase_refresh_token_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="legacy-file-key")

    cs.store_potpie_firebase_refresh_token(
        "refresh-token",
        created_at="2026-05-29T12:00:00+00:00",
        firebase_api_key="firebase-key",
    )

    secrets = json.loads(cs.integration_secrets_path().read_text(encoding="utf-8"))
    assert secrets["potpie_firebase_refresh_token"] == "refresh-token"
    assert secrets["potpie_firebase_api_key"] == "firebase-key"
    assert cs.get_potpie_auth_type() == "potpie"
    assert cs.get_potpie_firebase_refresh_token() == "refresh-token"
    assert cs.get_potpie_firebase_api_key() == "firebase-key"

    data = cs.read_credentials()
    assert data["api_key"] == "legacy-file-key"
    assert data["integrations"]["potpie"]["token_storage"] == "file"
    assert "refresh-token" not in json.dumps(data)
    assert "firebase-key" not in json.dumps(data)


def test_update_potpie_firebase_refresh_token_keeps_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.store_potpie_firebase_refresh_token(
        "refresh-token",
        created_at="2026-05-29T12:00:00+00:00",
    )

    cs.update_potpie_firebase_refresh_token("new-refresh-token")

    assert cs.get_potpie_firebase_refresh_token() == "new-refresh-token"
    assert cs.read_credentials()["integrations"]["potpie"]["created_at"] == (
        "2026-05-29T12:00:00+00:00"
    )


def test_store_potpie_firebase_id_token_uses_file_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    cs.store_potpie_firebase_id_token("id-token")

    secrets = json.loads(cs.integration_secrets_path().read_text(encoding="utf-8"))
    assert secrets["potpie_firebase_id_token"] == "id-token"
    assert cs.get_potpie_firebase_id_token() == "id-token"
    assert cs.read_credentials() == {}


def test_clear_potpie_auth_preserves_api_key_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(
        api_key="legacy-file-key", api_base_url="http://localhost:8000"
    )
    cs.store_potpie_api_key("sk-chain-key", created_at="2026-05-29T12:00:00+00:00")
    cs.store_potpie_firebase_refresh_token(
        "refresh-token",
        created_at="2026-05-29T12:01:00+00:00",
    )

    cs.clear_potpie_auth()

    assert cs.get_stored_api_key() == "sk-chain-key"
    assert cs.get_potpie_firebase_refresh_token() == ""
    assert cs.get_integration_metadata("potpie") == {}
    assert cs.read_credentials() == {"api_base_url": "http://localhost:8000"}


def test_clear_potpie_auth_can_clear_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(
        api_key="legacy-file-key",
        api_base_url="http://localhost:8000",
    )
    cs.store_potpie_api_key("sk-chain-key", created_at="2026-05-29T12:00:00+00:00")

    cs.clear_potpie_auth(clear_api_key=True)

    assert cs.get_stored_api_key() == ""
    assert cs.read_credentials() == {"api_base_url": "http://localhost:8000"}


def test_write_provider_credentials_preserves_existing_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="secret-token", api_base_url="http://localhost:9999")
    cs.set_active_pot_id("11111111-1111-1111-1111-111111111111")
    cs.register_pot_alias("demo", "22222222-2222-2222-2222-222222222222")

    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "plaintext-token",
            "token_type": "bearer",
            "scopes": ["repo", "read:org", "read:user"],
            "account": {"login": "octocat", "id": 1, "name": None, "email": None},
            "created_at": "2026-05-29T00:00:00+00:00",
            "updated_at": "2026-05-29T00:00:00+00:00",
            "expires_at": None,
            "metadata": {"auth_flow": "device"},
        },
    )

    path = cs.credentials_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["api_key"] == "secret-token"
    assert data["api_base_url"] == "http://localhost:9999"
    assert data["active_pot_id"] == "11111111-1111-1111-1111-111111111111"
    assert data["pot_aliases"] == {"demo": "22222222-2222-2222-2222-222222222222"}
    assert "access_token" not in data["integrations"]["github"]
    assert data["integrations"]["github"]["token_storage"] == "file"
    secrets = json.loads(cs.integration_secrets_path().read_text(encoding="utf-8"))
    assert secrets["github_token"] == "plaintext-token"
    assert cs.get_provider_credentials("github")["access_token"] == "plaintext-token"


def test_get_provider_credentials_reads_from_integration_secrets_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "plaintext-token",
            "token_type": "bearer",
            "scopes": ["repo"],
            "account": {"login": "octocat", "id": 1, "name": None, "email": None},
            "created_at": "2026-05-29T00:00:00+00:00",
            "updated_at": "2026-05-29T00:00:00+00:00",
            "expires_at": None,
            "metadata": {"auth_flow": "device"},
        },
    )

    cs._write_integration_secrets_file({"github_token": "from-file"})

    assert cs.get_provider_credentials("github")["access_token"] == "from-file"


def test_github_status_detects_file_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "plaintext-token",
            "account": {"login": "octocat", "email": "octo@example.com"},
            "updated_at": "2026-05-29T00:00:00+00:00",
        },
    )

    status = cs.get_integration_status("github")

    assert status["authenticated"] is True
    assert status["login"] == "octocat"
    assert status["email"] == "octo@example.com"
    assert status["token_storage"] == "file"


def test_get_provider_credentials_raises_when_integration_secret_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_integration_metadata(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "token_storage": "file",
            "account": {"login": "octocat", "id": 1},
        },
    )

    with pytest.raises(cs.ProviderCredentialError) as exc:
        cs.get_provider_credentials("github")

    assert str(exc.value) == (
        "GitHub token not found in local credentials file. Run: potpie github login"
    )


def test_write_provider_credentials_raises_on_file_store_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    def _fail(_secrets: dict[str, str]) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(cs, "_write_integration_secrets_file", _fail)

    with pytest.raises(cs.ProviderCredentialError) as exc:
        cs.write_provider_credentials(
            "github",
            {
                "provider": "github",
                "provider_host": "github.com",
                "access_token": "plaintext-token",
                "token_type": "bearer",
                "scopes": ["repo"],
                "account": {"login": "octocat", "id": 1, "name": None, "email": None},
                "created_at": "2026-05-29T00:00:00+00:00",
                "updated_at": "2026-05-29T00:00:00+00:00",
                "expires_at": None,
                "metadata": {"auth_flow": "device"},
            },
        )

    assert "Failed to store GitHub token in local credentials file" in str(exc.value)


def test_write_provider_credentials_rolls_back_file_token_on_metadata_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    def _fail_metadata(_provider: str, _metadata: dict[str, object]) -> None:
        raise OSError("metadata write failed")

    monkeypatch.setattr(cs, "write_integration_metadata", _fail_metadata)

    with pytest.raises(OSError, match="metadata write failed"):
        cs.write_provider_credentials(
            "github",
            {
                "provider": "github",
                "provider_host": "github.com",
                "access_token": "plaintext-token",
                "account": {"login": "octocat", "id": 1},
            },
        )

    assert not cs.integration_secrets_path().is_file()


def test_write_provider_credentials_restores_existing_token_on_metadata_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "old-token",
            "account": {"login": "octocat", "id": 1},
        },
    )

    def _fail_metadata(_provider: str, _metadata: dict[str, object]) -> None:
        raise OSError("metadata write failed")

    monkeypatch.setattr(cs, "write_integration_metadata", _fail_metadata)

    with pytest.raises(OSError, match="metadata write failed"):
        cs.write_provider_credentials(
            "github",
            {
                "provider": "github",
                "provider_host": "github.com",
                "access_token": "new-token",
                "account": {"login": "octocat", "id": 1},
            },
        )

    secrets = json.loads(cs.integration_secrets_path().read_text(encoding="utf-8"))
    assert secrets["github_token"] == "old-token"
    assert cs.get_provider_credentials("github")["access_token"] == "old-token"


def test_write_provider_credentials_preserves_metadata_error_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    def _fail_metadata(_provider: str, _metadata: dict[str, object]) -> None:
        raise OSError("metadata write failed")

    cleanup_calls = 0

    def _fail_cleanup(_label: str, _secret_name: str) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise cs.ProviderCredentialError("cleanup failed")

    monkeypatch.setattr(cs, "write_integration_metadata", _fail_metadata)
    monkeypatch.setattr(cs, "_delete_file_secret", _fail_cleanup)

    with pytest.raises(OSError, match="metadata write failed"):
        cs.write_provider_credentials(
            "github",
            {
                "provider": "github",
                "provider_host": "github.com",
                "access_token": "plaintext-token",
                "account": {"login": "octocat", "id": 1},
            },
        )

    assert cleanup_calls == 1


def test_integration_secrets_stored_in_json_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "linux-file-token",
            "token_type": "bearer",
            "scopes": ["repo"],
            "account": {"login": "octocat", "id": 1, "name": None, "email": None},
            "created_at": "2026-05-29T00:00:00+00:00",
            "updated_at": "2026-05-29T00:00:00+00:00",
            "expires_at": None,
            "metadata": {"auth_flow": "device"},
        },
    )

    secrets_path = cs.integration_secrets_path()
    assert secrets_path.is_file()
    secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
    assert secrets["github_token"] == "linux-file-token"

    metadata = json.loads(cs.credentials_path().read_text(encoding="utf-8"))
    assert metadata["integrations"]["github"]["token_storage"] == "file"
    assert cs.get_provider_credentials("github")["access_token"] == "linux-file-token"


def test_github_status_detects_file_credentials_on_any_platform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "linux-file-token",
            "account": {"login": "octocat", "email": "octo@example.com"},
            "updated_at": "2026-05-29T00:00:00+00:00",
        },
    )

    status = cs.get_integration_status("github")

    assert status["authenticated"] is True
    assert status["login"] == "octocat"
    assert status["email"] == "octo@example.com"
    assert status["token_storage"] == "file"


def test_potpie_api_key_uses_file_secret_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    cs.store_potpie_api_key("sk-test-file-store", created_at="2026-01-01T00:00:00Z")

    secrets = json.loads(cs.integration_secrets_path().read_text(encoding="utf-8"))
    assert secrets["potpie_api_key"] == "sk-test-file-store"
    assert cs.get_stored_api_key() == "sk-test-file-store"


def test_integration_secret_ignores_keychain_metadata_without_file_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_integration_metadata(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "token_storage": "keychain",
            "account": {"login": "octocat", "id": 1},
        },
    )

    with pytest.raises(cs.ProviderCredentialError) as exc:
        cs.get_provider_credentials("github")
    assert "local credentials file" in str(exc.value)


def test_integration_secret_uses_file_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "linux-file-token",
            "token_type": "bearer",
            "scopes": ["repo"],
            "account": {"login": "octocat", "id": 1, "name": None, "email": None},
            "created_at": "2026-05-29T00:00:00+00:00",
            "updated_at": "2026-05-29T00:00:00+00:00",
            "expires_at": None,
            "metadata": {"auth_flow": "device"},
        },
    )

    assert cs.integration_secrets_path().is_file()

    metadata = json.loads(cs.credentials_path().read_text(encoding="utf-8"))
    assert metadata["integrations"]["github"]["token_storage"] == "file"
    assert cs.get_provider_credentials("github")["access_token"] == "linux-file-token"


def test_github_status_ignores_keychain_metadata_without_file_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_integration_metadata(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "token_storage": "keychain",
            "account": {"login": "octocat", "email": "octo@example.com"},
        },
    )

    status = cs.get_integration_status("github")

    assert status["authenticated"] is False
    assert status["auth_type"] == "oauth"


def test_delete_integration_secret_surfaces_provider_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_provider_credentials(
        "github",
        {
            "provider": "github",
            "provider_host": "github.com",
            "access_token": "linux-file-token",
            "token_type": "bearer",
            "scopes": ["repo"],
            "account": {"login": "octocat", "id": 1, "name": None, "email": None},
            "created_at": "2026-05-29T00:00:00+00:00",
            "updated_at": "2026-05-29T00:00:00+00:00",
            "expires_at": None,
            "metadata": {"auth_flow": "device"},
        },
    )

    def _fail_write(_secrets: dict[str, str]) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(cs, "_write_integration_secrets_file", _fail_write)

    with pytest.raises(cs.ProviderCredentialError) as exc:
        cs.clear_provider_credentials("github")

    assert "Failed to remove GitHub token from local credentials file" in str(
        exc.value
    )
