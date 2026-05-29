"""credentials_store persistence."""

import json
import stat
from pathlib import Path

import pytest
from keyring.errors import KeyringError

from adapters.inbound.cli import credentials_store as cs


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    store: dict[tuple[str, str], str] = {}

    def _set_password(service: str, username: str, password: str) -> None:
        store[(service, username)] = password

    def _get_password(service: str, username: str) -> str | None:
        return store.get((service, username))

    monkeypatch.setattr(cs.keyring, "set_password", _set_password)
    monkeypatch.setattr(cs.keyring, "get_password", _get_password)
    return store


def test_config_dir_respects_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_write_provider_credentials_preserves_existing_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_keyring: dict[tuple[str, str], str]
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
    assert data["integrations"]["github"]["token_storage"] == "keychain"
    assert fake_keyring[("potpie", "github_token")] == "plaintext-token"
    assert cs.get_provider_credentials("github")["access_token"] == "plaintext-token"


def test_get_provider_credentials_reads_from_keychain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_keyring: dict[tuple[str, str], str]
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

    fake_keyring[("potpie", "github_token")] = "from-keychain"

    assert cs.get_provider_credentials("github")["access_token"] == "from-keychain"


def test_get_provider_credentials_raises_when_keychain_token_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    monkeypatch.setattr(cs.keyring, "get_password", lambda _service, _username: None)

    with pytest.raises(cs.ProviderCredentialError) as exc:
        cs.get_provider_credentials("github")

    assert str(exc.value) == "GitHub token not found in system keychain. Run: potpie git login"


def test_write_provider_credentials_raises_on_keychain_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    def _fail(_service: str, _username: str, _password: str) -> None:
        raise KeyringError("backend unavailable")

    monkeypatch.setattr(cs.keyring, "set_password", _fail)

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

    assert "Failed to store GitHub token in system keychain" in str(exc.value)


def test_clear_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="x")
    cs.clear_credentials()
    assert not cs.credentials_path().is_file()


def test_clear_active_pot_id_preserves_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_register_and_resolve_pot_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_resolve_cli_pot_ref_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    got, err = cs.resolve_cli_pot_ref("nope")
    assert got is None
    assert "Unknown pot" in err
    assert "pot create" in err


def test_clear_pot_scope_state_keeps_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cs.write_credentials(api_key="secret", api_base_url="http://localhost:9")
    cs.register_pot_alias("w", "77777777-7777-7777-7777-777777777777")
    cs.set_active_pot_id("77777777-7777-7777-7777-777777777777")
    cs.clear_pot_scope_state()
    assert cs.get_stored_api_key() == "secret"
    assert cs.get_active_pot_id() == ""
    assert cs.get_pot_aliases() == {}


def test_resolve_cli_pot_ref_uuid_normalizes() -> None:
    s = "550E8400-E29B-41D4-A716-446655440000"
    got, err = cs.resolve_cli_pot_ref(s)
    assert err == ""
    assert got == "550e8400-e29b-41d4-a716-446655440000"
