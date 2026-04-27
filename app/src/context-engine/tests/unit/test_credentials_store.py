"""credentials_store persistence."""

import json
import stat
from pathlib import Path

import pytest

from adapters.inbound.cli import credentials_store as cs


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


