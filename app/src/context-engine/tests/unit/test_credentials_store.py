"""credentials_store persistence."""

import json
import stat
from pathlib import Path

import pytest

from adapters.inbound.cli import credentials_store as cs


def test_config_dir_respects_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    assert cs.config_dir() == cfg / "context-engine"


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


