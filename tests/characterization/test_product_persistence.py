from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from potpie.auth import credentials_store
from potpie.daemon.process.discovery import load_discovery

FIXTURES = Path(__file__).parent / "fixtures" / "current_product_state"


def test_current_credentials_fixture_loads_from_product_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xdg_home = tmp_path / "xdg"
    config_home = xdg_home / "potpie"
    config_home.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "credentials.json", config_home / "credentials.json")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))

    assert credentials_store.read_credentials() == {
        "api_key": "fixture-api-key",
        "api_base_url": "https://api.fixture.invalid",
        "active_pot_id": "11111111-1111-1111-1111-111111111111",
        "pot_aliases": {"fixture-project": "11111111-1111-1111-1111-111111111111"},
        "integrations": {
            "github": {
                "auth_type": "device_flow",
                "login": "fixture-user",
                "token_storage": "file",
            }
        },
    }
    assert credentials_store.get_active_pot_id() == (
        "11111111-1111-1111-1111-111111111111"
    )
    assert credentials_store.get_integration_metadata("github")["login"] == (
        "fixture-user"
    )


def test_current_daemon_discovery_fixture_preserves_typed_fields(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    shutil.copyfile(FIXTURES / "discovery.json", home / "discovery.json")

    assert load_discovery(home) == {
        "transport": "http",
        "base_url": "http://127.0.0.1:9123",
        "token": "fixture-daemon-token",
        "log_file": "/tmp/potpie-fixture-daemon.log",
        "backend": "embedded",
        "pid": 4242,
    }
