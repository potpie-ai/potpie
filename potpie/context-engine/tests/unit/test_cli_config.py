"""CLI tests for config get/list (audit 23)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli as cli_main
from adapters.inbound.cli.commands import bootstrap
from application.services.config_service import LocalConfigService

runner = CliRunner()


class _FakeConfig:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def list_public(self) -> dict[str, str | None]:
        from application.services.config_service import public_config_value

        return {
            key: public_config_value(key, value)
            for key, value in sorted(self._values.items())
        }


@pytest.fixture(autouse=True)
def _reset_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from adapters.inbound.cli.commands import _common

    _common.set_json(False)
    yield
    _common.set_json(False)


def _mock_host(config: _FakeConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_host = MagicMock()
    mock_host.config = config
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)


def test_config_list_returns_all_non_secret_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_host(
        _FakeConfig(
            {
                "profile": "local",
                "backend": "falkordb",
                "home": "/Users/me/.potpie",
                "ledger.binding": "none",
            }
        ),
        monkeypatch,
    )
    from adapters.inbound.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "list"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config"]["profile"] == "local"
    assert payload["config"]["backend"] == "falkordb"
    assert "profile" in payload["known_keys"]


def test_config_get_without_key_lists_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_host(_FakeConfig({"profile": "local", "backend": "falkordb"}), monkeypatch)
    from adapters.inbound.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config"]["profile"] == "local"
    assert payload["config"]["backend"] == "falkordb"


def test_config_get_with_key_returns_single_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_host(_FakeConfig({"profile": "local"}), monkeypatch)
    from adapters.inbound.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get", "profile"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"profile": "local"}


def test_config_get_redacts_secret_like_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_host(_FakeConfig({"api_key": "sk-live-secret"}), monkeypatch)
    from adapters.inbound.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get", "api_key"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["api_key"] == "<redacted>"


def test_config_list_redacts_secret_like_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_host(
        _FakeConfig({"profile": "local", "github_token": "ghp_secret"}),
        monkeypatch,
    )
    from adapters.inbound.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "list"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config"]["profile"] == "local"
    assert payload["config"]["github_token"] == "<redacted>"


def test_local_config_service_list_public_redacts_secrets(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "local",
                "backend": "falkordb",
                "custom_password": "hunter2",
            }
        ),
        encoding="utf-8",
    )
    service = LocalConfigService(home=tmp_path)

    public = service.list_public()

    assert public["profile"] == "local"
    assert public["backend"] == "falkordb"
    assert public["custom_password"] == "<redacted>"
