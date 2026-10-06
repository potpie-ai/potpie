"""CLI tests for config get/list (audit 23)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli import main as cli_main
from potpie.cli.commands import _common, bootstrap
from potpie.config.local import LocalConfigService

runner = CliRunner()


class _FakeConfig:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def list_public(self) -> dict[str, str | None]:
        from potpie.config.local import (
            public_config_value,
        )

        return {
            key: public_config_value(key, value)
            for key, value in sorted(self._values.items())
        }


@pytest.fixture(autouse=True)
def _reset_json() -> None:
    _common.set_json(False)
    yield
    _common.set_json(False)


def _mock_config(config: _FakeConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "get_config_service", lambda: config)


def test_config_list_returns_all_non_secret_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_config(
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
    _mock_config(_FakeConfig({"profile": "local", "backend": "falkordb"}), monkeypatch)

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config"]["profile"] == "local"
    assert payload["config"]["backend"] == "falkordb"


def test_config_get_with_key_returns_single_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_config(_FakeConfig({"profile": "local"}), monkeypatch)

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get", "profile"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"profile": "local"}


def test_config_get_redacts_secret_like_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_config(_FakeConfig({"api_key": "sk-live-secret"}), monkeypatch)

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get", "api_key"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["api_key"] == "<redacted>"


def test_config_list_redacts_secret_like_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_config(
        _FakeConfig({"profile": "local", "github_token": "ghp_secret"}),
        monkeypatch,
    )

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


@pytest.mark.parametrize(
    ("key", "secret"),
    [
        ("api_key", True),
        ("apiKey", True),
        ("apikey", True),
        ("service.apiKey", True),
        ("ledger.api_key", True),
        ("github_token", True),
        ("access_token", True),
        ("accessToken", True),
        ("user.password", True),
        ("clientSecret", True),
        ("credential", True),
        ("profile", False),
        ("backend", False),
        ("ledger.binding", False),
        ("oauth.proxy_url", False),
        ("max_tokens", False),
        ("maxTokens", False),
        ("tokenizer", False),
        ("tokenizerModel", False),
    ],
)
def test_is_secret_config_key_handles_camelcase_and_separators(
    key: str, secret: bool
) -> None:
    from potpie.config.local import (
        is_secret_config_key,
    )

    assert is_secret_config_key(key) is secret


def test_config_get_redacts_camelcase_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_config(_FakeConfig({"service.apiKey": "sk-live-secret"}), monkeypatch)

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get", "service.apiKey"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["service.apiKey"] == "<redacted>"


class _RecordingConfig(_FakeConfig):
    def __init__(self) -> None:
        super().__init__({})
        self.writes: list[tuple[str, str]] = []

    def set(self, key: str, value: str) -> None:
        self.writes.append((key, value))
        self._values[key] = value


def test_config_set_refuses_a_key_nothing_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo used to persist and print ``set``; nothing would ever read it."""
    config = _RecordingConfig()
    _mock_config(config, monkeypatch)

    result = runner.invoke(cli_main.app, ["--json", "config", "set", "emebdder", "x"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "validation_error"
    assert "resource_index" in payload["detail"]["known_keys"]
    assert config.writes == []


@pytest.mark.parametrize("key", ["embedder", "embedding_provider", "resource_index"])
def test_config_set_accepts_catalog_and_alias_keys(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``resource_index`` is read from ``config.json``, so it must be settable;
    the older embedder spellings are still read, so they stay writable."""
    config = _RecordingConfig()
    _mock_config(config, monkeypatch)
    value = "sqlite_fts" if key == "resource_index" else "hashing"

    result = runner.invoke(cli_main.app, ["--json", "config", "set", key, value])

    assert result.exit_code == 0, result.stdout
    assert config.writes == [(key, value)]


def test_config_set_resource_index_normalizes_and_validates_the_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _RecordingConfig()
    _mock_config(config, monkeypatch)

    ok = runner.invoke(
        cli_main.app, ["--json", "config", "set", "resource_index", "SQLite-FTS"]
    )
    off = runner.invoke(
        cli_main.app, ["--json", "config", "set", "resource_index", "off"]
    )
    bad = runner.invoke(
        cli_main.app, ["--json", "config", "set", "resource_index", "sqlite_vec"]
    )

    assert ok.exit_code == 0 and off.exit_code == 0
    assert config.writes == [
        ("resource_index", "sqlite_fts"),
        ("resource_index", "none"),
    ]
    assert bad.exit_code == 1
    assert json.loads(bad.stdout)["detail"]["profiles"] == [
        "sqlite_hybrid",
        "sqlite_fts",
        "none",
    ]


def test_the_index_registry_reads_the_resource_index_key(tmp_path, monkeypatch) -> None:
    """Writer and reader agree: what ``config set`` persists is what selects."""
    from potpie_context_engine.adapters.outbound.resources.index import (
        default_resource_index_profile,
    )

    monkeypatch.delenv("CONTEXT_ENGINE_RESOURCE_INDEX", raising=False)
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    LocalConfigService(home=tmp_path).set("resource_index", "sqlite_fts")

    assert default_resource_index_profile() == "sqlite_fts"
