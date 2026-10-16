"""CLI tests for config get/set/unset/list."""

from __future__ import annotations

import json
import os
import stat

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


# --- `config set`: redacted echo, unredacted write -------------------------------
#
# These go through a real ``LocalConfigService`` rather than ``_FakeConfig``:
# the whole class of bug here is the echo disagreeing with what was persisted,
# and a fake with no ``set`` cannot tell you which of the two a fix changed.


def _real_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> LocalConfigService:
    service = LocalConfigService(home=tmp_path)
    monkeypatch.setattr(bootstrap, "get_config_service", lambda: service)
    return service


def _catalog_with(monkeypatch: pytest.MonkeyPatch, *extra: str) -> tuple[str, ...]:
    """Grow the advertised catalog for the duration of one test.

    Nothing in today's catalog is secret-shaped, so the redaction in
    ``config set`` is unreachable through the shipped key set; it is defence
    for the day a credential key is added. Patching the constant (not the
    predicate) exercises the real ``is_known_config_key`` gate.
    """
    from potpie.config import local as config_local

    catalog = config_local.KNOWN_CONFIG_KEYS + tuple(extra)
    monkeypatch.setattr(config_local, "KNOWN_CONFIG_KEYS", catalog)
    monkeypatch.setattr(bootstrap, "KNOWN_CONFIG_KEYS", catalog)
    return catalog


def test_config_set_redacts_secret_like_value_in_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config(tmp_path, monkeypatch)
    _catalog_with(monkeypatch, "ledger.token")
    _common.set_json(True)

    result = runner.invoke(
        cli_main.app,
        ["--json", "config", "set", "ledger.token", "ghp_SUPERSECRET123"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["value"] == "<redacted>"
    assert payload["redacted"] is True
    assert payload["persisted"] is True
    assert "ghp_SUPERSECRET123" not in result.output


def test_config_set_human_output_redacts_secret_like_value(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config(tmp_path, monkeypatch)
    _catalog_with(monkeypatch, "ledger.token")

    result = runner.invoke(
        cli_main.app, ["config", "set", "ledger.token", "ghp_SUPERSECRET123"]
    )

    assert result.exit_code == 0, result.output
    assert "<redacted>" in result.output
    assert "ghp_SUPERSECRET123" not in result.output


def test_config_set_persists_the_real_value_not_the_redaction(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redact the echo, not the write."""
    _real_config(tmp_path, monkeypatch)
    _catalog_with(monkeypatch, "ledger.token")

    result = runner.invoke(
        cli_main.app, ["config", "set", "ledger.token", "ghp_SUPERSECRET123"]
    )

    assert result.exit_code == 0, result.output
    on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert on_disk["ledger.token"] == "ghp_SUPERSECRET123"
    assert LocalConfigService(home=tmp_path).get("ledger.token") == "ghp_SUPERSECRET123"


def test_config_set_redacts_a_credential_inside_a_url_value(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ledger.url`` is not a secret-shaped key, so the key-based redaction
    cannot see a ``user:password@`` typed into its value."""
    service = _real_config(tmp_path, monkeypatch)
    _common.set_json(True)

    result = runner.invoke(
        cli_main.app,
        ["--json", "config", "set", "ledger.url", "https://user:tok@ledger.example/x"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["value"] == "https://<redacted>@ledger.example/x"
    assert payload["redacted"] is True
    assert "tok@" not in result.output
    assert service.get("ledger.url") == "https://user:tok@ledger.example/x"
    assert service.list_public()["ledger.url"] == "https://<redacted>@ledger.example/x"


def test_an_ordinary_url_value_is_echoed_verbatim() -> None:
    from potpie.config.local import public_config_value

    assert (
        public_config_value("ledger.url", "https://ledger.example/x")
        == "https://ledger.example/x"
    )


def test_config_set_echoes_non_secret_value_verbatim(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over-redaction would turn the command into noise for every real key."""
    _real_config(tmp_path, monkeypatch)
    _common.set_json(True)

    result = runner.invoke(
        cli_main.app, ["--json", "config", "set", "profile", "local"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "key": "profile",
        "value": "local",
        "redacted": False,
        "persisted": True,
    }
    assert LocalConfigService(home=tmp_path).get("profile") == "local"


def test_config_get_after_set_still_redacts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writer and readers agree on one predicate."""
    _real_config(tmp_path, monkeypatch)
    _catalog_with(monkeypatch, "ledger.token")

    assert (
        runner.invoke(
            cli_main.app, ["config", "set", "ledger.token", "ghp_SUPERSECRET123"]
        ).exit_code
        == 0
    )

    _common.set_json(True)
    result = runner.invoke(cli_main.app, ["--json", "config", "get", "ledger.token"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"ledger.token": "<redacted>"}


def test_config_set_rejects_unknown_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from potpie.config.local import KNOWN_CONFIG_KEYS

    _real_config(tmp_path, monkeypatch)
    _common.set_json(True)

    result = runner.invoke(
        cli_main.app, ["--json", "config", "set", "totally.bogus.key", "42"]
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert payload["detail"]["known_keys"] == list(KNOWN_CONFIG_KEYS)
    # The refusal must not have written anything.
    assert not (tmp_path / "config.json").exists()


def test_config_set_unknown_key_uses_the_same_exit_code_in_human_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config(tmp_path, monkeypatch)

    result = runner.invoke(cli_main.app, ["config", "set", "totally.bogus.key", "42"])

    assert result.exit_code == 1, result.output
    assert "totally.bogus.key" in result.output
    assert not (tmp_path / "config.json").exists()


def test_config_set_rejects_empty_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config(tmp_path, monkeypatch)

    result = runner.invoke(cli_main.app, ["config", "set", "", "x"])

    assert result.exit_code == 1, result.output
    assert not (tmp_path / "config.json").exists()


def test_config_set_accepts_every_known_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate must not lock out a key it advertises."""
    from potpie.config.local import KNOWN_CONFIG_KEYS

    _real_config(tmp_path, monkeypatch)

    # ``resource_index`` only takes an index profile and ``graph.protocols``
    # only on/off; their validation is covered by their own tests.
    values = {"resource_index": "sqlite_fts", "graph.protocols": "on"}
    for key in KNOWN_CONFIG_KEYS:
        value = values.get(key, "x")
        result = runner.invoke(cli_main.app, ["config", "set", key, value])
        assert result.exit_code == 0, (key, result.output)
        assert LocalConfigService(home=tmp_path).get(key) == value


@pytest.mark.parametrize(
    ("key", "reader"),
    [
        ("embedding_provider", "embedder"),
        ("embedding_backend", "embedder"),
        ("sentence_transformer_model", "model"),
    ],
)
def test_config_set_accepts_keys_the_embedder_still_reads(
    key: str, reader: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unadvertised aliases the runtime obeys stay settable; asserted through
    the real reader so the alias list and its fallbacks cannot drift apart."""
    _real_config(tmp_path, monkeypatch)
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    from potpie_context_engine.adapters.outbound.intelligence import local_embedder

    result = runner.invoke(cli_main.app, ["config", "set", key, "hashing-x"])

    assert result.exit_code == 0, result.output
    if reader == "embedder":
        assert (
            local_embedder.configured_embedder_choice(include_env=False) == "hashing-x"
        )
    else:
        assert (
            local_embedder.configured_embedding_model(include_env=False) == "hashing-x"
        )


def test_local_config_service_set_get_roundtrip_is_unredacted(tmp_path) -> None:
    """Redaction is presentation, not storage: the embedder reads this service
    directly and must never receive the literal ``<redacted>``."""
    service = LocalConfigService(home=tmp_path)

    service.set("github_token", "ghp_x")

    assert service.get("github_token") == "ghp_x"
    assert service.list_public()["github_token"] == "<redacted>"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_local_config_service_saves_config_with_owner_only_permissions(
    tmp_path,
) -> None:
    # The umask is pinned so the assertion cannot be satisfied by an
    # environment that already masks group and other bits.
    service = LocalConfigService(home=tmp_path)

    previous_umask = os.umask(0o022)
    try:
        service.set("profile", "local")
    finally:
        os.umask(previous_umask)

    mode = stat.S_IMODE((tmp_path / "config.json").stat().st_mode)
    assert mode == 0o600, oct(mode)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_an_existing_world_readable_config_is_tightened_on_write(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"github_token": "ghp_x"}', encoding="utf-8")
    path.chmod(0o644)

    LocalConfigService(home=tmp_path).set("profile", "local")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# --- `config unset`: the exit the write gate needs -------------------------------
#
# `config set` refuses keys outside the catalog, which is right: nothing reads
# them. A gate with no exit would leave a stored credential the CLI can neither
# rotate nor clear, so `unset` accepts any key.


def test_config_unset_removes_a_key_the_write_gate_would_refuse(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _real_config(tmp_path, monkeypatch)
    service.set("github_token", "ghp_STRANDED_BY_THE_GATE")
    refused = runner.invoke(
        cli_main.app, ["config", "set", "github_token", "ghp_ROTATED"]
    )
    assert refused.exit_code == 1, refused.output

    result = runner.invoke(cli_main.app, ["config", "unset", "github_token"])

    assert result.exit_code == 0, result.output
    assert service.get("github_token") is None
    assert "github_token" not in json.loads((tmp_path / "config.json").read_text())


def test_config_set_refusal_names_unset_as_the_repair(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config(tmp_path, monkeypatch)
    _common.set_json(True)

    result = runner.invoke(
        cli_main.app, ["--json", "config", "set", "github_token", "ghp_x"]
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert "potpie config unset github_token" in payload["recommended_next_action"]


def test_config_unset_reports_that_nothing_was_removed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config(tmp_path, monkeypatch)
    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "unset", "never_set"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"key": "never_set", "removed": False}


def test_config_unset_does_not_echo_the_value_it_removed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _real_config(tmp_path, monkeypatch)
    service.set("github_token", "ghp_MUST_NOT_APPEAR")
    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "unset", "github_token"])

    assert result.exit_code == 0, result.output
    assert "ghp_MUST_NOT_APPEAR" not in result.output
    assert json.loads(result.output) == {"key": "github_token", "removed": True}


def test_config_unset_leaves_other_keys_alone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _real_config(tmp_path, monkeypatch)
    service.set("profile", "local")
    service.set("github_token", "ghp_x")

    runner.invoke(cli_main.app, ["config", "unset", "github_token"])

    assert service.get("profile") == "local"


def test_local_config_service_unset_reports_presence(tmp_path) -> None:
    service = LocalConfigService(home=tmp_path)
    service.set("profile", "local")

    assert service.unset("profile") is True
    assert service.unset("profile") is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_local_config_service_unset_keeps_owner_only_permissions(tmp_path) -> None:
    service = LocalConfigService(home=tmp_path)
    service.set("profile", "local")
    service.set("github_token", "ghp_x")

    previous_umask = os.umask(0o022)
    try:
        service.unset("github_token")
    finally:
        os.umask(previous_umask)

    mode = stat.S_IMODE((tmp_path / "config.json").stat().st_mode)
    assert mode == 0o600, oct(mode)


@pytest.mark.parametrize("key", ["", "   "])
def test_config_get_refuses_an_empty_key(
    key: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The omitted key lists everything; an empty one is a key that cannot
    exist and must not answer like an unset one."""
    _real_config(tmp_path, monkeypatch)
    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get", key])

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "cannot be empty" in payload["message"]


def test_config_get_with_no_key_still_lists(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _real_config(tmp_path, monkeypatch)
    service.set("profile", "local")
    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["config"] == {"profile": "local"}


@pytest.mark.parametrize("key", ["", "   "])
def test_config_unset_refuses_an_empty_key(
    key: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config(tmp_path, monkeypatch)
    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "unset", key])

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert json.loads(result.output)["code"] == "validation_error"


# --- `graph.protocols`: the opt-in protocol ontology switch -----------------------


def test_config_set_graph_protocols_normalizes_and_says_a_restart_applies_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _real_config(tmp_path, monkeypatch)

    on = runner.invoke(
        cli_main.app, ["--json", "config", "set", "graph.protocols", "ON"]
    )
    assert on.exit_code == 0, on.output
    payload = json.loads(on.output)
    assert payload["value"] == "on" and payload["persisted"] is True
    assert payload["restart_required"] is True
    assert "potpie daemon restart" in payload["next_action"]
    assert service.get("graph.protocols") == "on"
    assert service.graph_protocols_enabled() is True

    human = runner.invoke(cli_main.app, ["config", "set", "graph.protocols", "false"])
    assert human.exit_code == 0, human.output
    assert "set graph.protocols=off" in human.output
    assert "potpie daemon restart" in human.output
    assert service.graph_protocols_enabled() is False


def test_config_set_graph_protocols_refuses_a_value_that_is_not_on_or_off(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _real_config(tmp_path, monkeypatch)

    result = runner.invoke(
        cli_main.app, ["--json", "config", "set", "graph.protocols", "enabled"]
    )

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert payload["detail"]["values"] == ["on", "off"]
    assert service.get("graph.protocols") is None


def test_config_unset_graph_protocols_turns_it_off_on_the_next_start(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _real_config(tmp_path, monkeypatch)
    service.set("graph.protocols", "on")

    result = runner.invoke(
        cli_main.app, ["--json", "config", "unset", "graph.protocols"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["removed"] is True and payload["restart_required"] is True
    assert service.graph_protocols_enabled() is False


@pytest.mark.parametrize(
    ("stored", "enabled"),
    [
        (None, False),
        ("", False),
        ("off", False),
        ("maybe", False),
        ("on", True),
        (" True ", True),
        (True, True),
        (False, False),
    ],
)
def test_graph_protocols_defaults_off_and_reads_only_on_as_enabled(
    tmp_path, stored, enabled
) -> None:
    if stored is not None:
        (tmp_path / "config.json").write_text(
            json.dumps({"graph.protocols": stored}), encoding="utf-8"
        )

    assert LocalConfigService(home=tmp_path).graph_protocols_enabled() is enabled
