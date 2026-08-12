"""CLI tests for config get/list (audit 23)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from potpie.cli import main as cli_main
from potpie.cli.commands import bootstrap
from potpie_context_engine.application.services.config_service import (
    KNOWN_CONFIG_KEYS,
    LocalConfigService,
)

runner = CliRunner()


class _FakeConfig:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def list_public(self) -> dict[str, str | None]:
        from potpie_context_engine.application.services.config_service import (
            public_config_value,
        )

        return {
            key: public_config_value(key, value)
            for key, value in sorted(self._values.items())
        }


@pytest.fixture(autouse=True)
def _reset_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from potpie.cli.commands import _common

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
    from potpie.cli.commands import _common

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
    from potpie.cli.commands import _common

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
    from potpie.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get", "profile"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"profile": "local"}


def test_config_get_redacts_secret_like_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_host(_FakeConfig({"api_key": "sk-live-secret"}), monkeypatch)
    from potpie.cli.commands import _common

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
    from potpie.cli.commands import _common

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
    from potpie_context_engine.application.services.config_service import (
        is_secret_config_key,
    )

    assert is_secret_config_key(key) is secret


def test_config_get_redacts_camelcase_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_host(_FakeConfig({"service.apiKey": "sk-live-secret"}), monkeypatch)
    from potpie.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get", "service.apiKey"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["service.apiKey"] == "<redacted>"


# --- `config set`: redacted echo, unredacted write (P1-9 / S10-28..31) --------
#
# These go through a real ``LocalConfigService`` rather than ``_FakeConfig``:
# the whole class of bug here is the echo disagreeing with what was persisted,
# and a fake with no ``set`` cannot tell you which of the two a fix changed.


def _real_config_host(tmp_path, monkeypatch: pytest.MonkeyPatch) -> LocalConfigService:
    service = LocalConfigService(home=tmp_path)
    mock_host = MagicMock()
    mock_host.config = service
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    return service


def _catalog_with(monkeypatch: pytest.MonkeyPatch, *extra: str) -> tuple[str, ...]:
    """Grow the advertised catalog for the duration of one test.

    Nothing in today's catalog is secret-shaped, so the redaction in
    ``config set`` is unreachable through the shipped key set — it is defence
    for the day a credential key is added. Patching the constant (not the
    predicate) is what lets these tests exercise the real
    ``is_known_config_key`` gate on that future catalog.
    """
    from potpie_context_engine.application.services import config_service

    catalog = config_service.KNOWN_CONFIG_KEYS + tuple(extra)
    monkeypatch.setattr(config_service, "KNOWN_CONFIG_KEYS", catalog)
    monkeypatch.setattr(bootstrap, "KNOWN_CONFIG_KEYS", catalog)
    return catalog


def test_config_set_redacts_secret_like_value_in_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config_host(tmp_path, monkeypatch)
    _catalog_with(monkeypatch, "ledger.token")
    from potpie.cli.commands import _common

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
    _real_config_host(tmp_path, monkeypatch)
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
    """The fix must redact the echo, not the write."""
    _real_config_host(tmp_path, monkeypatch)
    _catalog_with(monkeypatch, "ledger.token")

    result = runner.invoke(
        cli_main.app, ["config", "set", "ledger.token", "ghp_SUPERSECRET123"]
    )

    assert result.exit_code == 0, result.output
    on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert on_disk["ledger.token"] == "ghp_SUPERSECRET123"
    assert LocalConfigService(home=tmp_path).get("ledger.token") == "ghp_SUPERSECRET123"


def test_config_set_echoes_non_secret_value_verbatim(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over-redaction would turn the command into noise for every real key."""
    _real_config_host(tmp_path, monkeypatch)
    from potpie.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(
        cli_main.app, ["--json", "config", "set", "profile", "local"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
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
    _real_config_host(tmp_path, monkeypatch)
    _catalog_with(monkeypatch, "ledger.token")
    from potpie.cli.commands import _common

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
    _real_config_host(tmp_path, monkeypatch)
    from potpie.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(
        cli_main.app, ["--json", "config", "set", "totally.bogus.key", "42"]
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    # Asserting the reported catalog (not just the refusal) is what would catch a
    # future refactor that snapshots the key set at `register()` time and then
    # silently checks a stale one.
    assert payload["detail"]["known_keys"] == list(KNOWN_CONFIG_KEYS)
    # The refusal must not have written anything.
    assert not (tmp_path / "config.json").exists()
    assert LocalConfigService(home=tmp_path).get("totally.bogus.key") is None


def test_config_set_unknown_key_uses_the_same_exit_code_in_human_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config_host(tmp_path, monkeypatch)

    result = runner.invoke(cli_main.app, ["config", "set", "totally.bogus.key", "42"])

    assert result.exit_code == 1, result.output
    assert "totally.bogus.key" in result.output
    assert not (tmp_path / "config.json").exists()


def test_config_set_rejects_empty_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_config_host(tmp_path, monkeypatch)

    result = runner.invoke(cli_main.app, ["config", "set", "", "x"])

    assert result.exit_code == 1, result.output
    assert not (tmp_path / "config.json").exists()


@pytest.mark.parametrize("key", KNOWN_CONFIG_KEYS)
def test_config_set_accepts_every_known_key(
    key: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new gate must not lock out a key it is supposed to advertise."""
    _real_config_host(tmp_path, monkeypatch)

    result = runner.invoke(cli_main.app, ["config", "set", key, "x"])

    assert result.exit_code == 0, result.output
    assert LocalConfigService(home=tmp_path).get(key) == "x"


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
    """Un-advertised aliases the runtime obeys must stay settable.

    ``local_embedder`` falls back to these older spellings, so refusing them
    would make the CLI claim a key is unknown while the embedder is reading it.
    The assertion goes through the real reader so this test breaks if the alias
    list and the fallback loops ever drift apart.
    """
    _real_config_host(tmp_path, monkeypatch)
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
    """Redaction is presentation, not storage.

    ``local_embedder`` reads this service directly; moving redaction down into
    ``get()`` would hand it the literal string ``<redacted>``.
    """
    service = LocalConfigService(home=tmp_path)

    service.set("github_token", "ghp_x")

    assert service.get("github_token") == "ghp_x"
    assert service.list_public()["github_token"] == "<redacted>"


def test_local_config_service_saves_config_with_owner_only_permissions(
    tmp_path,
) -> None:
    # The umask is pinned deliberately. Without it this assertion is satisfied
    # by the *environment* on any box or CI runner that already masks group and
    # other bits — the pre-fix `_save`, which never chmod'd anything, produces
    # 0600 under umask 077 — so the test would go green while the fix it exists
    # to pin had been deleted. 022 is the setting the file was found leaking at.
    import os
    import stat as stat_module

    service = LocalConfigService(home=tmp_path)

    previous_umask = os.umask(0o022)
    try:
        service.set("profile", "local")
    finally:
        os.umask(previous_umask)

    mode = stat_module.S_IMODE((tmp_path / "config.json").stat().st_mode)
    assert mode == 0o600, oct(mode)


# --- `config unset`: the exit the write gate needs ---------------------------
#
# `config set` refuses keys outside the catalog, which is right — nothing reads
# them. But this file accepted any key for long enough that real homes hold a
# `github_token` in it, and a gate with no exit turns "stored where nothing
# reads it" into "stored where nothing reads it and the CLI cannot clear it".
# These pin the exit open, including for the keys `set` itself will not take.


def test_config_unset_removes_a_key_the_write_gate_would_refuse(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _real_config_host(tmp_path, monkeypatch)
    service.set("github_token", "ghp_STRANDED_BY_THE_GATE")
    # Precondition, not decoration: `unset` is only interesting for keys `set`
    # rejects, so assert the refusal is real before asserting the escape works.
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
    # The refusal is the only place a user meets this problem, so the repair has
    # to be in it. Before `unset` existed this sentence told them to hand-edit
    # config.json.
    _real_config_host(tmp_path, monkeypatch)
    from potpie.cli.commands import _common

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
    # Exit 0 either way — the user is where they asked to be — but the payload
    # has to tell the two apart. Reporting a removal that did not happen is the
    # success-for-work-that-did-not-happen shape the audit exists to remove.
    _real_config_host(tmp_path, monkeypatch)
    from potpie.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "unset", "never_set"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"key": "never_set", "removed": False}


def test_config_unset_does_not_echo_the_value_it_removed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The command whose whole purpose is clearing a credential must not print it
    # on the way out — the P1-9 leak, one command to the right.
    service = _real_config_host(tmp_path, monkeypatch)
    service.set("github_token", "ghp_MUST_NOT_APPEAR")
    from potpie.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "unset", "github_token"])

    assert result.exit_code == 0, result.output
    assert "ghp_MUST_NOT_APPEAR" not in result.output
    assert json.loads(result.output) == {"key": "github_token", "removed": True}


def test_config_unset_leaves_other_keys_alone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _real_config_host(tmp_path, monkeypatch)
    service.set("profile", "local")
    service.set("github_token", "ghp_x")

    runner.invoke(cli_main.app, ["config", "unset", "github_token"])

    assert service.get("profile") == "local"


def test_local_config_service_unset_reports_presence(tmp_path) -> None:
    service = LocalConfigService(home=tmp_path)
    service.set("profile", "local")

    assert service.unset("profile") is True
    assert service.unset("profile") is False


def test_local_config_service_unset_keeps_owner_only_permissions(tmp_path) -> None:
    # `unset` rewrites the file, so it inherits the same exposure `set` had:
    # a removal that left the remaining keys world-readable would undo the
    # chmod for exactly the users most likely to run it.
    import os
    import stat as stat_module

    service = LocalConfigService(home=tmp_path)
    service.set("profile", "local")
    service.set("github_token", "ghp_x")

    previous_umask = os.umask(0o022)
    try:
        service.unset("github_token")
    finally:
        os.umask(previous_umask)

    mode = stat_module.S_IMODE((tmp_path / "config.json").stat().st_mode)
    assert mode == 0o600, oct(mode)


@pytest.mark.parametrize("key", ["", "   "])
def test_config_get_refuses_an_empty_key(
    key: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``{"": null}`` is not a reading — it is a key that cannot exist.

    Distinct from the *omitted* argument, which lists everything: the empty
    string answered as though it were a real key that happened to be unset, so
    a caller reading a config value out of a variable that came back blank got
    a plausible null instead of an error.
    """
    _real_config_host(tmp_path, monkeypatch)
    from potpie.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get", key])

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "cannot be empty" in payload["message"]


def test_config_get_with_no_key_still_lists(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal above must not swallow the documented list shorthand."""
    service = _real_config_host(tmp_path, monkeypatch)
    service.set("profile", "local")
    from potpie.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "get"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["config"] == {"profile": "local"}


@pytest.mark.parametrize("key", ["", "   "])
def test_config_unset_refuses_an_empty_key(
    key: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "'' was not set (nothing removed)" reads as a checked, negative answer."""
    _real_config_host(tmp_path, monkeypatch)
    from potpie.cli.commands import _common

    _common.set_json(True)

    result = runner.invoke(cli_main.app, ["--json", "config", "unset", key])

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert json.loads(result.output)["code"] == "validation_error"
