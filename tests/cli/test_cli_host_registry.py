"""The host registry as a store: what survives a failed write.

``cli_hosts.json`` holds the only copy of a managed-service token. Most of what
is pinned here is about that file still being there afterwards (after a refused
connection, a typo, a corrupt read), plus the address and credential rules that
decide whether a stored host is usable at all.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
from potpie_context_engine.core.errors import ContextEngineDisabled
from typer.testing import CliRunner

from potpie.cli.commands import _common
from potpie.cli.commands import host as host_cmds
from potpie.cli.main import app
from potpie.config import host_registry as hosts

_STORED_URL = "https://your-host.example"
_STORED_TOKEN = "s3secrettoken"


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A throwaway home with the real registry helpers left in place.

    The env overrides are cleared too: one in a developer's shell would answer
    for the file every assertion here is about.
    """
    hosts.reset_for_tests()
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    for name in (
        "POTPIE_MANAGED_URL",
        "POTPIE_MANAGED_TOKEN",
        "POTPIE_ALLOW_INSECURE_HOST",
    ):
        monkeypatch.delenv(name, raising=False)
    yield tmp_path
    hosts.reset_for_tests()


@pytest.fixture
def configured(home):
    """A registry that already points at a managed host, plus its exact bytes."""
    hosts.set_managed_endpoint(_STORED_URL, _STORED_TOKEN)
    hosts.set_persisted_origin(hosts.MANAGED)
    return (home / "cli_hosts.json").read_bytes()


def _run(*args: str, stdin: str | None = None):
    return CliRunner().invoke(app, list(args), input=stdin)


def _no_probe_allowed(monkeypatch) -> list[str]:
    """Record probe attempts and refuse them; returns the (expected empty) log."""
    attempted: list[str] = []

    def _spy(base_url: str, token: str, allow_insecure_host: str | None) -> Any:
        attempted.append(base_url)
        raise AssertionError("the endpoint should have been refused before probing")

    monkeypatch.setattr(host_cmds, "_probe_managed", _spy)
    return attempted


# --- a failed `host set` is not a write -----------------------------------


def test_an_unreachable_endpoint_leaves_the_stored_pair_byte_for_byte(
    home, configured, monkeypatch
) -> None:
    def _refused(*_args: Any) -> Any:
        raise ContextEngineDisabled("connection refused")

    monkeypatch.setattr(host_cmds, "_probe_managed", _refused)

    result = _run(
        "--json",
        "host",
        "set",
        "https://other.example:9999",
        "--token",
        "-",
        stdin="new\n",
    )

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    error = json.loads(result.output)
    assert error["code"] == "unavailable"
    assert "https://other.example:9999" in error["message"]
    assert (home / "cli_hosts.json").read_bytes() == configured
    assert hosts.managed_endpoint() == (_STORED_URL, _STORED_TOKEN)


def test_a_typo_in_the_port_is_refused_before_any_probe_or_write(
    home, configured, monkeypatch
) -> None:
    attempted = _no_probe_allowed(monkeypatch)

    result = _run("--json", "host", "set", "https://your-host.example:8090x")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    error = json.loads(result.output)
    assert error["code"] == "validation_error"
    assert "port" in error["message"]
    assert attempted == []
    assert (home / "cli_hosts.json").read_bytes() == configured


@pytest.mark.parametrize(
    "url",
    [
        "https://your-host.example:8090x",
        "127.0.0.1:8090",
        "https://",
        "ftp://your-host.example",
        "http://your-host.example",
    ],
)
def test_no_check_refuses_an_address_it_could_never_use(home, monkeypatch, url) -> None:
    """``--no-check`` waives reachability, not syntax or transport security."""
    _no_probe_allowed(monkeypatch)

    result = _run("--json", "host", "set", url, "--no-check")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert json.loads(result.output)["code"] == "validation_error"
    assert hosts.managed_endpoint() is None
    assert not (home / "cli_hosts.json").exists()


def test_a_successful_set_stores_the_pair_with_credential_permissions(
    home, monkeypatch
) -> None:
    monkeypatch.setattr(host_cmds, "_probe_managed", lambda *_a: [object(), object()])

    result = _run(
        "--json",
        "host",
        "set",
        "https://your-host.example:8443/",
        "--token",
        "-",
        stdin="tok\n",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["endpoint"] == "https://your-host.example:8443"
    assert payload["pots"] == 2
    assert hosts.managed_endpoint() == ("https://your-host.example:8443", "tok")
    assert (home / "cli_hosts.json").stat().st_mode & 0o777 == 0o600


def test_omitting_the_token_will_not_quietly_erase_a_stored_one(
    home, configured, monkeypatch
) -> None:
    """``host set`` replaces url and token together, so omission is refused
    while a token is stored: the same keystrokes as a first setup would wipe a
    key recoverable from nowhere."""
    attempted = _no_probe_allowed(monkeypatch)

    result = _run("--json", "host", "set", "http://127.0.0.1:8091", "--no-check")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    error = json.loads(result.output)
    assert error["code"] == "validation_error"
    assert "--token ''" in error["recommended_next_action"]
    assert "--token -" in error["recommended_next_action"]
    assert attempted == []
    assert (home / "cli_hosts.json").read_bytes() == configured


def test_an_empty_token_clears_the_stored_one_when_it_is_asked_for(
    home, configured
) -> None:
    result = _run(
        "--json", "host", "set", "http://127.0.0.1:8091", "--no-check", "--token", ""
    )

    assert result.exit_code == 0, result.output
    assert hosts.managed_endpoint() == ("http://127.0.0.1:8091", "")


# --- tokens ------------------------------------------------------------------


def test_a_literal_token_is_refused_and_never_echoed(home, monkeypatch) -> None:
    """A key on argv is readable from the process table and shell history."""
    attempted = _no_probe_allowed(monkeypatch)

    result = _run("--json", "host", "set", _STORED_URL, "--token", "pcg_secret_value")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    error = json.loads(result.output)
    assert error["code"] == "validation_error"
    text = error["message"] + error["recommended_next_action"]
    assert "--token -" in text
    assert "POTPIE_MANAGED_TOKEN" in text
    assert "pcg_secret_value" not in result.output
    assert attempted == []
    assert not (home / "cli_hosts.json").exists()


def test_token_dash_reads_one_line_from_stdin(home, monkeypatch) -> None:
    probed: list[str] = []

    def _probe(base_url: str, token: str, allow_insecure_host: str | None) -> Any:
        probed.append(token)
        return []

    monkeypatch.setattr(host_cmds, "_probe_managed", _probe)

    result = _run(
        "--json",
        "host",
        "set",
        _STORED_URL,
        "--token",
        "-",
        stdin="k3y \nsecond line is ignored\n",
    )

    assert result.exit_code == 0, result.output
    # Probed in exactly the form it is stored and sent.
    assert probed == ["k3y"]
    assert hosts.managed_endpoint() == (_STORED_URL, "k3y")
    assert "k3y" not in result.output


def test_token_dash_with_nothing_on_stdin_is_refused_not_a_clear(
    home, configured, monkeypatch
) -> None:
    """A broken pipe must not read as ``--token ''`` and wipe the stored key."""
    _no_probe_allowed(monkeypatch)

    result = _run(
        "--json", "host", "set", _STORED_URL, "--token", "-", "--no-check", stdin=""
    )

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert json.loads(result.output)["code"] == "validation_error"
    assert (home / "cli_hosts.json").read_bytes() == configured


def test_an_empty_token_is_refused_for_a_non_loopback_host(home, monkeypatch) -> None:
    _no_probe_allowed(monkeypatch)

    for extra in ((), ("--token", "")):
        result = _run("--json", "host", "set", _STORED_URL, "--no-check", *extra)

        assert result.exit_code == _common.EXIT_VALIDATION, result.output
        error = json.loads(result.output)
        assert error["code"] == "validation_error"
        assert "loopback" in error["message"]
        assert "--token -" in error["recommended_next_action"]
    assert not (home / "cli_hosts.json").exists()
    with pytest.raises(ValueError, match="loopback"):
        hosts.set_managed_endpoint(_STORED_URL, "")


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8090", "http://localhost:8090", "http://[::1]:8090"]
)
def test_a_loopback_service_with_auth_disabled_needs_no_token(home, url) -> None:
    result = _run("--json", "host", "set", url, "--no-check")

    assert result.exit_code == 0, result.output
    assert hosts.managed_endpoint() == (url, "")
    assert json.loads((home / "cli_hosts.json").read_text())["managed_token"] == ""


def test_a_whitespace_only_token_is_a_clear_not_a_credential(home) -> None:
    result = _run(
        "--json", "host", "set", "http://127.0.0.1:8090", "--no-check", "--token", "  "
    )

    assert result.exit_code == 0, result.output
    assert hosts.managed_endpoint() == ("http://127.0.0.1:8090", "")


# --- plain http opt-in ----------------------------------------------------------


def test_plain_http_to_a_remote_host_needs_the_exact_opt_in(home) -> None:
    with pytest.raises(ValueError, match="--allow-insecure"):
        hosts.normalize_managed_url("http://svc.example:8090")
    assert (
        hosts.normalize_managed_url(
            "http://svc.example:8090/", allow_insecure_host="svc.example:8090"
        )
        == "http://svc.example:8090"
    )
    # Exact host[:port]: neither the bare host nor another port stands in.
    for other in ("svc.example", "svc.example:8091", "other.example:8090"):
        with pytest.raises(ValueError, match="opt-in names"):
            hosts.normalize_managed_url(
                "http://svc.example:8090", allow_insecure_host=other
            )
    # Host names compare case-insensitively.
    assert hosts.normalize_managed_url(
        "http://SVC.example:8090", allow_insecure_host="svc.EXAMPLE:8090"
    )
    with pytest.raises(ValueError, match="host"):
        hosts.normalize_managed_url(
            "http://svc.example:8090", allow_insecure_host="http://svc.example:8090"
        )


def test_allow_insecure_persists_the_one_host_and_warns_on_stderr_only(home) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--json",
            "host",
            "set",
            "http://svc.example:8090",
            "--no-check",
            "--allow-insecure",
            "--token",
            "-",
        ],
        input="tok\n",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["allow_insecure_host"] == "svc.example:8090"
    assert "plain http" not in result.stdout
    assert len(result.stderr.strip().splitlines()) == 1
    assert "plain http" in result.stderr
    assert "http://svc.example:8090" in result.stderr
    stored = json.loads((home / "cli_hosts.json").read_text())
    assert stored["managed_allow_insecure_host"] == "svc.example:8090"
    host = hosts.managed_host()
    assert host is not None
    assert host.allow_insecure_host == "svc.example:8090"


def test_the_warning_is_one_line_per_process(home, capsys) -> None:
    hosts.warn_insecure_transport("http://svc.example:8090")
    hosts.warn_insecure_transport("http://svc.example:8090")
    hosts.warn_insecure_transport("https://your-host.example")
    hosts.warn_insecure_transport("http://127.0.0.1:8090")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1


def test_a_new_set_replaces_the_opt_in_rather_than_keeping_a_stale_one(home) -> None:
    hosts.set_managed_endpoint(
        "http://svc.example:8090", "tok", allow_insecure_host="svc.example:8090"
    )

    result = _run(
        "--json",
        "host",
        "set",
        _STORED_URL,
        "--no-check",
        "--token",
        "-",
        stdin="tok\n",
    )

    assert result.exit_code == 0, result.output
    stored = json.loads((home / "cli_hosts.json").read_text())
    assert "managed_allow_insecure_host" not in stored


def test_allow_insecure_is_a_no_op_for_https(home) -> None:
    result = _run(
        "--json",
        "host",
        "set",
        _STORED_URL,
        "--no-check",
        "--allow-insecure",
        "--token",
        "-",
        stdin="tok\n",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["allow_insecure_host"] is None
    assert "managed_allow_insecure_host" not in json.loads(
        (home / "cli_hosts.json").read_text()
    )


def test_a_hand_edited_plain_http_registry_fails_closed(home) -> None:
    """The rule is checked when the registry is read too, not only at ``set``."""
    (home / "cli_hosts.json").write_text(
        json.dumps({"managed_url": "http://svc.example:8090", "managed_token": "t"})
    )

    assert hosts.managed_endpoint() is None
    problem = hosts.managed_endpoint_problem()
    assert problem is not None and "http://" in problem

    # A mismatched opt-in is no opt-in.
    (home / "cli_hosts.json").write_text(
        json.dumps(
            {
                "managed_url": "http://svc.example:8090",
                "managed_token": "t",
                "managed_allow_insecure_host": "svc.example:9000",
            }
        )
    )
    assert hosts.managed_endpoint() is None


def test_the_environment_opt_in_covers_only_the_environment_host(
    home, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://svc.example:8090")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "tok")

    assert hosts.managed_endpoint() is None
    assert "POTPIE_ALLOW_INSECURE_HOST" in (hosts.managed_endpoint_problem() or "")

    monkeypatch.setenv("POTPIE_ALLOW_INSECURE_HOST", "svc.example:8090")
    assert hosts.managed_endpoint() == ("http://svc.example:8090", "tok")

    monkeypatch.setenv("POTPIE_ALLOW_INSECURE_HOST", "svc.example")
    assert hosts.managed_endpoint() is None


# --- an unreadable registry -----------------------------------------------


def test_a_corrupt_registry_fails_loud_and_names_the_file(home) -> None:
    """Read as ``{}`` it would look like a first run, one command from erasure."""
    state = home / "cli_hosts.json"
    state.write_text(
        '{"managed_url": "https://your-host.example", "managed_token": "s3'
    )

    result = _run("--json", "host", "list")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    error = json.loads(result.output)
    assert error["code"] == "unavailable"
    assert str(state) in error["message"]
    assert str(state) in error["recommended_next_action"]
    assert "--token -" in error["recommended_next_action"]


def test_a_corrupt_registry_is_not_overwritten_by_the_next_write(home) -> None:
    state = home / "cli_hosts.json"
    corrupt = '{"managed_url": "https://your-host.example", "managed_token": "s3'
    state.write_text(corrupt)

    result = _run("--json", "host", "use", "local")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    assert json.loads(result.output)["code"] == "unavailable"
    assert state.read_text() == corrupt


def test_write_state_refuses_on_its_own_terms(home) -> None:
    state = home / "cli_hosts.json"
    state.write_text("not json at all")

    with pytest.raises(hosts.HostRegistryUnreadable):
        hosts._write_state({"active_origin": "local"})

    assert state.read_text() == "not json at all"


def test_a_registry_holding_something_other_than_an_object_is_corrupt(home) -> None:
    (home / "cli_hosts.json").write_text("[]")

    with pytest.raises(hosts.HostRegistryUnreadable):
        hosts.managed_endpoint()


def test_host_clear_is_the_way_out_of_a_registry_nothing_can_read(home) -> None:
    state = home / "cli_hosts.json"
    corrupt = '{"managed_url": "https://your-host.example", "managed_token": "s3'
    state.write_text(corrupt)

    result = _run("--json", "host", "clear")

    assert result.exit_code == 0, result.output
    salvaged = json.loads(result.output)["salvaged_registry"]
    assert salvaged == str(home / "cli_hosts.json.corrupt")
    assert (home / "cli_hosts.json.corrupt").read_text() == corrupt
    assert hosts.managed_endpoint() is None
    assert _run("--json", "host", "list").exit_code == 0


def test_host_clear_refuses_when_the_unreadable_bytes_cannot_be_saved(
    home, monkeypatch
) -> None:
    """The salvage is a precondition of the overwrite, not a courtesy beside it."""
    state = home / "cli_hosts.json"
    corrupt = '{"managed_url": "https://your-host.example", "managed_token": "s3'
    state.write_text(corrupt)
    kept = home / "cli_hosts.json.corrupt"
    real_replace = hosts.os.replace

    def _no_move(src: Any, dst: Any) -> None:
        if str(dst).endswith(".corrupt"):
            raise OSError("Read-only file system")
        return real_replace(src, dst)

    monkeypatch.setattr(hosts.os, "replace", _no_move)

    result = _run("--json", "host", "clear")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    error = json.loads(result.output)
    assert error["code"] == "unavailable"
    assert str(state) in error["message"]
    assert str(kept) in error["message"]
    assert "left exactly as it was" in error["message"]
    assert "by hand" in error["recommended_next_action"]
    assert state.read_text() == corrupt
    assert not kept.exists()


# --- a write that does not land --------------------------------------------


def test_the_registry_reaches_0600_before_it_is_visible_under_its_name(
    home, monkeypatch
) -> None:
    modes: list[int] = []
    real_replace = hosts.os.replace

    def _record(src: Any, dst: Any) -> None:
        if str(dst).endswith("cli_hosts.json"):
            modes.append(pathlib.Path(src).stat().st_mode & 0o777)
        return real_replace(src, dst)

    monkeypatch.setattr(hosts.os, "replace", _record)

    hosts.set_managed_endpoint(_STORED_URL, _STORED_TOKEN)

    assert modes == [0o600]
    assert (home / "cli_hosts.json").stat().st_mode & 0o777 == 0o600


def test_a_write_that_dies_mid_flight_leaves_the_old_registry_byte_for_byte(
    home, configured, monkeypatch
) -> None:
    real_replace = hosts.os.replace

    def _boom(src: Any, dst: Any) -> None:
        if str(dst).endswith("cli_hosts.json"):
            raise OSError("No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(hosts.os, "replace", _boom)

    result = _run("--json", "host", "use", "local")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    assert (home / "cli_hosts.json").read_bytes() == configured
    # No half-written scratch file left where the next run might trip over it.
    assert [p.name for p in home.iterdir()] == ["cli_hosts.json"]


@pytest.mark.skipif(
    hasattr(hosts.os, "geteuid") and hosts.os.geteuid() == 0,
    reason="root ignores the directory mode this test depends on",
)
def test_a_registry_that_cannot_be_written_names_the_file_and_the_repair(
    home,
) -> None:
    home.chmod(0o500)
    try:
        result = _run("--json", "host", "use", "local")
    finally:
        home.chmod(0o700)

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    error = json.loads(result.output)
    assert error["code"] == "unavailable"
    assert str(home / "cli_hosts.json") in error["message"]
    assert str(home) in error["recommended_next_action"]


# --- first run -------------------------------------------------------------


def test_an_absent_registry_is_a_normal_first_run(home) -> None:
    assert not (home / "cli_hosts.json").exists()

    assert hosts.persisted_origin() == hosts.LOCAL
    assert hosts.managed_endpoint() is None
    assert hosts.current_origin() == hosts.LOCAL
    assert hosts.configured_origins() == (hosts.LOCAL,)
    assert hosts.origin_degraded() is False
    assert hosts.origin_label(hosts.MANAGED) == "Managed"

    result = _run("--json", "host", "list")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["active_origin"] == hosts.LOCAL


def test_the_first_write_creates_the_registry(home) -> None:
    hosts.set_persisted_origin(hosts.MANAGED)

    assert json.loads((home / "cli_hosts.json").read_text()) == {
        "active_origin": "managed"
    }
    hosts.clear_managed_endpoint()  # also a read-modify-write on a partial file
    assert hosts.persisted_origin() == hosts.MANAGED


def test_host_clear_keeps_the_rest_of_the_state(home) -> None:
    hosts.set_persisted_origin(hosts.MANAGED)
    hosts.set_managed_endpoint(
        "http://svc.example:8090", "tok", allow_insecure_host="svc.example:8090"
    )

    hosts.clear_managed_endpoint()

    assert hosts.managed_endpoint() is None
    assert json.loads((home / "cli_hosts.json").read_text()) == {
        "active_origin": "managed"
    }
    # The selection survives so re-adding the host restores it; meanwhile the
    # pointer degrades to local.
    assert hosts.current_origin() == hosts.LOCAL


# --- origins and refs ----------------------------------------------------------


def test_only_a_known_origin_counts_as_a_prefix() -> None:
    assert hosts.split_ref("managed:api") == ("managed", "api")
    assert hosts.split_ref("local:default") == ("local", "default")
    assert hosts.split_ref("team:api") == (None, "team:api")
    assert hosts.split_ref("api") == (None, "api")
    assert hosts.split_ref("managed:") == (None, "managed:")
    assert hosts.qualify("managed", "api") == "managed:api"


def test_degraded_is_the_gap_between_the_origin_asked_for_and_the_one_in_effect(
    home,
) -> None:
    hosts.set_persisted_origin(hosts.MANAGED)

    assert (hosts.selected_origin(), hosts.current_origin()) == (
        hosts.MANAGED,
        hosts.LOCAL,
    )
    assert hosts.origin_degraded() is True

    # An override is a target, so it is not degraded to anything.
    hosts.set_current_origin(hosts.MANAGED)
    assert hosts.current_origin() == hosts.MANAGED
    assert hosts.origin_degraded() is False
    assert hosts.origin_overridden() is True


# --- the environment override ----------------------------------------------


def test_an_unusable_environment_override_is_reported_not_counted_as_configured(
    home, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://127.0.0.1:8090x")

    assert hosts.managed_endpoint() is None
    assert hosts.configured_origins() == (hosts.LOCAL,)
    assert hosts.targeting_origins() == hosts.ORIGINS

    result = _run("--json", "host", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    managed = next(r for r in payload["hosts"] if r["origin"] == hosts.MANAGED)
    assert managed["configured"] is False
    assert "POTPIE_MANAGED_URL" in managed["problem"]
    assert "port" in managed["problem"]
    assert "POTPIE_MANAGED_URL" in _run("host", "list").output


def test_a_usable_environment_override_wins_over_the_file(
    home, configured, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_MANAGED_URL", "https://scratch.example:9000/")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "scratch-token")

    assert hosts.managed_endpoint() == ("https://scratch.example:9000", "scratch-token")
    assert hosts.managed_endpoint_problem() is None
    assert hosts.configured_origins() == hosts.targeting_origins() == hosts.ORIGINS


def test_an_address_only_override_keeps_the_token_stored_for_that_address(
    home, configured, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_MANAGED_URL", f"{_STORED_URL}/")

    assert hosts.managed_endpoint() == (_STORED_URL, _STORED_TOKEN)
    assert hosts.managed_endpoint_problem() is None


def test_an_address_only_override_for_another_host_refuses_to_guess(
    home, configured, monkeypatch
) -> None:
    """A key stored for one service is not handed to whatever else the
    environment names."""
    monkeypatch.setenv("POTPIE_MANAGED_URL", "https://scratch.example:9000")

    assert hosts.managed_endpoint() is None
    problem = hosts.managed_endpoint_problem()
    assert problem is not None
    assert "https://scratch.example:9000" in problem
    assert _STORED_URL in problem
    assert "POTPIE_MANAGED_TOKEN" in problem
    assert (home / "cli_hosts.json").read_bytes() == configured

    listing = _run("--json", "host", "list")
    assert listing.exit_code == 0, listing.output
    assert json.loads(listing.output)["problem"] == problem

    refused = _run("--json", "host", "use", "managed")
    assert refused.exit_code == _common.EXIT_VALIDATION, refused.output
    error = json.loads(refused.output)
    assert error["code"] == "validation_error"
    assert problem in error["message"]
    assert "potpie host set" not in error["recommended_next_action"]


def test_an_explicit_empty_environment_token_is_the_loopback_auth_disabled_setup(
    home, configured, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://127.0.0.1:9000")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "")

    assert hosts.managed_endpoint() == ("http://127.0.0.1:9000", "")
    assert hosts.managed_endpoint_problem() is None


def test_an_empty_environment_token_for_a_remote_host_is_a_problem(
    home, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_MANAGED_URL", "https://scratch.example:9000")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "")

    assert hosts.managed_endpoint() is None
    problem = hosts.managed_endpoint_problem() or ""
    assert "loopback" in problem
    assert "POTPIE_MANAGED_TOKEN" in problem


def test_host_set_says_when_the_environment_shadows_what_it_just_wrote(
    home, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_MANAGED_URL", "https://env.example:9000")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "env-token")

    result = _run(
        "--json", "host", "set", _STORED_URL, "--no-check", "--token", "-", stdin="t\n"
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["endpoint"] == _STORED_URL
    assert payload["shadowed_by_env"] == "https://env.example:9000"
    assert hosts.stored_managed_token() == "t"
    endpoint = hosts.managed_endpoint()
    assert endpoint is not None and endpoint[0] == "https://env.example:9000"
    human = _run(
        "host", "set", _STORED_URL, "--no-check", "--token", "-", stdin="t\n"
    ).output
    assert "POTPIE_MANAGED_URL" in human


def test_host_set_stays_quiet_when_nothing_shadows_the_write(home) -> None:
    result = _run(
        "--json", "host", "set", _STORED_URL, "--no-check", "--token", "-", stdin="t\n"
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["shadowed_by_env"] is None
    human = _run(
        "host", "set", _STORED_URL, "--no-check", "--token", "-", stdin="t\n"
    ).output
    assert "POTPIE_MANAGED_URL" not in human
