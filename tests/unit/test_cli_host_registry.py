"""The host registry as a *store*: what survives a failed write.

``cli_hosts.json`` holds the only copy of a managed-service token. Everything
pinned here is about that file still being there afterwards — after a refused
connection, after a typo, after a corrupt read — because the CLI degrades to the
local graph without saying so, and the token cannot be recovered from anywhere
else once it is gone.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli import hosts
from potpie.cli.commands import _common
from potpie.cli.main import app
from potpie.daemon import client
from potpie_context_core.errors import ContextEngineDisabled

_STORED_URL = "http://127.0.0.1:8090"
_STORED_TOKEN = "s3secrettoken"


class _Pot:
    def __init__(self, pot_id: str, name: str) -> None:
        self.pot_id = pot_id
        self.name = name


class _ProbeHost:
    """Just enough host for ``host set``'s probe: it lists pots."""

    def __init__(self, pots: list[_Pot]) -> None:
        self.pots = _Pots(pots)


class _Pots:
    def __init__(self, pots: list[_Pot]) -> None:
        self._pots = pots

    def list_pots(self) -> list[_Pot]:
        return self._pots


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A throwaway CLI home with the real state helpers left in place.

    The env override is cleared too: ``POTPIE_MANAGED_URL`` in a developer's
    shell would otherwise make ``managed_endpoint`` answer from the environment
    and every assertion about the *file* meaningless.
    """
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.delenv("POTPIE_MANAGED_URL", raising=False)
    monkeypatch.delenv("POTPIE_MANAGED_TOKEN", raising=False)
    yield tmp_path
    _common.set_host(None)
    hosts.reset_for_tests()


@pytest.fixture
def configured(home):
    """A registry that already points at a managed host, plus its exact bytes."""
    hosts.set_managed_endpoint(_STORED_URL, _STORED_TOKEN)
    hosts.set_persisted_origin(hosts.MANAGED)
    return (home / "cli_hosts.json").read_bytes()


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


def _no_probe_allowed(monkeypatch) -> list[str]:
    """Record probe attempts and refuse them; returns the (expected empty) log."""
    attempted: list[str] = []

    def _spy(base_url: str, token: str) -> Any:
        attempted.append(base_url)
        raise AssertionError("the endpoint should have been refused before probing")

    monkeypatch.setattr(hosts, "build_managed_host", _spy)
    return attempted


# --- a failed `host set` is not a write -----------------------------------


def test_an_unreachable_endpoint_leaves_the_stored_pair_byte_for_byte(
    home, configured, monkeypatch
) -> None:
    """The blast radius of the old order: one refused connection replaced a
    working url and a token with nothing, and the persisted pointer still said
    `managed`, so every later command ran against the local graph at exit 0."""

    def _refused(base_url: str, token: str) -> Any:
        raise ContextEngineDisabled("connection refused")

    monkeypatch.setattr(hosts, "build_managed_host", _refused)

    result = _run("--json", "host", "set", "http://127.0.0.1:9999", "--token", "new")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    error = json.loads(result.output)
    assert error["code"] == "unavailable"
    assert "http://127.0.0.1:9999" in error["message"]
    assert (home / "cli_hosts.json").read_bytes() == configured
    assert hosts.managed_endpoint() == (_STORED_URL, _STORED_TOKEN)


def test_the_unavailable_code_keeps_its_documented_exit_code(
    home, configured, monkeypatch
) -> None:
    """`unavailable` is exit 2 everywhere else in the CLI; this path reported it
    with exit 1, so a caller branching on the code saw two different outcomes."""
    monkeypatch.setattr(
        hosts,
        "build_managed_host",
        lambda *_a: (_ for _ in ()).throw(ContextEngineDisabled("down")),
    )

    result = _run("--json", "host", "set", "http://127.0.0.1:9999", "--token", "new")

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    assert json.loads(result.output)["code"] == "unavailable"


def test_a_typo_in_the_port_is_refused_before_any_probe_or_write(
    home, configured, monkeypatch
) -> None:
    """One stray character. It parses as a string and only fails at connect
    time, which is why it used to arrive *after* the endpoint was overwritten."""
    attempted = _no_probe_allowed(monkeypatch)

    result = _run("--json", "host", "set", "http://127.0.0.1:8090x")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    error = json.loads(result.output)
    assert error["code"] == "validation_error"
    assert "port" in error["message"]
    assert attempted == []
    assert (home / "cli_hosts.json").read_bytes() == configured


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8090x", "127.0.0.1:8090", "http://", "ftp://svc.example"],
)
def test_no_check_refuses_an_address_it_could_never_use(home, monkeypatch, url) -> None:
    """`--no-check` waives *reachability*, not syntax: it used to store these and
    report success, so the failure landed on some unrelated command later."""
    _no_probe_allowed(monkeypatch)

    result = _run("--json", "host", "set", url, "--no-check")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert json.loads(result.output)["code"] == "validation_error"
    assert hosts.managed_endpoint() is None
    assert not (home / "cli_hosts.json").exists()


def test_a_successful_set_stores_the_pair_with_credential_permissions(
    home, monkeypatch
) -> None:
    monkeypatch.setattr(
        hosts,
        "build_managed_host",
        lambda *_a: _ProbeHost([_Pot("p1", "default"), _Pot("p2", "api")]),
    )

    result = _run("--json", "host", "set", "http://svc.example:8090/", "--token", "tok")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["endpoint"] == "http://svc.example:8090"
    assert payload["pots"] == 2
    assert hosts.managed_endpoint() == ("http://svc.example:8090", "tok")
    assert (home / "cli_hosts.json").stat().st_mode & 0o777 == 0o600


def test_omitting_the_token_will_not_quietly_erase_a_stored_one(
    home, configured, monkeypatch
) -> None:
    """`host set` replaces url and token together, so re-pointing at a new
    address with the same keystrokes as a first setup wiped a key recoverable
    from nowhere and exited 0. Two meanings, one omission — so it asks."""
    attempted = _no_probe_allowed(monkeypatch)

    result = _run("--json", "host", "set", "http://127.0.0.1:8091", "--no-check")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    error = json.loads(result.output)
    assert error["code"] == "validation_error"
    assert "--token ''" in error["recommended_next_action"]
    assert attempted == []
    assert (home / "cli_hosts.json").read_bytes() == configured


def test_an_empty_token_clears_the_stored_one_when_it_is_asked_for(
    home, configured
) -> None:
    """The other half of the refusal above: saying it out loud still works, or
    a service that has had auth turned off becomes unreachable by this CLI."""
    result = _run(
        "--json", "host", "set", "http://127.0.0.1:8091", "--no-check", "--token", ""
    )

    assert result.exit_code == 0, result.output
    assert hosts.managed_endpoint() == ("http://127.0.0.1:8091", "")


# --- a service with auth disabled ------------------------------------------


def test_a_service_with_auth_disabled_is_set_up_by_omitting_the_token(
    home, monkeypatch
) -> None:
    """The workflow `--token`'s own help documents.

    Stubbed at the *transport*, not at ``build_managed_host``: the real host is
    built and the real ``DaemonRpcClient`` runs, because the break was inside
    it. The placeholder used to be substituted in ``managed_endpoint()``, above
    the point where ``build_host`` delegates to ``build_managed_host``, so the
    probe was handed "" and the client's empty-token guard refused it before a
    socket was opened — reported as the *managed host* not answering, repair
    "run 'potpie daemon restart'", which names the local daemon and had nothing
    to do with it.
    """
    sent: list[dict[str, str]] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {"ok": True, "result": []}

    def _post(url: str, **kwargs: Any) -> _Resp:
        sent.append(dict(kwargs.get("headers") or {}))
        return _Resp()

    monkeypatch.setattr(client.httpx, "post", _post)

    result = _run("--json", "host", "set", "http://svc.example:8090")

    assert result.exit_code == 0, result.output
    # It reached the wire at all, carrying the placeholder rather than nothing.
    assert sent == [{"Authorization": f"Bearer {hosts.NO_AUTH_TOKEN}"}]
    # Stored as the nothing it is; the placeholder is a transport detail.
    assert json.loads((home / "cli_hosts.json").read_text())["managed_token"] == ""


def test_both_construction_paths_place_the_no_auth_placeholder_identically(
    home,
) -> None:
    """The divergence itself, pinned at the seam rather than through the CLI.

    `build_host(MANAGED)` reads the stored pair and `build_managed_host` takes
    an explicit one; while the substitution lived above that fork they produced
    different credentials for the same host, and only the path `host set` uses
    was broken."""
    hosts.set_managed_endpoint("http://svc.example:8090", "")

    from_store = hosts.build_host(hosts.MANAGED).rpc.daemon.discovery()
    explicit = hosts.build_managed_host(
        "http://svc.example:8090", ""
    ).rpc.daemon.discovery()

    assert from_store == explicit
    assert explicit["token"] == hosts.NO_AUTH_TOKEN


def test_a_managed_host_can_be_built_from_a_pair_nobody_stored(
    home, configured
) -> None:
    """Not from stored state — that is the only reason probe-then-write is
    possible at all, so it is pinned rather than left to the call order."""
    host = hosts.build_managed_host("http://new.example:1234", "fresh-token")

    assert host.rpc.daemon.discovery() == {
        "base_url": "http://new.example:1234",
        "token": "fresh-token",
    }
    # And building it changed nothing: the stored pair is still the old one.
    assert hosts.managed_endpoint() == (_STORED_URL, _STORED_TOKEN)


def test_build_host_reads_the_stored_pair_through_the_same_constructor(
    home, configured
) -> None:
    host = hosts.build_host(hosts.MANAGED)

    assert host.profile == hosts.MANAGED
    assert host.rpc.daemon.discovery() == {
        "base_url": _STORED_URL,
        "token": _STORED_TOKEN,
    }


def test_a_probe_host_is_not_cached_under_the_address_it_failed_on(
    home, configured
) -> None:
    """`_built` is keyed by base_url, so a probe with a rejected token for an
    address already configured would otherwise be handed to the next command."""
    hosts.build_managed_host(_STORED_URL, "rejected-token")

    stored = hosts.build_host(hosts.MANAGED)

    assert stored.rpc.daemon.discovery()["token"] == _STORED_TOKEN


def test_a_rotated_token_is_never_served_out_of_the_process_cache(
    home, configured
) -> None:
    """The credential is part of the host's identity, not a detail of it.

    ``_built`` is cleared only in the process that *writes* the registry, and
    the daemon serving ``/ui?host=managed`` builds its managed host through
    ``build_host`` and outlives every ``potpie host set`` run beside it. Keyed by
    the address alone, it went on sending a rotated-away key until it was
    restarted — and reported the 401 that came back as the managed service being
    down.
    """
    before = hosts.build_host(hosts.MANAGED)
    assert before.rpc.daemon.discovery()["token"] == _STORED_TOKEN

    # Rotated by someone else: the file changes with nothing telling this
    # process's cache, which is the entire situation being pinned.
    (home / "cli_hosts.json").write_text(
        json.dumps({"managed_url": _STORED_URL, "managed_token": "rotated"})
    )

    after = hosts.build_host(hosts.MANAGED)

    assert after.rpc.daemon.discovery()["token"] == "rotated"
    # The pre-rotation entry is dropped rather than accumulating one per
    # rotation for the life of a long-running process.
    assert len(hosts._built) == 1


# --- the token is normalized once, where the url is -------------------------


def test_a_token_is_probed_in_exactly_the_form_it_will_be_sent(
    home, monkeypatch
) -> None:
    """``--check`` is only worth running on the credential that gets stored.

    Only the read path stripped, so ``--token 'k3y '`` probed ``Bearer k3y `` and
    every command afterwards sent ``Bearer k3y``: a key pasted with a trailing
    space was refused at the door in the one form that would have worked.
    """
    probed: list[str] = []

    def _probe(base_url: str, token: str) -> Any:
        probed.append(token)
        return _ProbeHost([_Pot("p1", "default")])

    monkeypatch.setattr(hosts, "build_managed_host", _probe)

    result = _run("--json", "host", "set", _STORED_URL, "--token", "k3y ")

    assert result.exit_code == 0, result.output
    endpoint = hosts.managed_endpoint()
    assert endpoint is not None
    assert probed == [endpoint[1]] == ["k3y"]


def test_a_whitespace_only_token_is_probed_as_the_credential_it_becomes(
    home, monkeypatch
) -> None:
    """The same seam, at the transport, where the divergence was widest.

    ``--token '   '`` probed ``Bearer   `` and was then stored as nothing, so
    every later command sent the auth-disabled placeholder instead: the pair
    that was checked and the pair that is used had no credential in common.
    """
    sent: list[str | None] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {"ok": True, "result": []}

    def _post(url: str, **kwargs: Any) -> _Resp:
        sent.append((kwargs.get("headers") or {}).get("Authorization"))
        return _Resp()

    monkeypatch.setattr(client.httpx, "post", _post)

    result = _run("--json", "host", "set", _STORED_URL, "--token", "   ")

    assert result.exit_code == 0, result.output
    assert sent == [f"Bearer {hosts.NO_AUTH_TOKEN}"]
    assert json.loads((home / "cli_hosts.json").read_text())["managed_token"] == ""


# --- an unreadable registry -----------------------------------------------


def test_a_corrupt_registry_fails_loud_and_names_the_file(home) -> None:
    """Read as `{}` it looked like a first run: "no managed host configured",
    exit 0, and the token still sitting in the file one command from erasure."""
    state = home / "cli_hosts.json"
    state.write_text('{"managed_url": "http://127.0.0.1:8090", "managed_token": "s3')

    result = _run("--json", "host", "list")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    error = json.loads(result.output)
    assert error["code"] == "unavailable"
    assert str(state) in error["message"]
    assert str(state) in error["recommended_next_action"]


def test_a_corrupt_registry_is_not_overwritten_by_the_next_write(home) -> None:
    """The recovery window used to be one command wide."""
    state = home / "cli_hosts.json"
    corrupt = '{"managed_url": "http://127.0.0.1:8090", "managed_token": "s3'
    state.write_text(corrupt)

    result = _run("--json", "host", "use", "local")

    # The code, not just "nonzero": an unhandled traceback is also nonzero, and
    # this refusal is only worth anything if it is the deliberate one.
    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    assert json.loads(result.output)["code"] == "unavailable"
    assert state.read_text() == corrupt


def test_write_state_refuses_on_its_own_terms(home) -> None:
    """Every caller reads first today; the guard is on the file, not on them."""
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
    """`host clear` is the one command whose job is "forget the managed host",
    and it refused to run in the one state where you most want it: with every
    command exiting 2, a hand edit was the only remedy the CLI left."""
    state = home / "cli_hosts.json"
    corrupt = '{"managed_url": "http://127.0.0.1:8090", "managed_token": "s3'
    state.write_text(corrupt)

    result = _run("--json", "host", "clear")

    assert result.exit_code == 0, result.output
    salvaged = json.loads(result.output)["salvaged_registry"]
    # It says what it did, and the unreadable bytes are kept: the user asked to
    # forget the host, not to lose a token they could never read out of it.
    assert salvaged == str(home / "cli_hosts.json.corrupt")
    assert (home / "cli_hosts.json.corrupt").read_text() == corrupt
    # And the CLI is usable again rather than one command from the same wall.
    assert hosts.managed_endpoint() is None
    assert _run("--json", "host", "list").exit_code == 0


def test_host_clear_refuses_when_the_unreadable_bytes_cannot_be_saved(
    home, monkeypatch
) -> None:
    """The salvage is a *precondition* of the overwrite, not a courtesy beside it.

    ``_quarantine_state`` swallowed a failed move into ``None`` and ``host
    clear`` called ``_write_state({}, force=True)`` regardless: the corrupt
    bytes — possibly the only copy of a token — were destroyed, and the command
    reported exit 0 with ``salvaged_registry: null``, byte-identical to a clear
    that had nothing to salvage. Either the bytes are kept or the command says
    it could not keep them.
    """
    state = home / "cli_hosts.json"
    corrupt = '{"managed_url": "http://127.0.0.1:8090", "managed_token": "s3'
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
    # The bytes the whole refusal exists for.
    assert state.read_text() == corrupt
    assert not kept.exists()


# --- a write that does not land --------------------------------------------


def test_the_registry_reaches_0600_before_it_is_visible_under_its_name(
    home, monkeypatch
) -> None:
    """The token goes in the file before the file has a name anyone can open.

    Writing the contents and chmod-ing afterwards left a freshly created
    registry world-readable for the width of that gap under a lax umask.
    """
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
    """A truncated registry reads back as corrupt, which takes every later
    command down with it — so the new contents land through a temp file and a
    rename, and the registry is either the old bytes or the new ones."""
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
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores the directory mode this test depends on",
)
def test_a_registry_that_cannot_be_written_names_the_file_and_the_repair(
    home, monkeypatch
) -> None:
    """It used to surface as `unexpected_cli_error` / "Unexpected internal
    error." at exit 1, naming neither the file nor what to do; the *next*
    command was the one that finally mentioned cli_hosts.json, and from then on
    every command exited 2."""
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
    """Regression guard for the loudness above: no file is not a fault, and
    every reader has to keep working before anything has ever been written."""
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


# --- the silent degrade ----------------------------------------------------


def test_host_list_says_so_when_the_selected_host_is_not_configured(home) -> None:
    """`managed` selected, nothing configured: commands run against the local
    graph under the same pot names, at exit 0, with nothing to see."""
    hosts.set_persisted_origin(hosts.MANAGED)

    assert hosts.origin_degraded() is True

    payload = json.loads(_run("--json", "host", "list").output)
    assert payload["degraded"] is True
    assert payload["persisted_origin"] == hosts.MANAGED
    assert payload["active_origin"] == hosts.LOCAL

    # A fragment of the note itself, not the two host names — those are in the
    # ordinary rows either way, so asserting them passed with the whole
    # `if degraded:` branch deleted. Short, because the human block hard-wraps.
    assert "but none is configured" in _run("host", "list").output


def test_a_configured_managed_host_is_not_degraded(home, configured) -> None:
    assert hosts.origin_degraded() is False
    assert json.loads(_run("--json", "host", "list").output)["degraded"] is False
    assert "but none is configured" not in _run("host", "list").output


def test_degraded_is_the_gap_between_the_origin_asked_for_and_the_one_in_effect(
    home,
) -> None:
    """`degraded` has to describe the routing, not re-derive a condition beside it.

    Stated as "managed and unconfigured" against ``persisted_origin()`` it was
    blind to every origin that did not come from the pointer; stated against
    ``selected_origin()`` it went the other way and called a *targeted* managed
    host degraded, when nothing degrades for a target — it fails loud instead.
    Both readings put ``configured: false``, ``active: true`` and
    ``degraded: false`` in one ``host list`` payload about one host.
    """
    hosts.set_persisted_origin(hosts.MANAGED)

    # The pointer really did move the routing: asked for managed, running local.
    assert (hosts.selected_origin(), hosts.current_origin()) == (
        hosts.MANAGED,
        hosts.LOCAL,
    )
    assert hosts.origin_degraded() is True

    # An override is a target, so it is not degraded to anything.
    hosts.set_current_origin(hosts.MANAGED)
    assert hosts.current_origin() == hosts.MANAGED
    assert hosts.origin_degraded() is False


def test_an_overridden_origin_is_a_target_rather_than_a_preference(home) -> None:
    """Only the persisted pointer degrades.

    Degrading an override too is what made ``--host managed`` — and
    ``use x --managed``, and ``--pot managed:x`` — run against the *local* graph
    and report success under the managed label. Left alone it reaches
    ``build_host``, which refuses by name; ``--host`` is refused a step earlier
    still, at the CLI door, where the error contract can render it.
    """
    hosts.set_current_origin(hosts.MANAGED)

    assert hosts.current_origin() == hosts.MANAGED
    # Exactly the chain `_common._ActiveHost` walks on the next attribute read.
    with pytest.raises(ContextEngineDisabled, match="No managed host is configured"):
        hosts.build_host(hosts.current_origin())


# --- the environment override ----------------------------------------------


def test_an_unusable_environment_override_is_reported_not_counted_as_configured(
    home, monkeypatch
) -> None:
    """The one door into the registry that skips `host set`'s validation. A
    stray character in the port reported `configured: true` at exit 0 for an
    address every later command fails on, named nowhere."""
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://127.0.0.1:8090x")

    assert hosts.managed_endpoint() is None
    assert hosts.configured_origins() == (hosts.LOCAL,)

    result = _run("--json", "host", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    managed = next(r for r in payload["hosts"] if r["origin"] == hosts.MANAGED)
    assert managed["configured"] is False
    assert "POTPIE_MANAGED_URL" in managed["problem"]
    assert "port" in managed["problem"]
    # Not routed at *and* not silent — the pair is the point.
    assert "POTPIE_MANAGED_URL" in _run("host", "list").output


def test_an_unusable_managed_host_is_still_a_targeting_candidate(
    home, monkeypatch
) -> None:
    """The two candidate sets answer two different questions.

    "Worth talking to" excludes a managed address that will not parse, and has
    to: nothing may route at it. "Has to be accounted for before a bare ref is
    resolved" does not, because dropping it there asserts that no *other* host
    holds a pot by that name — which a host nobody can enumerate cannot be shown
    to say. Collapsed into one answer, the stray character in the port below
    turned every bare ref into a confident local selection.
    """
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://127.0.0.1:8090x")

    assert hosts.configured_origins() == (hosts.LOCAL,)
    assert hosts.targeting_origins() == hosts.ORIGINS


def test_a_usable_and_an_absent_managed_host_agree_across_both_sets(
    home, monkeypatch
) -> None:
    """Only the unusable middle state differs; the two ends must not drift."""
    assert hosts.configured_origins() == hosts.targeting_origins() == (hosts.LOCAL,)

    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://127.0.0.1:8099")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "tok")

    assert hosts.configured_origins() == hosts.targeting_origins() == hosts.ORIGINS


def test_a_usable_environment_override_still_wins_over_the_file(
    home, configured, monkeypatch
) -> None:
    """Regression guard for the validation above: the scratch-service override
    is the whole reason it exists."""
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://scratch.example:9000/")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "scratch-token")

    assert hosts.managed_endpoint() == ("http://scratch.example:9000", "scratch-token")
    assert hosts.managed_endpoint_problem() is None


# --- POTPIE_MANAGED_URL without POTPIE_MANAGED_TOKEN ------------------------


def test_an_address_only_override_keeps_the_token_stored_for_that_address(
    home, configured, monkeypatch
) -> None:
    """The override names the address; it does not revoke the credential.

    ``POTPIE_MANAGED_URL`` alone dropped the stored token on the floor and the
    endpoint came back with ``""``, which the transport turns into the
    auth-disabled placeholder: every command went out as ``Bearer no-auth``
    against a service holding a real key, and the 401 that came back read as the
    managed host refusing a login the CLI had never made.
    """
    monkeypatch.setenv("POTPIE_MANAGED_URL", f"{_STORED_URL}/")

    assert hosts.managed_endpoint() == (_STORED_URL, _STORED_TOKEN)
    assert hosts.managed_endpoint_problem() is None
    # Through the transport seam, because that is where the placeholder is
    # applied and where the wrong credential would actually have gone out.
    discovery = hosts.build_host(hosts.MANAGED).rpc.daemon.discovery()
    assert discovery == {"base_url": _STORED_URL, "token": _STORED_TOKEN}
    assert discovery["token"] != hosts.NO_AUTH_TOKEN


def test_an_address_only_override_for_another_host_refuses_to_guess(
    home, configured, monkeypatch
) -> None:
    """A key pasted in for one service is not a key handed to whatever else the
    environment names, and the placeholder is not a credential to fall back on.

    So neither of the two silent answers is available: the endpoint is reported
    unusable, nothing routes at it, and the message names both addresses and the
    variable that settles it.
    """
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://scratch.example:9000")

    assert hosts.managed_endpoint() is None
    problem = hosts.managed_endpoint_problem()
    assert problem is not None
    assert "http://scratch.example:9000" in problem
    assert _STORED_URL in problem
    assert "POTPIE_MANAGED_TOKEN" in problem
    # The token is still in the file: this refuses to *send* it, not to keep it.
    assert (home / "cli_hosts.json").read_bytes() == configured

    # Enumeration degrades — `host list` is the command you run to find this out.
    listing = _run("--json", "host", "list")
    assert listing.exit_code == 0, listing.output
    assert json.loads(listing.output)["problem"] == problem

    # Targeting fails loud, and not with "run 'potpie host set <url>'": writing
    # the file is exactly what the environment override makes irrelevant.
    refused = _run("--json", "host", "use", "managed")
    assert refused.exit_code == _common.EXIT_VALIDATION, refused.output
    error = json.loads(refused.output)
    assert error["code"] == "validation_error"
    assert problem in error["message"]
    assert "potpie host set" not in error["recommended_next_action"]


def test_an_explicit_empty_environment_token_is_the_auth_disabled_setup(
    home, configured, monkeypatch
) -> None:
    """Set-to-empty is a statement about the credential; unset is not one.

    This is the escape hatch the refusal above points at, so it has to work
    without touching the stored pair.
    """
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://scratch.example:9000")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "")

    assert hosts.managed_endpoint() == ("http://scratch.example:9000", "")
    assert hosts.managed_endpoint_problem() is None
    assert (
        hosts.build_host(hosts.MANAGED).rpc.daemon.discovery()["token"]
        == hosts.NO_AUTH_TOKEN
    )


def test_an_address_only_override_with_nothing_stored_stays_the_scratch_setup(
    home, monkeypatch
) -> None:
    """Regression guard for the refusal: with no stored token there is nothing
    to discard, and pointing at an auth-disabled service must stay a one-liner."""
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://scratch.example:9000")

    assert hosts.managed_endpoint() == ("http://scratch.example:9000", "")
    assert hosts.managed_endpoint_problem() is None


def test_host_set_says_when_the_environment_shadows_what_it_just_wrote(
    home, monkeypatch
) -> None:
    """The write lands, and is still not what the next command will use.

    ``POTPIE_MANAGED_URL`` outranks the file in ``_resolve_managed``, so
    ``managed host → <url>`` at exit 0 while every later command talks to the
    environment's host is the same silence as writing to the wrong file. The
    write is not the lie; saying nothing about the override is.
    """
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://env.example:9000")
    # The environment names the credential as well as the address: an override
    # that names only the address is a different situation with its own refusal
    # (see the POTPIE_MANAGED_TOKEN block below), and this test is about the
    # write landing somewhere nothing will read.
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "env-token")

    result = _run(
        "--json", "host", "set", "http://svc.example:8090", "--no-check", "--token", "t"
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["endpoint"] == "http://svc.example:8090"
    assert payload["shadowed_by_env"] == "http://env.example:9000"
    # The file really was written; what it says is that the file is not in use.
    assert hosts.stored_managed_token() == "t"
    endpoint = hosts.managed_endpoint()
    assert endpoint is not None and endpoint[0] == "http://env.example:9000"
    # Said on the human surface too, since that is where the belief forms.
    human = _run(
        "host", "set", "http://svc.example:8090", "--no-check", "--token", "t"
    ).output
    assert "POTPIE_MANAGED_URL" in human


def test_host_set_stays_quiet_when_nothing_shadows_the_write(home) -> None:
    """Regression guard for the warning above: a plain setup says nothing extra,
    or the note becomes noise nobody reads on the run that matters."""
    result = _run(
        "--json", "host", "set", "http://svc.example:8090", "--no-check", "--token", "t"
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["shadowed_by_env"] is None
    human = _run(
        "host", "set", "http://svc.example:8090", "--no-check", "--token", "t"
    ).output
    assert "POTPIE_MANAGED_URL" not in human


# --- `--host managed` with no usable managed host ---------------------------


def test_host_override_refuses_an_unconfigured_managed_host(home) -> None:
    """The registry is empty, so ``host set`` is the repair and it is named."""
    result = _run("--json", "--host", "managed", "host", "list")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert payload["message"] == "No managed host is configured."
    assert "host set" in payload["recommended_next_action"]


def test_host_override_names_an_unusable_override_instead_of_the_file(
    home, monkeypatch
) -> None:
    """There are two ways to have no usable managed host; only one is a file.

    A configured-but-unusable ``POTPIE_MANAGED_URL`` reads as *absent* to every
    router by design, and this door answered it with "run 'potpie host set
    <url>'" — telling a caller whose environment override outranks the file to
    go and write the file, which the next command would ignore for the same
    reason. ``host use managed`` and ``build_host`` already take their text from
    the shared helper; this was the last call site that did not.
    """
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://127.0.0.1:8090x")

    result = _run("--json", "--host", "managed", "host", "list")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "unusable" in payload["message"]
    assert "POTPIE_MANAGED_URL" in payload["message"]
    assert "host set" not in payload["recommended_next_action"]
