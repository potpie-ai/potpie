"""Managed-host routing (HU3): two hosts, one CLI.

The risk this feature carries is not "does the listing render" — it is running a
command against the wrong graph because a bare name existed on both hosts, or
silently falling back to the local host when the remote is down. Most of what is
pinned here is refusals.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli import hosts
from potpie.cli.commands import _common
from potpie.cli.main import app


class _Pot:
    def __init__(self, pot_id: str, name: str, active: bool = False) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = active


class _Pots:
    def __init__(self, pots: list[_Pot], *, raises: Exception | None = None) -> None:
        self._pots = pots
        self._raises = raises
        self.used: list[str] = []

    def list_pots(self) -> list[_Pot]:
        if self._raises is not None:
            raise self._raises
        return self._pots

    def use_pot(self, *, ref: str) -> _Pot:
        self.used.append(ref)
        for pot in self._pots:
            if ref in (pot.pot_id, pot.name):
                return pot
        from potpie_context_core.errors import PotNotFound

        raise PotNotFound(f"No pot matching '{ref}'.")

    def active_pot(self) -> _Pot | None:
        return next((p for p in self._pots if p.active), None)

    def set_repo_default(self, **_kwargs: Any) -> None:
        return None

    def list_repo_defaults(self) -> dict[str, str]:
        return {}

    def list_repo_sources(self) -> list[Any]:
        return []


class _Host:
    def __init__(self, pots: _Pots) -> None:
        self.pots = pots


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """A two-host registry: local and a configured managed endpoint."""
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(
        hosts, "managed_endpoint", lambda: ("http://svc.example", "tok")
    )

    built: dict[str, _Host] = {
        hosts.LOCAL: _Host(
            _Pots([_Pot("pot_l1", "default", active=True), _Pot("pot_l2", "notes")])
        ),
        hosts.MANAGED: _Host(
            _Pots([_Pot("pot_m1", "default"), _Pot("pot_m2", "api", active=True)])
        ),
    }
    monkeypatch.setattr(hosts, "build_host", lambda origin: built[origin])
    yield built
    _common.set_host(None)
    hosts.reset_for_tests()


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


# --- ref parsing ----------------------------------------------------------


def test_only_a_known_origin_counts_as_a_prefix() -> None:
    assert hosts.split_ref("managed:api") == ("managed", "api")
    assert hosts.split_ref("local:default") == ("local", "default")
    # A pot may legitimately be named with a colon; stealing every prefix would
    # make such a name unaddressable.
    assert hosts.split_ref("team:api") == (None, "team:api")
    assert hosts.split_ref("api") == (None, "api")
    assert hosts.split_ref("managed:") == (None, "managed:")


def test_a_persisted_origin_that_is_no_longer_configured_degrades(
    monkeypatch, tmp_path
) -> None:
    """Clearing a login must not brick every command."""
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: ("http://svc", "tok"))
    hosts.set_persisted_origin(hosts.MANAGED)
    assert hosts.current_origin() == hosts.MANAGED

    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    assert hosts.current_origin() == hosts.LOCAL


# --- listing --------------------------------------------------------------


def test_pot_list_merges_both_hosts_and_tags_origin(registry) -> None:
    result = _run("--json", "pot", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [(row["origin"], row["name"]) for row in payload["pots"]] == [
        ("local", "default"),
        ("local", "notes"),
        ("managed", "default"),
        ("managed", "api"),
    ]
    # Each host keeps its own pointer; only the active origin's is current.
    current = [row["name"] for row in payload["pots"] if row["current"]]
    active = [row["name"] for row in payload["pots"] if row["active"]]
    assert current == ["default"]
    assert sorted(active) == ["api", "default"]


def test_pot_list_local_flag_does_not_touch_the_remote(registry) -> None:
    registry[hosts.MANAGED].pots = _Pots(
        [], raises=AssertionError("managed must not be listed")
    )

    result = _run("--json", "pot", "list", "--local")

    assert result.exit_code == 0, result.output
    assert {row["origin"] for row in json.loads(result.output)["pots"]} == {"local"}


def test_pot_list_degrades_when_the_remote_is_down(registry) -> None:
    """A listing that dies because a remote is unreachable is useless exactly
    when you need to see what is local."""
    from potpie_context_core.errors import ContextEngineDisabled

    registry[hosts.MANAGED].pots = _Pots(
        [], raises=ContextEngineDisabled("connection refused")
    )

    result = _run("--json", "pot", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {row["origin"] for row in payload["pots"]} == {"local"}
    assert "connection refused" in payload["unavailable"]["managed"]


# --- routing --------------------------------------------------------------


def test_a_name_on_both_hosts_is_refused_rather_than_guessed(registry) -> None:
    """'default' exists on both. Picking one would eventually write to the
    wrong graph, so the ambiguity is the answer."""
    result = _run("--json", "graph", "catalog", "--pot", "default")

    assert result.exit_code == _common.EXIT_VALIDATION
    # `potpie graph` nests errors inside its own envelope.
    error = json.loads(result.output)["error"]
    assert error["code"] == "ambiguous_pot"
    assert "local:default" in error["message"]
    assert "managed:default" in error["message"]


def test_a_bare_name_unique_to_the_remote_resolves_there(registry) -> None:
    origin, pot_id = None, None

    def _capture(ref: str) -> str:
        nonlocal origin, pot_id
        pot_id = _common._resolve_explicit_pot(ref)
        origin = hosts.current_origin()
        return pot_id

    _capture("api")
    assert (origin, pot_id) == ("managed", "pot_m2")


def test_a_qualified_ref_moves_the_whole_command(registry) -> None:
    pot_id = _common._resolve_explicit_pot("managed:default")

    assert pot_id == "pot_m1"
    # Not just the id: the host the command goes on to use has to move with it,
    # or the id is looked up on one host and used against the other.
    assert hosts.current_origin() == "managed"


def test_a_qualified_ref_for_a_missing_pot_names_the_host(registry) -> None:
    result = _run("--json", "graph", "catalog", "--pot", "managed:nope")

    assert result.exit_code == _common.EXIT_VALIDATION
    error = json.loads(result.output)["error"]
    assert error["code"] == "pot_not_found"
    assert "managed host" in error["message"]


def test_use_moves_the_active_origin_and_persists_it(registry, tmp_path) -> None:
    result = _run("--json", "pot", "use", "managed:api")

    assert result.exit_code == 0, result.output
    assert registry[hosts.MANAGED].pots.used == ["api"]
    assert registry[hosts.LOCAL].pots.used == []
    assert hosts.persisted_origin() == "managed"
    assert (
        json.loads((tmp_path / "cli_hosts.json").read_text())["active_origin"]
        == "managed"
    )


def test_a_failed_use_does_not_move_the_pointer(registry) -> None:
    """Otherwise a typo strands the CLI on a host the user never chose.

    The envelope is asserted alongside the pointer, because ``!= 0`` on its own
    pins nothing here: a crash, a refusal raised long before the pointer was
    reached, and the guard actually working all satisfy it, and the pointer sits
    still in every one of them for unrelated reasons.
    """
    before = hosts.persisted_origin()

    result = _run("--json", "pot", "use", "managed:nope")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert "nope" in payload["message"]
    # The refusal came from the host the ref named — the pointer-move is the
    # step *after* this one, which is the step being pinned.
    assert registry[hosts.MANAGED].pots.used == ["nope"]
    assert registry[hosts.LOCAL].pots.used == []
    assert hosts.persisted_origin() == before


def test_host_flag_targets_one_invocation_without_persisting(registry) -> None:
    result = _run("--json", "--host", "managed", "pot", "list", "--managed")

    assert result.exit_code == 0, result.output
    assert {row["origin"] for row in json.loads(result.output)["pots"]} == {"managed"}
    assert hosts.persisted_origin() == "local"


def test_an_unknown_host_flag_is_rejected_at_the_door(registry) -> None:
    """A typo should not read as 'that host has no pots' — nor as a traceback.

    Asserting only "nonzero" sat straight over the live defect. The check ran in
    the Typer root callback, outside any ``contract()`` and above
    ``configure_error_output``, and ``hosts.require_origin`` raises a bare
    ``ValueError`` that ``run_cli`` does not catch: ``--json --host bogus``
    produced a rendered traceback and an empty stdout at exit 1. An envelope was
    promised and a traceback delivered, and both are nonzero.
    """
    result = _run("--json", "--host", "prod", "pot", "list")

    # The pre-fix shape, named directly: the ValueError escaped the callback.
    assert not isinstance(result.exception, ValueError), result.exception
    assert "Traceback" not in result.output
    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "prod" in payload["message"]
    assert "potpie host list" in payload["recommended_next_action"]


def test_the_host_flag_refuses_a_managed_host_that_is_not_configured(
    monkeypatch, tmp_path
) -> None:
    """Naming a host is targeting, and targeting fails loud.

    Degraded to local instead, this exited 0 with the *local* pots in the
    listing under ``"active_origin": "managed"`` and no ``unavailable`` key —
    and since pot names are per-host labels, a ``default`` answering from the
    wrong graph looks exactly like the managed host answering.
    """
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    local = _Host(_Pots([_Pot("pot_l1", "default", active=True)]))

    def _build(origin: str) -> _Host:
        if origin == hosts.MANAGED:
            raise AssertionError("there is no managed host to build")
        return local

    monkeypatch.setattr(hosts, "build_host", _build)
    try:
        result = _run("--json", "--host", "managed", "pot", "list")
    finally:
        _common.set_host(None)
        hosts.reset_for_tests()

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "managed" in payload["message"]
    assert "potpie host set" in payload["recommended_next_action"]
    # Nothing was listed at all: the refusal is the whole outcome.
    assert "pots" not in payload


# --- integration auth is this machine's ------------------------------------


@pytest.fixture
def local_integrations(monkeypatch):
    """Fake the one adapter that reads the machine's credential files.

    Only the outermost read is stubbed: the command, its flag handling and the
    host registry are the real ones, because the defect was that the *command*
    ignored the host it was pointed at, not that the store answered wrongly.
    """
    asked: list[str] = []

    def _status(provider: str) -> dict[str, Any]:
        asked.append(provider)
        return {"provider": provider, "authenticated": provider == "github"}

    from potpie.cli.auth import auth_commands

    monkeypatch.setattr(auth_commands, "get_integration_status", _status)
    return asked


def test_auth_status_refuses_to_report_this_machine_under_another_host(
    registry, local_integrations
) -> None:
    """Silent wrong-host reporting is the worst of the three options.

    ``potpie --host managed auth status`` listened to the flag not at all: it
    read this laptop's GitHub and Linear logins and printed them at exit 0 under
    a managed target, with nothing in the output naming which host answered.
    Integration credentials are local OAuth/API-token files and a managed host
    holds none, so naming one is refused rather than answered from elsewhere.
    """
    result = _run("--json", "--host", "managed", "auth", "status")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "cannot target" in payload["message"]
    # Refused *before* the credential store was touched: a refusal that still
    # reads the keychain has already done the thing it is refusing to report.
    assert local_integrations == []


def test_auth_status_still_answers_for_the_machine_it_is_run_on(
    registry, local_integrations
) -> None:
    """Regression guard for the refusal above: no flag is not a managed target,
    and neither is naming the local host."""
    for args in (
        ("--json", "auth", "status"),
        ("--json", "--host", "local", "auth", "status"),
    ):
        result = _run(*args)

        assert result.exit_code == 0, result.output
        providers = [
            row["provider"] for row in json.loads(result.output)["integrations"]
        ]
        assert "github" in providers


def test_auth_status_keeps_working_for_a_persisted_managed_pointer(
    registry, local_integrations, monkeypatch
) -> None:
    """Only an *explicit* target refuses.

    Someone whose active host is managed still has to be able to see and repair
    the credentials on the machine they are typing on — the same rule ``setup``
    already follows.
    """
    monkeypatch.setattr(hosts, "persisted_origin", lambda: hosts.MANAGED)

    result = _run("--json", "auth", "status")

    assert result.exit_code == 0, result.output
    assert local_integrations != []


# --- configuring the managed endpoint -------------------------------------


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A throwaway CLI home, with the endpoint helpers left real.

    The env override is cleared too: ``POTPIE_MANAGED_URL`` in a developer's
    shell would otherwise answer for the file every assertion here is about.
    """
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.delenv("POTPIE_MANAGED_URL", raising=False)
    monkeypatch.delenv("POTPIE_MANAGED_TOKEN", raising=False)
    yield tmp_path
    _common.set_host(None)
    hosts.reset_for_tests()


def test_host_set_persists_the_endpoint_with_credential_permissions(home) -> None:
    """The file can hold a token, so it must not be world-readable."""
    result = _run("--json", "host", "set", "http://svc.example:8090/", "--no-check")

    assert result.exit_code == 0, result.output
    # Stored verbatim: an auth-disabled service has no token, and the
    # placeholder that lets the RPC client accept that belongs at the transport
    # seam, not in the registry (see test_cli_host_registry).
    assert hosts.managed_endpoint() == ("http://svc.example:8090", "")
    state = home / "cli_hosts.json"
    assert state.stat().st_mode & 0o777 == 0o600


def test_an_endpoint_without_a_token_still_counts_as_configured(home) -> None:
    """A service with auth disabled accepts any token and means none. Treating
    the endpoint as unconfigured because the token is blank would make the
    no-auth dev shape impossible to point at."""
    hosts.set_managed_endpoint("http://svc.example:8090", None)

    endpoint = hosts.managed_endpoint()
    assert endpoint is not None and endpoint[0] == "http://svc.example:8090"
    assert hosts.configured_origins() == hosts.ORIGINS


def test_host_set_rolls_back_an_endpoint_it_cannot_reach(home, monkeypatch) -> None:
    """A configured-but-dead host degrades every listing from then on, so a
    failed check must not leave one behind — nor take the working one with it.

    The failure is injected at ``build_managed_host``, the seam ``host set``
    probes through. Stubbing anything further out let this test pass because the
    endpoint was refused *before* the network rather than by it, which is how it
    kept passing while the auth-disabled setup was broken — and would send the
    suite off to resolve nope.example for real once that was fixed.
    """
    hosts.set_managed_endpoint("http://svc.example:8090", "s3secrettoken")
    stored = (home / "cli_hosts.json").read_bytes()

    def _refused(base_url: str, token: str):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(hosts, "build_managed_host", _refused)

    result = _run("--json", "host", "set", "http://nope.example:9999", "--token", "new")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    assert (home / "cli_hosts.json").read_bytes() == stored
    assert hosts.managed_endpoint() == ("http://svc.example:8090", "s3secrettoken")


def test_host_clear_keeps_the_rest_of_the_state(home) -> None:
    hosts.set_persisted_origin(hosts.MANAGED)
    hosts.set_managed_endpoint("http://svc.example:8090", "tok")

    hosts.clear_managed_endpoint()

    assert hosts.managed_endpoint() is None
    # The selection survives so re-adding the host restores it, and
    # current_origin() degrades to local meanwhile.
    assert hosts.persisted_origin() == hosts.MANAGED
    assert hosts.current_origin() == hosts.LOCAL


def test_the_environment_overrides_the_stored_endpoint(home, monkeypatch) -> None:
    hosts.set_managed_endpoint("http://stored.example", "stored-token")
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://scratch.example")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "scratch-token")

    assert hosts.managed_endpoint() == ("http://scratch.example", "scratch-token")


def test_host_list_reports_both_origins(home) -> None:
    hosts.set_managed_endpoint("http://svc.example:8090", "")

    result = _run("--json", "host", "list")

    assert result.exit_code == 0, result.output
    rows = {row["origin"]: row for row in json.loads(result.output)["hosts"]}
    assert rows["local"]["configured"] is True
    assert rows["managed"]["endpoint"] == "http://svc.example:8090"


# --- the explorer ---------------------------------------------------------


class _Daemon:
    """Just enough of the local daemon for `potpie ui` to find its base URL."""

    in_process = False

    def __init__(self) -> None:
        self.ensured = 0

    def ensure(self, *_args: Any) -> None:
        self.ensured += 1

    def discovery(self) -> dict[str, str]:
        return {"base_url": "http://127.0.0.1:8765", "token": "t"}


@pytest.fixture
def registry_with_daemon(registry, monkeypatch):
    """The two-host registry, with a daemon on the local host only.

    Mirrors production: the managed host is a remote service with no process
    here, so `StaticDiscovery` refuses `daemon` attributes outright.
    """
    from potpie.cli.commands import ui as ui_cmds

    # Arity-agnostic: the probe's own arguments are not what these tests pin.
    monkeypatch.setattr(ui_cmds, "_probe_ui", lambda *_args: None)
    # Both daemon round-trips get stubbed, not just the first. `_probe_ui`
    # returning None is what makes `ui` go on to ask for a handoff code, so
    # stubbing one and not the other sent a real POST at 127.0.0.1:8765 —
    # instantly refused here, and a live socket on whatever is listening there.
    # No code, as from a daemon predating the route: what these tests pin is
    # which host the URL names, and a session key in it is noise.
    monkeypatch.setattr(ui_cmds, "_handoff_code", lambda *_args: (None, None))
    registry[hosts.LOCAL].daemon = _Daemon()
    return registry


def test_ui_serves_a_managed_pot_from_the_local_daemon(registry_with_daemon) -> None:
    """The token stays here. A managed endpoint needs a bearer header that a
    browser navigation cannot send and that has no business in a URL bar, so
    the page is served from loopback and the daemon proxies the reads."""
    hosts.set_current_origin(hosts.MANAGED)

    result = _run("--json", "ui", "--no-open")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["origin"] == "managed"
    assert payload["url"] == "http://127.0.0.1:8765/ui?host=managed"
    assert registry_with_daemon[hosts.LOCAL].daemon.ensured == 1


def test_ui_pot_ref_names_the_host_it_resolved_on(registry_with_daemon) -> None:
    """The id and the host travel together: 'pot_m2' read against the local
    graph is either a miss or, worse, a different pot."""
    result = _run("--json", "ui", "--no-open", "--pot", "managed:api")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pot_id"] == "pot_m2"
    assert payload["origin"] == "managed"
    assert payload["url"] == "http://127.0.0.1:8765/ui?host=managed&pot=pot_m2"


def test_ui_falls_back_to_the_local_daemon_url_not_the_managed_endpoint(
    registry_with_daemon,
) -> None:
    """Regression guard: the serving base URL must never come from the active
    host, or a managed session would send the browser to a service that speaks
    RPC and serves no page."""
    hosts.set_current_origin(hosts.MANAGED)

    payload = json.loads(_run("--json", "ui", "--no-open").output)

    assert payload["url"].startswith("http://127.0.0.1:8765/")


def test_daemon_commands_stay_local_while_managed_is_active(
    registry_with_daemon,
) -> None:
    """`potpie daemon restart` has to keep working when you are pointed at a
    managed pot — that is exactly when a broken local daemon shows up."""
    from potpie.cli.commands import daemon as daemon_cmds

    hosts.set_current_origin(hosts.MANAGED)

    assert daemon_cmds._detached_daemon() is registry_with_daemon[hosts.LOCAL].daemon


def test_ui_warns_when_the_running_daemon_predates_host_routing(
    registry, monkeypatch
) -> None:
    """The daemon is long-lived, so after an upgrade the process still running
    is the old one. It answers /ui/api/pots fine — with local pots only — so
    the managed graph would go missing silently rather than be reported."""
    import httpx

    registry[hosts.LOCAL].daemon = _Daemon()

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {"pots": [{"id": "p1", "name": "default"}], "active": None}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    hosts.set_current_origin(hosts.MANAGED)

    payload = json.loads(_run("--json", "ui", "--no-open").output)

    assert "daemon restart" in (payload["warning"] or "")


def test_skills_install_stays_on_the_local_machine(registry, monkeypatch) -> None:
    """A skill install writes into this machine's harness directory. Routed to
    a managed host it would write onto the server, where no harness of yours
    will ever read it, and report success.

    Asserted as "no host at all", which is stronger than the old "the local
    host": the manager is now built in process, so the locality is structural
    rather than a routing decision that could be made differently. It also means
    skills work on a remote-only install, where there is no local daemon to ask.
    """
    from potpie_context_engine.bootstrap import host_wiring

    built: list[str] = []

    class _Skills:
        def status(self, **_kwargs: Any) -> Any:
            built.append("in_process")

            class _St:
                agent = "claude"
                installed = ()
                missing = ()
                outdated = ()

            return _St()

    def _no_host(origin: str) -> Any:
        raise AssertionError(f"skills must not build the {origin} host")

    monkeypatch.setattr(host_wiring, "build_skill_manager", lambda: _Skills())
    monkeypatch.setattr(hosts, "build_host", _no_host)
    hosts.set_current_origin(hosts.MANAGED)

    result = _run("--json", "skills", "status")

    assert result.exit_code == 0, result.output
    assert built == ["in_process"]
