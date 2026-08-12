"""`potpie use` origin selection: the flags pick the host, not the wording.

`--local` / `--managed` used to be bound and never read — the origin was
resolved against whichever host happened to be current and the flag survived
only as the label printed afterwards, so `potpie use shop --local` from a
managed session selected the *managed* `shop` and reported "(local)". Everything
here is about that class of failure: the origin the CLI says it used has to be
the origin it actually used, an explicit origin is a target rather than a
starting point, and anything the CLI cannot decide is a refusal.
"""

from __future__ import annotations

import json
import socket
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from potpie.cli import hosts
from potpie.cli.commands import _common
from potpie.cli.main import app


class _SocketOpenedInUnitTest(BaseException):
    """Outside ``Exception`` on purpose — the CLI swallows that hierarchy.

    ``_find_pot_in`` and every transport helper wrap their calls in
    ``except Exception``, so a refusal raised as one would be turned into a
    degrade or a warning and the test would pass having already dialled out.
    """


@pytest.fixture(autouse=True)
def _no_sockets(monkeypatch):
    """Refuse any connection attempt from this module.

    Only some tests here replace :func:`hosts.build_host`; the rest lean on a
    refusal landing *before* a host is built. When one of those refusals
    regresses — as it did while this was written — ``build_host`` falls through
    and builds a real host for the ref instead. What stops that from becoming
    an RPC today is a single line in ``tests/conftest.py``
    (``CONTEXT_ENGINE_HOST_MODE=in_process``); run the same sequence outside
    pytest and it reaches the developer's running daemon and calls the
    *mutating* ``use_pot`` on it.

    That env var is not a guarantee this file can rely on, because the bypass
    is one fake away: ``test_cli_ergonomics.py``'s fake discovery record names
    ``http://127.0.0.1:8765`` directly, and an in-process host mode has no
    opinion about a URL a test hands the transport. So the socket is asserted
    on here as its own invariant, independently of which refusal any one test
    is about — and loopback most of all, because that is the address a unit
    test reaches by accident and the one where the side effects are real.
    """

    def _refuse(*args, **_kwargs):
        raise _SocketOpenedInUnitTest(f"unit test opened a socket: {args!r}")

    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)


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
        # Enumeration is counted as well as selection: answering "which pots do
        # you have" for a host you are not is already the wrong-host failure,
        # and it happens a step before `use_pot` would record anything.
        self.list_calls = 0

    def list_pots(self) -> list[_Pot]:
        self.list_calls += 1
        if self._raises is not None:
            raise self._raises
        return self._pots

    def use_pot(self, *, ref: str) -> _Pot:
        self.used.append(ref)
        for pot in self._pots:
            if ref in (pot.pot_id, pot.name):
                pot.active = True
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
    """Two configured hosts, each with a pot named ``shared``.

    The shared name is the whole point: it is what makes a wrong-host selection
    silent, because both hosts answer the ref and only the id differs.
    """
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(
        hosts, "managed_endpoint", lambda: ("http://svc.example", "tok")
    )

    built: dict[str, _Host] = {
        hosts.LOCAL: _Host(
            _Pots(
                [_Pot("pot_l_shared", "shared", active=True), _Pot("pot_l2", "notes")]
            )
        ),
        hosts.MANAGED: _Host(
            _Pots([_Pot("pot_m_shared", "shared", active=True), _Pot("pot_m2", "api")])
        ),
    }
    monkeypatch.setattr(hosts, "build_host", lambda origin: built[origin])
    yield built
    _common.set_host(None)
    hosts.reset_for_tests()


@pytest.fixture
def injected_session(monkeypatch, tmp_path):
    """One injected host, in a session whose current origin is ``managed``.

    The mismatch is the point. An injected host stands in for the whole
    registry, so it *is* the managed host here — which means the origin the CLI
    reports has to be read from the session rather than assumed, and any other
    origin has no host behind it at all.
    """
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(
        hosts, "managed_endpoint", lambda: ("http://svc.example", "tok")
    )
    hosts.set_persisted_origin(hosts.MANAGED)
    injected = _Host(_Pots([_Pot("pot_x", "shared")]))
    _common.set_host(injected)
    yield injected
    _common.set_host(None)
    hosts.reset_for_tests()


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


# --- the flags select the host --------------------------------------------


def test_local_flag_selects_the_local_pot_from_a_managed_session(
    registry, tmp_path
) -> None:
    """The reported defect: `--local` while managed was active selected the
    managed pot of the same name and still printed "(local)"."""
    hosts.set_persisted_origin(hosts.MANAGED)

    result = _run("--json", "use", "shared", "--local")

    assert result.exit_code == 0, result.output
    # Assert on the host that was actually asked, not on the id alone: an id
    # read off the wrong host is the failure being guarded against.
    assert registry[hosts.LOCAL].pots.used == ["shared"]
    assert registry[hosts.MANAGED].pots.used == []
    payload = json.loads(result.output)
    assert payload["id"] == "pot_l_shared"
    assert payload["origin"] == "local"
    assert hosts.persisted_origin() == "local"
    assert (
        json.loads((tmp_path / "cli_hosts.json").read_text())["active_origin"]
        == "local"
    )


def test_managed_flag_selects_the_managed_pot_from_a_local_session(
    registry, tmp_path
) -> None:
    hosts.set_persisted_origin(hosts.LOCAL)

    result = _run("--json", "use", "shared", "--managed")

    assert result.exit_code == 0, result.output
    assert registry[hosts.MANAGED].pots.used == ["shared"]
    assert registry[hosts.LOCAL].pots.used == []
    payload = json.loads(result.output)
    assert payload["id"] == "pot_m_shared"
    assert payload["origin"] == "managed"
    assert hosts.persisted_origin() == "managed"
    assert (
        json.loads((tmp_path / "cli_hosts.json").read_text())["active_origin"]
        == "managed"
    )


def test_the_flag_is_equivalent_to_qualifying_the_ref(registry) -> None:
    hosts.set_persisted_origin(hosts.MANAGED)

    flagged = json.loads(_run("--json", "use", "shared", "--local").output)
    hosts.reset_for_tests()
    hosts.set_persisted_origin(hosts.MANAGED)
    qualified = json.loads(_run("--json", "use", "local:shared").output)

    assert flagged["id"] == qualified["id"] == "pot_l_shared"
    assert flagged["origin"] == qualified["origin"] == "local"


def test_an_explicit_origin_never_falls_through_to_the_other_host(registry) -> None:
    """`notes` exists only locally. `--managed` means "on the managed host", so
    it has to fail there rather than quietly find the local one."""
    result = _run("--json", "use", "notes", "--managed")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert registry[hosts.LOCAL].pots.used == []
    assert hosts.persisted_origin() == "local"


def test_the_human_line_names_the_origin_that_was_resolved(registry) -> None:
    hosts.set_persisted_origin(hosts.MANAGED)

    result = _run("use", "shared", "--local")

    assert result.exit_code == 0, result.output
    assert "shared (local)" in result.output
    assert registry[hosts.LOCAL].pots.used == ["shared"]


# --- refusals --------------------------------------------------------------


def test_both_flags_at_once_is_refused(registry) -> None:
    result = _run("--json", "use", "shared", "--local", "--managed")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "--local" in payload["message"] and "--managed" in payload["message"]
    assert registry[hosts.LOCAL].pots.used == []
    assert registry[hosts.MANAGED].pots.used == []


def test_a_flag_contradicting_the_qualified_ref_is_refused(registry) -> None:
    result = _run("--json", "use", "managed:shared", "--local")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "--local" in payload["message"]
    assert "managed:shared" in payload["message"]
    assert registry[hosts.LOCAL].pots.used == []
    assert registry[hosts.MANAGED].pots.used == []


def test_managed_without_a_configured_host_reports_the_configured_error(
    monkeypatch, tmp_path
) -> None:
    """No managed endpoint is a configuration gap with a known repair, not an
    unimplemented capability — the old branch answered "managed routing lands
    in HU3" while managed routing worked."""
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    try:
        result = _run("--json", "use", "shared", "--managed")
    finally:
        hosts.reset_for_tests()

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert "No managed host is configured" in payload["message"]
    assert "HU3" not in result.output
    assert "Traceback" not in result.output


def test_a_bare_ref_on_both_hosts_is_refused_naming_both(registry) -> None:
    result = _run("--json", "use", "shared")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "ambiguous_pot"
    assert "local:shared" in payload["message"]
    assert "managed:shared" in payload["message"]
    assert registry[hosts.LOCAL].pots.used == []
    assert registry[hosts.MANAGED].pots.used == []


def test_the_ambiguity_refusal_names_the_candidates_the_way_pot_list_does(
    monkeypatch, tmp_path
) -> None:
    """Both resolution paths share one refusal, so both name pots the same way.

    ``potpie use`` used to qualify the string the caller *typed*, which for a
    ref that matched by id answered ``local:pot_a, managed:pot_a`` — two lines
    that both work and neither of which can be matched against a pot anyone
    knows by name. ``--pot`` had always qualified by name. One helper now, so
    the wording cannot drift apart again the way the guards themselves did.
    """
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(
        hosts, "managed_endpoint", lambda: ("http://svc.example", "tok")
    )
    # The same string is an id on one host and a name on the other.
    built = {
        hosts.LOCAL: _Host(_Pots([_Pot("twin", "left")])),
        hosts.MANAGED: _Host(_Pots([_Pot("pot_m", "twin")])),
    }
    monkeypatch.setattr(hosts, "build_host", lambda origin: built[origin])
    try:
        used = _run("--json", "use", "twin")
        flagged = _run("--json", "graph", "catalog", "--pot", "twin")
    finally:
        hosts.reset_for_tests()

    for payload in (json.loads(used.output), json.loads(flagged.output)["error"]):
        assert payload["code"] == "ambiguous_pot"
        assert "local:left" in payload["message"]
        assert "managed:twin" in payload["message"]
    assert built[hosts.LOCAL].pots.used == []
    assert built[hosts.MANAGED].pots.used == []


def test_a_bare_ref_refuses_when_a_configured_host_cannot_be_enumerated(
    registry,
) -> None:
    """Read as "no match", an unreachable host collapses the ambiguity check to
    one candidate and the selection lands on the other graph in silence."""
    from potpie_context_core.errors import ContextEngineDisabled

    registry[hosts.MANAGED].pots = _Pots(
        [], raises=ContextEngineDisabled("connection refused")
    )

    result = _run("--json", "use", "notes")

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert "managed" in payload["message"]
    assert "potpie use local:notes" in payload["recommended_next_action"]
    assert registry[hosts.LOCAL].pots.used == []


def test_an_unreachable_host_still_lets_a_qualified_ref_through(registry) -> None:
    """The refusal above must be escapable, or a dead remote blocks local work."""
    from potpie_context_core.errors import ContextEngineDisabled

    registry[hosts.MANAGED].pots = _Pots(
        [], raises=ContextEngineDisabled("connection refused")
    )

    result = _run("--json", "use", "local:notes")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == "pot_l2"


def test_pot_list_still_degrades_when_a_host_is_unreachable(registry) -> None:
    """Regression guard for the fix above: targeting fails loud, enumeration
    keeps degrading, or `pot list` stops working offline."""
    from potpie_context_core.errors import ContextEngineDisabled

    registry[hosts.MANAGED].pots = _Pots(
        [], raises=ContextEngineDisabled("connection refused")
    )

    result = _run("--json", "pot", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {row["origin"] for row in payload["pots"]} == {"local"}
    assert "connection refused" in payload["unavailable"]["managed"]


def test_a_refused_selection_leaves_the_pointer_alone(registry) -> None:
    """The pointer survives *this* refusal — the ambiguity one.

    Asserting only "it failed" is what let two other tests in this exercise sit
    on top of live defects: a crash, a wrong-host answer that then errored, or a
    refusal raised long before the pointer was reached all satisfy it equally,
    and the pointer stays put in every one of them for reasons that have nothing
    to do with the guard being pinned. So the envelope is asserted too.
    """
    hosts.set_persisted_origin(hosts.MANAGED)

    result = _run("--json", "use", "shared")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "ambiguous_pot"
    assert "local:shared" in payload["message"]
    assert "managed:shared" in payload["message"]
    assert payload["recommended_next_action"] is not None
    assert registry[hosts.LOCAL].pots.used == []
    assert registry[hosts.MANAGED].pots.used == []
    assert hosts.persisted_origin() == "managed"


# --- a host that cannot even be built -------------------------------------


@pytest.fixture
def unusable_managed(monkeypatch, tmp_path):
    """A managed host that is configured and whose address will not parse.

    The interesting state, because it is *not* the same as no managed host: the
    user configured one, and the CLI cannot ask it anything. Every router reads
    it as absent on purpose — nothing may route at an address that cannot be
    parsed — which is exactly how it went missing from the candidate set of a
    bare ref one step before the unreachable-host refusal could see it.
    """
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://svc.example:80x90")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "tok")
    local = _Host(_Pots([_Pot("pot_l_shared", "shared", active=True)]))
    # Only the local origin is faked: building the managed one has to fail the
    # way it really fails, which is before a socket exists.
    real_build_host = hosts.build_host
    monkeypatch.setattr(
        hosts,
        "build_host",
        lambda origin: local if origin == hosts.LOCAL else real_build_host(origin),
    )
    yield local
    _common.set_host(None)
    hosts.reset_for_tests()


def test_a_bare_ref_refuses_while_a_configured_host_cannot_be_built(
    unusable_managed,
) -> None:
    """The reported invariant, reached a step earlier than an unreachable host.

    A managed address with a stray character in its port drops out of the
    candidate set before anything tries to read it, so the ambiguity check sees
    one candidate and ``potpie use shared`` selected the *local* ``shared`` at
    exit 0 — a confident targeting decision made from an incomplete candidate
    set, with the host that was never asked named nowhere in the output.
    """
    result = _run("--json", "use", "shared")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert "managed host" in payload["message"]
    # The repair is the address, and no socket was opened, so the message must
    # not claim the host could not be *reached*.
    assert "port" in payload["message"]
    assert "Cannot reach" not in payload["message"]
    assert ".." not in payload["message"]
    # Escapable, or a typo in one url blocks all local work.
    assert "potpie use local:shared" in payload["recommended_next_action"]
    assert unusable_managed.pots.used == []
    assert hosts.persisted_origin() == "local"


def test_a_qualified_ref_still_escapes_a_host_that_cannot_be_built(
    unusable_managed,
) -> None:
    """The refusal above must be escapable, or a stray character in one url
    locks the caller out of the local graph until they find the file."""
    result = _run("--json", "use", "local:shared")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"] == "pot_l_shared"
    assert payload["origin"] == "local"
    assert unusable_managed.pots.used == ["shared"]


def test_the_same_refusal_reaches_the_pot_flag(unusable_managed) -> None:
    """``--pot`` resolves through the same candidate set, so it must not keep
    silently answering from the one host that happens to be readable."""
    result = _run("--json", "graph", "catalog", "--pot", "shared")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    # `potpie graph` nests errors inside its own envelope.
    assert payload["error"]["code"] == "unavailable"
    assert "managed host" in payload["error"]["message"]
    assert "--pot local:shared" in payload["recommended_next_action"]


def test_pot_list_still_enumerates_around_a_host_that_cannot_be_built(
    unusable_managed,
) -> None:
    """Regression guard for the fix above, on the other side of the invariant:
    the targeting candidate set grew, the enumeration one did not."""
    result = _run("--json", "pot", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {row["origin"] for row in payload["pots"]} == {"local"}


# --- targeting a host that is down ----------------------------------------


def test_a_qualified_pot_ref_reports_the_unreachable_host_it_named(registry) -> None:
    """A ref that says ``managed:`` has already chosen a host, so a host that
    cannot be enumerated is that host being down — not the pot being absent.
    Reported as ``pot_not_found`` it sent the operator to ``pot list --managed``
    to hunt for a pot that was never missing, on the host that is the problem."""
    from potpie_context_core.errors import ContextEngineDisabled

    registry[hosts.MANAGED].pots = _Pots(
        [], raises=ContextEngineDisabled("connection refused")
    )

    result = _run("--json", "graph", "catalog", "--pot", "managed:anything")

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    payload = json.loads(result.output)
    # `potpie graph` nests errors inside its own envelope.
    assert payload["error"]["code"] == "unavailable"
    assert "managed host" in payload["error"]["message"]
    assert "connection refused" in payload["error"]["message"]
    # The repair is the host, not the pot list: the ref already named the host,
    # so "target one host explicitly" would be advice already followed.
    assert "potpie host list" in payload["recommended_next_action"]
    assert "pot list" not in payload["recommended_next_action"]


def test_a_sole_configured_host_that_is_down_is_not_reported_as_a_missing_pot(
    monkeypatch, tmp_path
) -> None:
    """The same swallow with nothing to route around: with only ``local``
    configured a bare ``--pot`` ref has one candidate, and reading its
    unreachable host as "holds no match" turns a dead daemon into
    ``pot_not_found`` — a failure whose stated repair cannot succeed."""
    from potpie_context_core.errors import ContextEngineDisabled

    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    monkeypatch.setattr(
        hosts,
        "build_host",
        lambda origin: _Host(
            _Pots([], raises=ContextEngineDisabled("daemon is not running"))
        ),
    )
    try:
        result = _run("--json", "graph", "catalog", "--pot", "shop")
    finally:
        hosts.reset_for_tests()

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "unavailable"
    assert "local host" in payload["error"]["message"]
    assert "daemon is not running" in payload["error"]["message"]
    assert "potpie doctor" in payload["recommended_next_action"]


# --- one injected host is one host, and it says which ----------------------


def test_an_explicit_origin_with_no_host_behind_it_is_refused(
    monkeypatch, tmp_path
) -> None:
    """The reported defect: ``use shared --managed`` with an injected host and
    no managed endpoint exited 0, ran the selection on the injected host, and
    persisted ``active_origin=managed`` — a managed selection no managed host
    ever saw. The lie is the selection, so a corrected label would not fix it:
    nothing may run and nothing may be written."""
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    injected = _Host(_Pots([_Pot("pot_x", "shared")]))
    _common.set_host(injected)
    try:
        result = _run("--json", "use", "shared", "--managed")
    finally:
        _common.set_host(None)
        hosts.reset_for_tests()

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "managed" in payload["message"]
    assert injected.pots.used == []
    assert not (tmp_path / "cli_hosts.json").exists()


def test_a_qualified_pot_ref_with_no_host_behind_it_is_refused(
    monkeypatch, tmp_path
) -> None:
    """The same lie reached through ``--pot`` instead of ``potpie use``.

    ``--pot managed:shared`` with an injected host and no managed endpoint asked
    the injected host for the managed host's pots, took the id it returned, and
    moved the whole command to ``managed`` on the strength of it — every later
    read and write in that invocation aimed at a host nothing was resolved
    against. ``potpie use`` was guarded a round earlier; a guard on one of two
    resolution paths only narrows the door.
    """
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    injected = _Host(_Pots([_Pot("pot_x", "shared")]))
    _common.set_host(injected)
    try:
        result = _run("--json", "graph", "catalog", "--pot", "managed:shared")
    finally:
        _common.set_host(None)
        hosts.reset_for_tests()

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    # `potpie graph` nests errors inside its own envelope.
    assert payload["error"]["code"] == "validation_error"
    assert "managed" in payload["error"]["message"]
    assert "qualifier" in payload["recommended_next_action"]
    # The refusal has to land *before* the stand-in answers: an id read off the
    # injected host is the defect, and it is already read by the time anything
    # downstream could notice the origin has no host.
    assert injected.pots.list_calls == 0
    assert injected.pots.used == []


def test_an_injected_host_reports_the_origin_that_is_current(
    injected_session, tmp_path
) -> None:
    """The origin is read, never assumed. A hardcoded ``local`` here labels a
    managed session's selection ``local`` — the same disagreement between the
    origin reported and the origin resolved against that the flags were fixed
    for, just reached from the other side."""
    result = _run("--json", "use", "shared")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"] == "pot_x"
    assert payload["origin"] == "managed"
    assert injected_session.pots.used == ["shared"]
    assert hosts.persisted_origin() == "managed"


def test_an_origin_matching_the_injected_session_still_selects(
    injected_session,
) -> None:
    """The refusal is about a host that is not there, not about the flag: the
    origin the session is already on has to keep working."""
    result = _run("--json", "use", "shared", "--managed")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["origin"] == "managed"
    assert injected_session.pots.used == ["shared"]


# --- a refusal names a repair that can actually work -----------------------


def _raise_in_contract(exc: Exception):
    """Run a one-command app whose body raises ``exc`` inside ``contract()``.

    Synthetic rather than driven through a real command on purpose: what is
    under test is the error boundary's own mapping, so it must not depend on
    which command happens to attach a repair today.
    """
    boundary_app = typer.Typer()

    @boundary_app.command()
    def boom() -> None:
        with _common.contract():
            raise exc

    _common.set_json(True)
    try:
        return CliRunner().invoke(boundary_app, [])
    finally:
        _common.set_json(False)


def test_a_pot_not_found_keeps_the_repair_its_raiser_named() -> None:
    """``PotNotFound`` is also how "this pot does not hold that" is reported.

    ``source remove`` of an id the pot never held raises it, and the boundary
    answered every one of them with "list pots with 'potpie pot list'" — sending
    the operator to hunt for a pot that was never missing, the same shape as the
    unreachable-host case above where the stated repair cannot succeed.
    """
    from potpie_context_core.errors import PotNotFound

    exc = PotNotFound("No source 'src_9' in pot 'p1'.")
    # Set after construction so this pins the boundary's behaviour rather than
    # the error class's keyword signature.
    exc.recommended_next_action = "list this pot's sources with 'potpie source list'"

    result = _raise_in_contract(exc)

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert payload["message"] == "No source 'src_9' in pot 'p1'."
    assert payload["recommended_next_action"] == (
        "list this pot's sources with 'potpie source list'"
    )
    assert "pot list" not in payload["recommended_next_action"]


def test_a_pot_not_found_about_a_pot_still_points_at_the_pot_list() -> None:
    """The generic repair is the fallback, not a casualty of honouring the
    specific one: a miss that really is a missing pot keeps its own guidance."""
    from potpie_context_core.errors import PotNotFound

    result = _raise_in_contract(PotNotFound("No pot matching 'shop'."))

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert "potpie pot list" in payload["recommended_next_action"]
