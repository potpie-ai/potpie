"""Pot commands that reported success without having done, found, or been told
anything.

Three shapes of the same failure, all of them exit 0 (or the wrong exit) on a
command that did not do what its output implies:

* ``pot list`` catches per origin so a dead remote cannot kill the whole
  listing. That is the feature — a listing that dies when a remote is down is
  useless exactly when you need to see what is local — but it is only honest
  while some *other* host answered. When nothing answered, ``{"pots": []}`` at
  exit 0 is indistinguishable from "you have no pots" for every consumer.
* ``source remove`` of an id the pot never held raises the right message with
  the wrong repair attached, sending the operator to look for a pot that was
  never missing.
* ``_teardown`` splits "the host did not say" from "there was no resource
  store" and then re-merges them, letting a result shape that said nothing be
  quoted as if it had.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli import hosts
from potpie.cli.commands import _common, pots
from potpie_context_core.errors import ContextEngineDisabled

#: Verbatim from the reproduction: what `build_host` raises for a stopped daemon.
_DAEMON_DOWN = "Potpie daemon is not running; run 'potpie daemon start'."
_SERVICE_DOWN = "Managed host refused the connection."


class _Pot:
    def __init__(self, pot_id: str, name: str, active: bool = False) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = active


class _Pots:
    """Enumerates, or refuses to. ``list_calls`` is counted because "which host
    was even asked" is the question every one of these tests turns on."""

    def __init__(self, pots_: list[_Pot] | None = None, *, raises: str | None = None):
        self._pots = list(pots_ or ())
        self._raises = raises
        self.list_calls = 0

    def list_pots(self) -> list[_Pot]:
        self.list_calls += 1
        if self._raises is not None:
            raise ContextEngineDisabled(self._raises)
        return self._pots

    def active_pot(self) -> _Pot | None:
        return next((pot for pot in self._pots if pot.active), None)

    def list_repo_sources(self) -> list[Any]:
        return []


class _Host:
    def __init__(self, pots_service: Any) -> None:
        self.pots = pots_service


@pytest.fixture(autouse=True)
def _isolated_cli(monkeypatch, tmp_path):
    """No real host, no real ``~/.potpie``, JSON on.

    ``home_dir`` is redirected before anything reads the origin pointer: every
    command here calls ``current_origin()``, which reads ``cli_hosts.json``.
    """
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    _common.set_json(True)
    yield
    _common.set_json(False)
    _common.set_verbose(False)
    _common.set_host(None)
    hosts.reset_for_tests()


@pytest.fixture
def both_hosts(monkeypatch) -> dict[str, _Host]:
    """A configured managed host beside the local one, both answering."""
    monkeypatch.setattr(
        hosts, "managed_endpoint", lambda: ("http://svc.example", "tok")
    )
    built = {
        hosts.LOCAL: _Host(_Pots([_Pot("pot_l", "notes", active=True)])),
        hosts.MANAGED: _Host(_Pots([_Pot("pot_m", "api", active=True)])),
    }
    monkeypatch.setattr(hosts, "build_host", lambda origin: built[origin])
    return built


@pytest.fixture
def local_only(monkeypatch) -> dict[str, _Host]:
    """No managed host configured — the default install."""
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    built = {hosts.LOCAL: _Host(_Pots([_Pot("pot_l", "notes", active=True)]))}
    monkeypatch.setattr(hosts, "build_host", lambda origin: built[origin])
    return built


def _list(*args: str):
    return CliRunner().invoke(pots.pot_app, ["list", *args])


# --- N3: a listing that listed nothing ------------------------------------


def test_a_sole_host_that_is_down_fails_instead_of_reporting_an_empty_listing(
    local_only,
) -> None:
    """The reproduction: `potpie --json pot list` against a stopped daemon exited
    0 with ``{"pots": [], "unavailable": {...}}``. Every consumer reads that as
    "you have no pots", because exit 0 said there was nothing to read."""
    local_only[hosts.LOCAL].pots = _Pots(raises=_DAEMON_DOWN)

    result = _list()

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert _DAEMON_DOWN in payload["message"]
    assert payload["detail"]["unavailable"] == {"local": _DAEMON_DOWN}
    assert "potpie doctor" in payload["recommended_next_action"]
    # The success envelope must not have been printed at all: an empty `pots`
    # list beside an error is exactly the ambiguity this closes.
    assert "pots" not in payload


def test_targeting_the_one_host_that_is_down_is_not_rescued_by_the_other(
    both_hosts,
) -> None:
    """``--local`` names a host, and naming a host is targeting. The managed
    host answering is irrelevant — it was never asked, and reporting `pots: []`
    at exit 0 claims the *local* host holds nothing."""
    both_hosts[hosts.LOCAL].pots = _Pots(raises=_DAEMON_DOWN)

    result = _list("--local")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert _DAEMON_DOWN in payload["message"]
    assert payload["detail"]["unavailable"] == {"local": _DAEMON_DOWN}
    assert both_hosts[hosts.MANAGED].pots.list_calls == 0


def test_a_targeted_managed_host_is_repaired_as_a_host_not_as_a_daemon(
    both_hosts,
) -> None:
    """``potpie doctor`` only knows about the local daemon, so it is a repair
    that cannot succeed for an unreachable managed service."""
    both_hosts[hosts.MANAGED].pots = _Pots(raises=_SERVICE_DOWN)

    result = _list("--managed")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert _SERVICE_DOWN in payload["message"]
    assert "potpie host list" in payload["recommended_next_action"]
    assert "potpie doctor" not in payload["recommended_next_action"]
    assert both_hosts[hosts.LOCAL].pots.list_calls == 0


def test_all_with_every_host_down_names_every_host_it_could_not_reach(
    both_hosts,
) -> None:
    both_hosts[hosts.LOCAL].pots = _Pots(raises=_DAEMON_DOWN)
    both_hosts[hosts.MANAGED].pots = _Pots(raises=_SERVICE_DOWN)

    result = _list("--all")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert _DAEMON_DOWN in payload["message"]
    assert _SERVICE_DOWN in payload["message"]
    assert payload["detail"]["unavailable"] == {
        "local": _DAEMON_DOWN,
        "managed": _SERVICE_DOWN,
    }
    # Both hosts are in play, so both repairs are offered.
    assert "potpie doctor" in payload["recommended_next_action"]
    assert "potpie host list" in payload["recommended_next_action"]


def test_a_partial_listing_still_degrades_because_a_host_answered(both_hosts) -> None:
    """The feature, not the defect: a dead remote must not take the local
    listing down with it. Guarded here as well as in the origin-selection suite
    because the loud-failure branch above is one predicate away from eating it."""
    both_hosts[hosts.MANAGED].pots = _Pots(raises=_SERVICE_DOWN)

    result = _list()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["id"] for row in payload["pots"]] == ["pot_l"]
    assert payload["unavailable"] == {"managed": _SERVICE_DOWN}


def test_a_reachable_host_holding_no_pots_still_reports_an_empty_listing(
    local_only,
) -> None:
    """The other half of the ambiguity: "you have no pots" is a real answer and
    has to stay an exit-0 one, or the fix above merely moves the lie."""
    local_only[hosts.LOCAL].pots = _Pots([])

    result = _list()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pots"] == []
    assert "unavailable" not in payload


def test_the_human_listing_reports_the_failure_on_the_structured_surface(
    local_only,
) -> None:
    """The loud branch has to stay on ``fail``. Letting the exception through
    instead would land in the boundary's catch-all as "Unexpected internal
    error" at exit 1 — a worse answer than the exit 0 it replaces."""
    local_only[hosts.LOCAL].pots = _Pots(raises=_DAEMON_DOWN)
    _common.set_json(False)

    result = _list()

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    # Single words on purpose: the error console wraps at 80 columns, so
    # asserting a whole sentence would pin the wrapping rather than the message.
    assert "daemon" in result.output
    assert "doctor" in result.output
    assert "Unexpected internal error" not in result.output
    assert "Traceback" not in result.output


# --- S6: the wrong repair on the right message ----------------------------


class _SourcePots:
    """A pot that holds no sources, and says so when asked to remove one."""

    def __init__(self, removed: bool | None) -> None:
        self._removed = removed
        self.calls: list[tuple[str, str]] = []

    def list_pots(self) -> list[_Pot]:
        return [_Pot("p1", "shop", active=True)]

    def active_pot(self) -> _Pot:
        return _Pot("p1", "shop", active=True)

    def list_repo_sources(self) -> list[Any]:
        return []

    def remove_source(self, *, pot_id: str, source_id: str) -> bool | None:
        self.calls.append((pot_id, source_id))
        return self._removed


def _remove_source(removed: bool | None) -> tuple[Any, _SourcePots]:
    service = _SourcePots(removed)
    _common.set_host(_Host(service))
    result = CliRunner().invoke(pots.source_app, ["remove", "src_9", "--pot", "p1"])
    return result, service


def test_a_source_the_pot_never_held_repairs_by_listing_that_pots_sources() -> None:
    """The miss is about a *source*, so sending the operator to ``pot list`` to
    hunt for a pot that was never missing is the identical misdirection the
    message itself was written to avoid."""
    result, service = _remove_source(False)

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert payload["message"] == "No source 'src_9' in pot 'p1'."
    assert payload["recommended_next_action"] == (
        "list this pot's sources with 'potpie source list --pot p1'"
    )
    assert "potpie pot list" not in payload["recommended_next_action"]
    assert "potpie setup" not in payload["recommended_next_action"]
    assert service.calls == [("p1", "src_9")]


def test_a_host_that_answers_nothing_is_still_not_accused_of_a_miss() -> None:
    """Only an explicit ``False`` is a miss; an older host that returns nothing
    cannot be read as having removed nothing."""
    result, service = _remove_source(None)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["removed"] == "src_9"
    assert service.calls == [("p1", "src_9")]


# --- S4: claiming what the host never said --------------------------------


class _Teardown:
    """The current contract: the pot plus what teardown actually did."""

    def __init__(self, pot: _Pot, resources_purged: bool | None) -> None:
        self.pot = pot
        self.resources_purged = resources_purged


class _PotOnly:
    """A third shape: it names the pot and says nothing about teardown."""

    def __init__(self, pot: _Pot) -> None:
        self.pot = pot


def test_a_result_that_names_the_pot_but_not_the_teardown_reports_nothing() -> None:
    """``getattr(result, "resources_purged", None)`` folded "the host did not
    say" back into "there was no resource store" one line after the helper split
    them, and the reset then printed "(no stored resources)" — a claim about the
    pot that nothing in the result supports."""
    pot = _Pot("p1", "shop")

    assert pots._teardown(_PotOnly(pot)) == (pot, None, False)


def test_the_two_shipped_result_shapes_keep_their_answers() -> None:
    """The distinction is only worth anything if the shapes that *do* report
    still do: ``resources_purged=None`` is the store's own answer."""
    pot = _Pot("p1", "shop")

    assert pots._teardown(_Teardown(pot, True)) == (pot, True, True)
    assert pots._teardown(_Teardown(pot, None)) == (pot, None, True)
    assert pots._teardown(pot) == (pot, None, False)


class _ResetPots:
    def __init__(self, result_for) -> None:
        self._result_for = result_for

    def list_pots(self) -> list[_Pot]:
        return [_Pot("p1", "shop")]

    def active_pot(self) -> _Pot:
        return _Pot("p1", "shop")

    def list_repo_sources(self) -> list[Any]:
        return []

    def reset_pot(self, *, ref: str, confirm: bool = False) -> Any:
        return self._result_for(_Pot("p1", "shop"))


def test_reset_never_prints_no_stored_resources_for_a_host_that_did_not_say() -> None:
    _common.set_host(_Host(_ResetPots(_PotOnly)))

    result = CliRunner().invoke(pots.pot_app, ["reset", "p1", "--confirm"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["teardown_reported"] is False
    assert payload["resources_purged"] is None


def test_the_human_reset_line_for_an_unreporting_shape_claims_no_teardown() -> None:
    _common.set_host(_Host(_ResetPots(_PotOnly)))
    _common.set_json(False)

    result = CliRunner().invoke(pots.pot_app, ["reset", "p1", "--confirm"])

    assert result.exit_code == 0, result.output
    assert "no stored resources" not in result.output
    assert "older contract" in result.output
