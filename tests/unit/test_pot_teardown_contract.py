"""Reset/archive across host contract versions.

``reset``/``archive`` are the two commands whose whole job is destroying data,
so the failure that matters is not "it crashed" but *what it said while
crashing*: the CLI read ``.pot`` off a result shape an older host does not
return, and the error boundary turned the ``AttributeError`` into "Unexpected
internal error" **after** the host had already archived the pot. A destructive
command reporting failure on success invites the one response that makes it
worse — running it again.

The second half of what is pinned here is the message. An older host says only
that the call succeeded; its archive may be a flag and nothing more. Printing
"graph and resources cleared" on that word would be the destructive command
lying about the destruction — as would printing it for a host that reported the
wipe *failed*, which is the same lie told in the more dangerous direction:
nobody re-runs a command that claimed to work.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, pots
from potpie_context_core.errors import PotTeardownFailed


class _Pot:
    def __init__(self, pot_id: str, name: str) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = False


class _Teardown:
    """The current contract: the pot plus what teardown actually did."""

    def __init__(self, pot: _Pot, resources_purged: bool | None) -> None:
        self.pot = pot
        self.resources_purged = resources_purged


class _Pots:
    """A host whose reset/archive return whatever ``result`` is built as."""

    def __init__(self, result_for) -> None:
        self._result_for = result_for
        self.calls: list[str] = []

    def list_pots(self) -> list[_Pot]:
        return [_Pot("p1", "shop")]

    def active_pot(self) -> _Pot:
        return _Pot("p1", "shop")

    def list_repo_sources(self) -> list[Any]:
        return []

    def reset_pot(self, *, ref: str, confirm: bool = False) -> Any:
        self.calls.append(f"reset:{ref}")
        return self._result_for(_Pot("p1", "shop"))

    def archive_pot(self, *, ref: str) -> Any:
        self.calls.append(f"archive:{ref}")
        return self._result_for(_Pot("p1", "shop"))


class _Host:
    def __init__(self, pots_service: _Pots) -> None:
        self.pots = pots_service


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_host(None)
    _common.set_json(False)
    _common.set_verbose(False)


def _run(result_for, *args: str, as_json: bool = True):
    service = _Pots(result_for)
    _common.set_host(_Host(service))
    _common.set_json(as_json)
    result = CliRunner().invoke(pots.pot_app, list(args))
    return result, service


# --- the crash ------------------------------------------------------------

#: What a host predating ``PotTeardownResult`` returns: the pot, bare.
_OLD = lambda pot: pot  # noqa: E731
_NEW_PURGED = lambda pot: _Teardown(pot, True)  # noqa: E731
_NEW_NOTHING_TO_PURGE = lambda pot: _Teardown(pot, None)  # noqa: E731

#: Verbatim from the reproduction against a dead graph endpoint.
_REFUSED = "Error 61 connecting to 127.0.0.1:59999. Connection refused."


def _wipe_failed(_pot: _Pot):
    """A host that says the graph wipe did not happen.

    Locally the service raises this directly; over RPC the daemon's error
    envelope lands as the same ``ContextEngineDisabled`` family on the client,
    so one branch of the CLI's error contract covers both hosts.
    """
    raise PotTeardownFailed(
        f"Graph reset failed for pot 'p1': {_REFUSED}. "
        "Nothing was purged; the pot is unchanged.",
        recommended_next_action="check backend readiness with 'potpie doctor'",
    )


@pytest.mark.parametrize("command", ["archive", "reset"])
def test_an_older_hosts_result_shape_does_not_crash(command: str) -> None:
    """This is the regression: the pot *was* archived server-side and the CLI
    still exited non-zero with 'Unexpected internal error'."""
    result, service = _run(_OLD, command, "p1", "--confirm")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"] == "p1"
    assert payload["teardown_reported"] is False
    assert service.calls == [f"{command}:p1"]


def test_an_older_host_is_not_credited_with_clearing_the_graph() -> None:
    """The managed service's archive leaves every claim in place. Saying
    'graph and resources cleared' there is the destructive command lying about
    the destruction."""
    result, _ = _run(_OLD, "archive", "p1", "--confirm", as_json=False)

    assert result.exit_code == 0, result.output
    assert "cleared" not in result.output
    assert "older contract" in result.output
    assert "graph status" in result.output


# --- the current contract still reports precisely ---------------------------


def test_a_purged_resource_store_is_still_reported() -> None:
    result, _ = _run(_NEW_PURGED, "archive", "p1", "--confirm")

    payload = json.loads(result.output)
    assert (payload["resources_purged"], payload["teardown_reported"]) == (True, True)


def test_no_resource_store_is_distinct_from_an_unreporting_host() -> None:
    """``resources_purged=None`` means the pot had no resource tree — a fact
    about the pot. Not knowing is a fact about the host. Collapsing them would
    print 'no stored resources' for a host that never said so."""
    reported, _ = _run(_NEW_NOTHING_TO_PURGE, "archive", "p1", "--confirm")
    unreported, _ = _run(_OLD, "archive", "p1", "--confirm")

    a, b = json.loads(reported.output), json.loads(unreported.output)
    assert a["resources_purged"] is None and b["resources_purged"] is None
    assert (a["teardown_reported"], b["teardown_reported"]) == (True, False)


@pytest.mark.parametrize("command", ["archive", "reset"])
def test_a_host_that_reports_a_failed_wipe_never_prints_a_success_envelope(
    command: str,
) -> None:
    """The mirror of the crash above: reporting success on failure. It shipped —
    a reset against a dead backend exited 0 with ``{"reset": true,
    "resources_purged": true}`` — and it is the worse half, because nobody
    re-runs a command that says it worked."""
    result, service = _run(_wipe_failed, command, "p1", "--confirm")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert _REFUSED in payload["message"]
    assert payload["recommended_next_action"] == (
        "check backend readiness with 'potpie doctor'"
    )
    # ``resources_purged`` belongs to both success envelopes and to neither
    # failure: a teardown that did not run has nothing to report about it.
    assert "resources_purged" not in payload
    assert service.calls == [f"{command}:p1"]


@pytest.mark.parametrize("command", ["archive", "reset"])
def test_confirmation_is_still_required(command: str) -> None:
    """Tolerating the older shape must not have loosened the guard."""
    result, service = _run(_NEW_PURGED, command, "p1")

    assert result.exit_code == _common.EXIT_VALIDATION
    assert json.loads(result.output)["code"] == "confirmation_required"
    assert service.calls == []


# --- why this was undiagnosable --------------------------------------------


def test_verbose_surfaces_the_traceback_behind_an_unexpected_error(capsys) -> None:
    """`--verbose` is documented as 'Verbose tracebacks on errors' and dropped
    them, so the only way to see what an 'Unexpected internal error' was is to
    reproduce the call by hand outside the CLI — which is exactly what finding
    this bug took."""
    import typer

    _common.set_json(False)
    _common.set_verbose(True)

    with pytest.raises(typer.Exit):
        with _common.contract():
            raise AttributeError("'PotInfo' object has no attribute 'pot'")

    # Structured errors go to the rich *stderr* console.
    output = capsys.readouterr().err
    assert "AttributeError" in output
    assert "has no attribute 'pot'" in output


def test_without_verbose_the_traceback_stays_out_of_the_output(capsys) -> None:
    import typer

    _common.set_json(False)
    _common.set_verbose(False)

    with pytest.raises(typer.Exit):
        with _common.contract():
            raise AttributeError("boom")

    # Structured errors go to the rich *stderr* console.
    output = capsys.readouterr().err
    assert "Unexpected internal error" in output
    assert "AttributeError" not in output
