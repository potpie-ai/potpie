"""What the daemon puts on the wire, and what the client makes of it.

The shipped default host *is* the daemon, so every domain error the CLI renders
has crossed this boundary first. Two failures live here: an encoding that lets
an arbitrary class ride in, and an error envelope that reports an expected
answer as a daemon fault — logging a traceback, raising a Sentry event tagged
``is_expected=false``, and replacing the domain's own repair hint with the
generic pointer at ``potpie doctor``.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import typer

from potpie.cli.commands import _common
from potpie.daemon import main as daemon_main
from potpie.daemon.client import DaemonRpcClient, _raise_remote_error
from potpie.daemon.telemetry import sentry_runtime
from potpie_context_core.errors import (
    ContextEngineDisabled,
    GraphSubstrateUnavailable,
    PotNotFound,
    PotTeardownFailed,
)
from potpie_context_core.lifecycle import SetupPlan
from potpie.daemon.rpc import TYPE_KEY, decode, encode


def test_daemon_rpc_roundtrips_domain_dataclasses() -> None:
    plan = SetupPlan(
        backend="embedded",
        repo="potpie",
        pot="default",
        agent="claude",
        assume_yes=True,
    )

    assert decode(encode(plan)) == plan


def test_daemon_rpc_rejects_non_domain_class_references() -> None:
    with pytest.raises(TypeError, match="RPC class module not allowed"):
        decode(
            {
                TYPE_KEY: "dataclass",
                "class": "os:stat_result",
                "value": {},
            }
        )


# --- an unavailable backend is an answer, not a daemon fault ----------------

#: Verbatim from a reset against a dead graph endpoint.
_WIPE_FAILED = (
    "Graph reset failed for pot 'p1': Error 61 connecting to 127.0.0.1:59999. "
    "Connection refused. Nothing was purged; the pot is unchanged."
)
_REPAIR = "check backend readiness with 'potpie doctor'"


@pytest.mark.parametrize(
    "error",
    [
        PotTeardownFailed(_WIPE_FAILED, recommended_next_action=_REPAIR),
        GraphSubstrateUnavailable(
            "graph substrate shut down uncleanly", recommended_next_action=_REPAIR
        ),
        ContextEngineDisabled("the graph backend is unavailable"),
    ],
    ids=["teardown", "substrate", "base"],
)
def test_an_unavailable_backend_crosses_the_wire_as_an_expected_answer(
    error: ContextEngineDisabled,
) -> None:
    """``daemon_error`` is reserved for the daemon being wrong. Sending this
    family under that code cost the CLI the one thing it needed — the repair the
    error was raised to carry."""
    error_payload = daemon_main._error_payload(error)["error"]

    assert error_payload["code"] == "unavailable"
    assert error_payload["message"] == str(error)
    assert error_payload["recommended_next_action"] == getattr(
        error, "recommended_next_action", None
    )


def test_an_expected_unavailability_raises_no_alert_and_logs_no_traceback(
    monkeypatch, caplog
) -> None:
    """A backend that is down is not an incident: a traceback in the daemon log
    and a Sentry event per failed call is noise that trains everyone to ignore
    the channel that carries the real ones."""
    captured: list[dict[str, str]] = []
    monkeypatch.setattr(
        sentry_runtime,
        "capture_unexpected_daemon_error",
        lambda exc, **kwargs: captured.append(kwargs),
    )

    with caplog.at_level(logging.DEBUG, logger=daemon_main.__name__):
        daemon_main._error_payload(
            PotTeardownFailed(_WIPE_FAILED, recommended_next_action=_REPAIR)
        )

    assert captured == []
    assert [record for record in caplog.records if record.exc_info] == []


def test_a_genuinely_unexpected_error_still_alerts(monkeypatch, caplog) -> None:
    """The control on the branch above: quieting the expected family must not
    quiet the faults the channel exists for."""
    captured: list[dict[str, str]] = []
    monkeypatch.setattr(
        sentry_runtime,
        "capture_unexpected_daemon_error",
        lambda exc, **kwargs: captured.append(kwargs),
    )

    with caplog.at_level(logging.DEBUG, logger=daemon_main.__name__):
        payload = daemon_main._error_payload(RuntimeError("dict changed size"))

    assert payload["error"]["code"] == "daemon_error"
    assert captured == [{"error_code": "daemon_error", "error_kind": "unexpected"}]
    assert [record for record in caplog.records if record.exc_info]


def test_the_client_rebuilds_the_unavailability_with_its_repair() -> None:
    """Reconstructed as the same family the in-process host raises, so one
    branch of the CLI's error contract covers both hosts."""
    with pytest.raises(ContextEngineDisabled) as raised:
        _raise_remote_error(
            {
                "error": {
                    "code": "unavailable",
                    "message": _WIPE_FAILED,
                    "recommended_next_action": _REPAIR,
                }
            }
        )

    assert str(raised.value) == _WIPE_FAILED
    assert getattr(raised.value, "recommended_next_action", None) == _REPAIR


def test_a_daemon_fault_still_arrives_without_a_repair_to_offer() -> None:
    with pytest.raises(ContextEngineDisabled) as raised:
        _raise_remote_error({"error": {"code": "daemon_error", "message": "boom"}})

    assert getattr(raised.value, "recommended_next_action", None) is None


# --- a refused key is not an absent host ------------------------------------
#
# Both used to arrive as the same sentence. `_raise_remote_error` saw only the
# body, and a 401 carries FastAPI's `{"detail": ...}` rather than this daemon's
# error envelope, so it fell through to `ContextEngineDisabled("Potpie daemon
# request failed.")` — which `potpie host set` then wrapped as "The managed host
# at <url> did not answer". The host answered in milliseconds; it refused. One
# reading sends you to restart a service and check a network, the other to fix a
# token, and the status code was the only thing that ever knew which.
#
# No socket is opened here: `httpx.post` is the seam, and a `httpx.Response`
# built by hand is exactly what the client would have been handed.

_MANAGED_URL = "https://graph.example.com"
_MANAGED_LABEL = "Managed (graph.example.com)"


class _Endpoint:
    """An address and a key, which is all ``DaemonRpcClient`` asks its daemon for."""

    def __init__(self, base_url: str, token: str) -> None:
        self._discovery = {"base_url": base_url, "token": token}

    def discovery(self) -> dict[str, str]:
        return dict(self._discovery)


def _answering(status_code: int, *, body: dict | None = None, text: str | None = None):
    def _post(url: str, **_: object) -> httpx.Response:
        request = httpx.Request("POST", url)
        if text is not None:
            return httpx.Response(status_code, text=text, request=request)
        return httpx.Response(status_code, json=body or {}, request=request)

    return _post


def _managed_client() -> DaemonRpcClient:
    return DaemonRpcClient(
        daemon=_Endpoint(_MANAGED_URL, "stale-key"), label=_MANAGED_LABEL
    )


@pytest.mark.parametrize("status_code", [401, 403])
def test_a_host_that_refuses_the_key_says_so(monkeypatch, status_code: int) -> None:
    monkeypatch.setattr(
        httpx, "post", _answering(status_code, body={"detail": "invalid daemon token"})
    )

    with pytest.raises(ContextEngineDisabled) as raised:
        _managed_client().call("pots", "list_pots")

    message = str(raised.value)
    assert "refused the credential" in message
    assert str(status_code) in message
    # The endpoint by name: the client drives two hosts, and the one that
    # refused is the only fact that tells them apart.
    assert f"{_MANAGED_URL}/rpc" in message
    assert "invalid daemon token" in message
    assert "did not answer" not in message
    # Carried on the exception too, so a caller wrapping this in its own wording
    # can still tell a refusal from a silence without parsing English.
    assert getattr(raised.value, "status_code", None) == status_code


def test_a_host_that_never_answered_remains_a_different_failure(monkeypatch) -> None:
    """The control the whole distinction depends on: this one really is
    unreachable, and it must not start reading like a credential problem."""

    def _refuse_connection(url: str, **_: object):
        raise httpx.ConnectError(
            "Connection refused", request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", _refuse_connection)

    with pytest.raises(ContextEngineDisabled) as raised:
        _managed_client().call("pots", "list_pots")

    message = str(raised.value)
    assert f"{_MANAGED_LABEL} is unavailable" in message
    assert "credential" not in message
    # Nothing answered, so there is no status to carry.
    assert getattr(raised.value, "status_code", None) is None


def test_a_proxy_error_page_does_not_bury_the_refusal(monkeypatch) -> None:
    """A managed host behind a gateway answers 401 with HTML. Parsing the body
    first turned that into "returned a non-JSON response", which describes the
    shape of the answer and drops what it said."""
    monkeypatch.setattr(
        httpx,
        "post",
        _answering(
            401, text="<html><head><title>401 Unauthorized</title></head></html>"
        ),
    )

    with pytest.raises(ContextEngineDisabled) as raised:
        _managed_client().call("pots", "list_pots")

    message = str(raised.value)
    assert "refused the credential" in message
    assert "non-JSON" not in message
    # The page itself is not the message; the status already said everything it
    # could have.
    assert "<html" not in message


def test_the_attr_surface_reports_a_refusal_the_same_way(monkeypatch) -> None:
    """``attr`` is the other door into the same endpoint and had its own copy of
    the error handling."""
    monkeypatch.setattr(httpx, "post", _answering(401, body={"detail": "nope"}))

    with pytest.raises(ContextEngineDisabled) as raised:
        _managed_client().attr("backend", "profile")

    assert "refused the credential" in str(raised.value)
    assert f"{_MANAGED_URL}/attr" in str(raised.value)


def test_the_status_code_reaches_the_error_builder() -> None:
    """The seam itself: whatever route into ``_raise_remote_error`` a caller
    takes, handing it the status is what makes the refusal expressible."""
    with pytest.raises(ContextEngineDisabled) as raised:
        _raise_remote_error(
            {"detail": "invalid daemon token"},
            status_code=401,
            endpoint=f"{_MANAGED_URL}/rpc",
            label=_MANAGED_LABEL,
        )

    assert "refused the credential" in str(raised.value)
    assert getattr(raised.value, "status_code", None) == 401
    # A refusal has a repair, and it is not "check whether the service is up".
    assert "token" in (getattr(raised.value, "recommended_next_action", "") or "")


def test_a_body_shaped_error_still_wins_over_the_status(monkeypatch) -> None:
    """A 4xx that carries this daemon's own envelope keeps its own classification
    — the status only decides the cases the envelope never covered."""
    monkeypatch.setattr(
        httpx,
        "post",
        _answering(
            400,
            body={
                "ok": False,
                "error": {"code": "pot_not_found", "message": "no pot 'p9'"},
            },
        ),
    )

    with pytest.raises(PotNotFound) as raised:
        _managed_client().call("pots", "use_pot", ref="p9")

    assert "p9" in str(raised.value)


def test_the_specific_repair_survives_the_round_trip_to_the_cli(capsys) -> None:
    """End to end over the shipped path: the daemon's envelope, the client's
    exception, and the CLI's error contract. What the user must see is what the
    teardown knew — not the generic doctor line the fall-through produced."""
    payload = daemon_main._error_payload(
        PotTeardownFailed(_WIPE_FAILED, recommended_next_action=_REPAIR)
    )
    _common.set_json(True)
    try:
        with pytest.raises(typer.Exit) as exited:
            with _common.contract():
                _raise_remote_error(payload)
    finally:
        _common.set_json(False)

    envelope = json.loads(capsys.readouterr().out)
    assert exited.value.exit_code == _common.EXIT_UNAVAILABLE
    assert envelope["code"] == "unavailable"
    assert envelope["message"] == _WIPE_FAILED
    assert envelope["recommended_next_action"] == _REPAIR
