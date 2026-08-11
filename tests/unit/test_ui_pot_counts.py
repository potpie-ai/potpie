"""The pot selector must never invent a count.

``0 claims`` next to a pot name is a routing decision: it says "not this one".
The listing therefore has exactly two honest shapes — a number the backend
gave, or no ``counts`` key at all plus ``counts_complete: false`` and a stated
reason, which is what the SPA already renders for a pot past the remote count
budget.

What is pinned here is the third shape that kept coming back: an empty mapping.
It is truthy in JS, so it renders as a real zero, and it arrives from a *success*
— a status object with no counts, values that are not numbers, or the empty dict
``DataPlaneStatus`` substitutes when the analytics call fails — so no exception
handler upstream ever sees it.

``/api/status`` is the same fact one route over, and it gets a different shape
for one reason: the header it feeds has nowhere to put "unknown". It renders
``counts.entities ?? 0``, so omission reads as zero there, and the response
would be asserting ``backend_ready: true`` beside it. A pot whose numbers could
not be read therefore fails that route outright.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from potpie.cli import hosts
from potpie.daemon.http.ui.auth import UiAuth
from potpie.daemon.http.ui.router import build_ui_api_router

_TOKEN = "ui-counts-test-token"  # noqa: S105 - non-secret test fixture


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch, tmp_path):
    """Keep the developer's own managed login out of these tests.

    The router reads the host registry from the shared home, so without this a
    machine with a managed host configured would have every listing here reach
    across the network for that host's pots.
    """
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    yield
    hosts.reset_for_tests()


class _Pot:
    pot_id = "p1"
    name = "shop"
    active = True


class _Pots:
    def list_pots(self) -> list[_Pot]:
        return [_Pot()]

    def active_pot(self) -> _Pot:
        return _Pot()

    def list_sources(self, *, pot_id: str) -> list[Any]:
        return []


class _Graph:
    """A host whose ``data_plane_status`` succeeds and answers ``status``."""

    def __init__(self, status: Any) -> None:
        self._status = status

    def data_plane_status(self, pot_id: str) -> Any:
        return self._status


class _Host:
    def __init__(self, status: Any) -> None:
        self.pots = _Pots()
        self.graph = _Graph(status)


class _NoCounts:
    """A status object from a host that does not report counts at all."""

    backend_profile = "in_memory"
    backend_ready = True


def _client(status: Any) -> TestClient:
    app = FastAPI()
    app.include_router(build_ui_api_router(_Host(status)))
    app.state.ui_auth = UiAuth(token=_TOKEN)
    return TestClient(app, headers={"Authorization": f"Bearer {_TOKEN}"})


def _row(status: Any) -> dict[str, Any]:
    return _client(status).get("/api/pots").json()


class _Counts:
    """A status object carrying exactly what the backend said the counts were.

    Answers ``backend_profile``/``backend_ready`` too, because ``/api/status``
    reports those alongside the numbers — and the pairing is the defect: a
    confident ``backend_ready: true`` is what makes a fabricated zero credible.
    """

    backend_profile = "falkordb"
    backend_ready = True

    def __init__(self, counts: Any) -> None:
        self.counts = counts


#: Every way a status object arrives without countable numbers. None of them is
#: an exception, which is why nothing upstream ever caught them.
_UNUSABLE = [
    _NoCounts(),
    # The empty dict `DataPlaneStatus` substitutes when analytics raises —
    # never what an empty pot looks like, which is `claims: 0`.
    _Counts({}),
    _Counts({"claims": "many", "entities": None}),
    _Counts(None),
]
_UNUSABLE_IDS = ["attribute-missing", "analytics-swallowed", "non-numeric", "null"]


@pytest.mark.parametrize("status", _UNUSABLE, ids=_UNUSABLE_IDS)
def test_a_pot_whose_counts_are_unusable_is_listed_without_them(status: Any) -> None:
    body = _row(status)
    (pot,) = body["pots"]

    assert pot["name"] == "shop"
    assert "counts" not in pot
    assert body["counts_complete"] is False
    assert "p1" in body["unavailable"]["local counts"]


def test_real_counts_are_still_reported_as_complete() -> None:
    body = _row(_Counts({"claims": 82, "entities": 46}))
    (pot,) = body["pots"]

    assert pot["counts"] == {"claims": 82, "entities": 46}
    assert body["counts_complete"] is True
    assert body["unavailable"] == {}


def test_an_empty_pot_still_reports_its_zero() -> None:
    """The distinction the whole rule rests on: a backend that says "zero" is
    answering, and suppressing that number would hide an empty pot instead."""
    body = _row(_Counts({"claims": 0, "entities": 0}))
    (pot,) = body["pots"]

    assert pot["counts"] == {"claims": 0, "entities": 0}
    assert body["counts_complete"] is True


def test_one_unreadable_number_does_not_discard_the_readable_ones() -> None:
    body = _row(_Counts({"claims": 3, "entities": "?"}))
    (pot,) = body["pots"]

    assert pot["counts"] == {"claims": 3}
    assert body["counts_complete"] is True


# --- /api/status: the header has nowhere to put "unknown" -------------------
#
# The selector can drop a count it does not have, because the SPA renders the
# row without one. The header cannot: it reads `counts.entities ?? 0` and
# `counts.claims ?? 0` off `status?.counts || {}`, so a missing key, a null and
# an empty mapping all come out as "0 entities / 0 claims" — beside the
# `backend_profile` chip and a `backend_ready: true` the same response asserted.
# Omitting the key here would have moved the lie, not removed it, so this route
# refuses instead and the SPA shows the reason in its error banner.


@pytest.mark.parametrize("status", _UNUSABLE, ids=_UNUSABLE_IDS)
def test_status_refuses_the_pot_whose_counts_it_could_not_read(status: Any) -> None:
    response = _client(status).get("/api/status?pot=p1")

    assert response.status_code == 503
    body = response.json()
    # Nothing partial: half a status is what let a zero ride alongside a
    # readiness flag that had nothing to do with the numbers.
    assert set(body) == {"detail"}
    assert "p1" in body["detail"]


def test_status_reports_the_numbers_the_backend_did_give() -> None:
    response = _client(_Counts({"claims": 82, "entities": 46})).get(
        "/api/status?pot=p1"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["counts"] == {"claims": 82, "entities": 46}
    assert body["backend_ready"] is True
    assert body["backend_profile"] == "falkordb"


def test_status_reports_a_genuinely_empty_pot_as_zero() -> None:
    """The distinction the rule rests on, on the route that cannot omit: a
    backend answering "zero" is answering. Refusing here would take the header
    out for every pot that is simply new, which is most of them on day one."""
    response = _client(_Counts({"claims": 0, "entities": 0})).get("/api/status?pot=p1")

    assert response.status_code == 200
    assert response.json()["counts"] == {"claims": 0, "entities": 0}


def test_status_keeps_the_numbers_it_can_read() -> None:
    """One unreadable value is not a broken backend; the readable ones still
    beat no header at all."""
    response = _client(_Counts({"claims": 3, "entities": "?"})).get(
        "/api/status?pot=p1"
    )

    assert response.status_code == 200
    assert response.json()["counts"] == {"claims": 3}
