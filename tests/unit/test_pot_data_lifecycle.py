"""Teardown must not outrun the graph wipe it claims to have done.

``pot reset`` / ``pot archive`` destroy two stores, and the order is the whole
contract: the resource tree may go once the claims citing its chunks are gone.
Mutation adapters do not raise when their store is unreachable — FalkorDB's
writer logs and *returns* ``{"ok": False, "error": "...Connection refused."}``
— so a service that discards the return purges the chunk files anyway. Live
run of the shipped build: ``pot reset <pot> --confirm`` against a dead backend
printed ``{"reset": true, "resources_purged": true}`` at exit 0 while every
claim survived and its evidence did not.

These tests run the real ``LocalPotManagementService`` over a real
``LocalPotStore`` and fake only the layer that actually fails — the mutation
adapter — because the previous tests stubbed the service itself and so agreed
with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, pots
from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.application.services.pot_management import (
    LocalPotManagementService,
)
from potpie_context_core.errors import PotTeardownFailed

POT = "shop"

#: Verbatim from the reproduction: what redis/FalkorDB hands back when the
#: configured endpoint is not listening.
REFUSED = "Error 61 connecting to 127.0.0.1:59999. Connection refused."


class _Mutation:
    """Answers ``reset_pot`` with a fixed value, or raises it if it is an error."""

    def __init__(self, answer: Any) -> None:
        self._answer = answer
        self.calls: list[str] = []

    def reset_pot(self, pot_id: str) -> Any:
        self.calls.append(pot_id)
        if isinstance(self._answer, BaseException):
            raise self._answer
        return self._answer


@dataclass(slots=True)
class _Backend:
    mutation: _Mutation


class _Resources:
    """Records every purge attempt; ``purged`` is what the store would answer."""

    def __init__(self, purged: bool = True) -> None:
        self.purged = purged
        self.calls: list[str] = []

    def purge_pot(self, pot_id: str) -> bool:
        self.calls.append(pot_id)
        return self.purged


class _Host:
    def __init__(self, pots_service: LocalPotManagementService) -> None:
        self.pots = pots_service


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_host(None)
    _common.set_json(False)


def _service(
    tmp_path,
    *,
    answer: Any,
    resources: _Resources | None,
) -> LocalPotManagementService:
    service = LocalPotManagementService(
        store=LocalPotStore(home=tmp_path / "home"),
        backend=_Backend(_Mutation(answer)),
        resources=resources,
    )
    service.create_pot(name=POT, use=True)
    return service


def _run(service: LocalPotManagementService, *args: str):
    _common.set_host(_Host(service))
    _common.set_json(True)
    return CliRunner().invoke(pots.pot_app, list(args))


# --- a graph wipe that did not happen ---------------------------------------


def test_reset_reports_the_adapter_error_and_purges_nothing(tmp_path) -> None:
    resources = _Resources()
    service = _service(
        tmp_path, answer={"ok": False, "error": REFUSED}, resources=resources
    )

    result = _run(service, "reset", POT, "--confirm")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert REFUSED in payload["message"]
    # The success envelope must not be reachable: it claimed both halves.
    assert "reset" not in payload
    assert resources.calls == []


def test_archive_does_not_flip_the_flag_when_the_graph_wipe_fails(tmp_path) -> None:
    """An archived pot with live claims is worse than an un-archived one: the
    flag hides it from ``pot list`` while its data stays in the graph, so the
    user can no longer see it *or* reach the command that would clear it."""
    resources = _Resources()
    service = _service(
        tmp_path, answer={"ok": False, "error": REFUSED}, resources=resources
    )

    result = _run(service, "archive", POT, "--confirm")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    assert json.loads(result.output)["code"] == "unavailable"
    assert resources.calls == []
    assert [p.archived for p in service.list_pots()] == [False]
    assert service.active_pot() is not None


def test_a_failed_wipe_still_fails_with_no_resource_store_wired(tmp_path) -> None:
    """Nothing to purge does not make the reset true — the claims are still there."""
    service = _service(tmp_path, answer={"ok": False, "error": REFUSED}, resources=None)

    with pytest.raises(PotTeardownFailed) as exc:
        service.reset_pot(ref=POT, confirm=True)

    assert REFUSED in str(exc.value)
    assert exc.value.recommended_next_action


def test_an_adapter_that_raises_propagates_without_purging(tmp_path) -> None:
    """Re-swallowing a raise here would rebuild the bug in the other direction."""
    resources = _Resources()
    service = _service(
        tmp_path, answer=RuntimeError("driver closed"), resources=resources
    )

    with pytest.raises(RuntimeError):
        service.reset_pot(ref=POT, confirm=True)

    assert resources.calls == []


# --- a graph wipe that half-happened ----------------------------------------

#: What FalkorDB's ``_reset_pot_sync`` and Neo4j's ``reset_pot`` return when the
#: batched sweep runs out of road: the delete already took 60 of the 100 nodes.
PARTIAL = {
    "ok": False,
    "error": "group_id_reset_incomplete",
    "group_id_nodes_before": 100,
    "group_id_nodes_remaining": 40,
}


@pytest.mark.parametrize("command", ["reset", "archive"])
def test_a_partial_wipe_says_what_survived_instead_of_claiming_unchanged(
    tmp_path, command: str
) -> None:
    """Both real adapters delete in batches and only *then* report failure, so
    ``ok: False`` routinely arrives with most of the pot already gone. Telling
    that user "the pot is unchanged" is the same lie the raise exists to
    remove, one size smaller: they re-run believing nothing happened and never
    learn that 40 nodes — and the claims on them — survived."""
    resources = _Resources()
    service = _service(tmp_path, answer=PARTIAL, resources=resources)

    result = _run(service, command, POT, "--confirm")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    message = json.loads(result.output)["message"]
    assert "unchanged" not in message, message
    assert "60 of 100" in message
    assert "40 remain" in message
    # The resource tree is the half that must not move on a partial wipe: the
    # surviving claims still cite its chunk files.
    assert resources.calls == []
    assert [p.archived for p in service.list_pots()] == [False]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # No progress at all — still not a licence to stay silent about the 7.
        (
            {
                "ok": False,
                "error": "group_id_reset_incomplete",
                "group_id_nodes_before": 7,
                "group_id_nodes_remaining": 7,
            },
            "7 graph nodes remain",
        ),
        # An adapter that counts only what is left.
        (
            {
                "ok": False,
                "error": "group_id_reset_incomplete",
                "group_id_nodes_remaining": 12,
            },
            "12 graph nodes remain",
        ),
    ],
)
def test_every_reported_survivor_count_reaches_the_caller(
    tmp_path, answer: Any, expected: str
) -> None:
    service = _service(tmp_path, answer=answer, resources=None)

    with pytest.raises(PotTeardownFailed) as exc:
        service.reset_pot(ref=POT, confirm=True)

    assert expected in str(exc.value)


def test_an_adapter_that_reports_no_counts_still_says_the_pot_is_unchanged(
    tmp_path,
) -> None:
    """The opposite over-correction. A refused connection deleted nothing, and
    hedging there would tell every user with a dead backend that some unknown
    part of their pot might already be gone."""
    service = _service(tmp_path, answer={"ok": False, "error": REFUSED}, resources=None)

    with pytest.raises(PotTeardownFailed) as exc:
        service.reset_pot(ref=POT, confirm=True)

    assert "Nothing was purged; the pot is unchanged." in str(exc.value)


# --- adapters that succeed, including the ones that say nothing --------------


@pytest.mark.parametrize(
    ("answer", "label"),
    [
        (None, "reports nothing"),
        ({"ok": True}, "reports ok"),
        ({"removed_claims": 3}, "reports a count"),  # embedded/lite default
    ],
)
@pytest.mark.parametrize("command", ["reset", "archive"])
def test_an_adapter_without_an_explicit_failure_still_tears_down(
    tmp_path, answer: Any, label: str, command: str
) -> None:
    """Only an explicit ``ok: False`` is a failure. Reading silence as one would
    break the default backend on every reset — the opposite mistake, equally
    destructive to trust in the command."""
    resources = _Resources()
    service = _service(tmp_path, answer=answer, resources=resources)

    result = _run(service, command, POT, "--confirm")

    assert result.exit_code == 0, f"{label}: {result.output}"
    payload = json.loads(result.output)
    assert payload["resources_purged"] is True
    assert payload["teardown_reported"] is True
    assert resources.calls == [payload["id"]]
