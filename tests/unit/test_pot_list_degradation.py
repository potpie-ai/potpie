"""``pot list`` degrades per host, but never reports an empty listing it did not get.

``pot list`` catches per origin so a dead managed host cannot kill the whole
listing. That is only honest while some *other* host answered: when nothing
answered, ``{"pots": []}`` at exit 0 would be indistinguishable from "you have
no pots" for every consumer.
"""

from __future__ import annotations

import json

import pytest
from potpie_context_engine.core.errors import ContextEngineDisabled
from typer.testing import CliRunner

from potpie.cli.commands import _common, pots
from potpie.config import host_registry
from potpie.pots.contracts import PotInfo
from tests.support.pot_directories import two_hosts

_DAEMON_DOWN = "Potpie daemon is not running; run 'potpie daemon start'."
_SERVICE_DOWN = "Managed host refused the connection."


@pytest.fixture(autouse=True)
def _isolated_cli():
    host_registry.reset_for_tests()
    _common.set_json(True)
    yield
    _common.set_json(False)
    _common.set_pot_directory_factory(None)
    host_registry.reset_for_tests()


@pytest.fixture
def both_hosts(monkeypatch):
    return two_hosts(
        monkeypatch,
        local=[PotInfo(pot_id="pot_l", name="notes", active=True)],
        managed=[PotInfo(pot_id="pot_m", name="api", active=True)],
    )


@pytest.fixture
def local_only(monkeypatch):
    return two_hosts(
        monkeypatch,
        local=[PotInfo(pot_id="pot_l", name="notes", active=True)],
        managed_configured=False,
    )


def _list(*args: str):
    return CliRunner().invoke(pots.pot_app, ["list", *args])


def test_a_sole_host_that_is_down_fails_instead_of_reporting_an_empty_listing(
    local_only,
) -> None:
    local_only["local"].raises = ContextEngineDisabled(_DAEMON_DOWN)

    result = _list()

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert _DAEMON_DOWN in payload["message"]
    assert payload["detail"]["unavailable"] == {"local": _DAEMON_DOWN}
    assert "potpie doctor" in payload["recommended_next_action"]
    assert "pots" not in payload


def test_targeting_the_one_host_that_is_down_is_not_rescued_by_the_other(
    both_hosts,
) -> None:
    both_hosts["local"].raises = ContextEngineDisabled(_DAEMON_DOWN)

    result = _list("--local")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["detail"]["unavailable"] == {"local": _DAEMON_DOWN}
    assert both_hosts["managed"].list_calls == 0


def test_a_targeted_managed_host_is_repaired_as_a_host_not_as_a_daemon(
    both_hosts,
) -> None:
    both_hosts["managed"].raises = ContextEngineDisabled(_SERVICE_DOWN)

    result = _list("--managed")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert _SERVICE_DOWN in payload["message"]
    assert "potpie host list" in payload["recommended_next_action"]
    assert "potpie doctor" not in payload["recommended_next_action"]
    assert both_hosts["local"].list_calls == 0


def test_all_with_every_host_down_names_every_host_it_could_not_reach(
    both_hosts,
) -> None:
    both_hosts["local"].raises = ContextEngineDisabled(_DAEMON_DOWN)
    both_hosts["managed"].raises = ContextEngineDisabled(_SERVICE_DOWN)

    result = _list("--all")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["detail"]["unavailable"] == {
        "local": _DAEMON_DOWN,
        "managed": _SERVICE_DOWN,
    }
    assert "potpie doctor" in payload["recommended_next_action"]
    assert "potpie host list" in payload["recommended_next_action"]


def test_a_partial_listing_still_degrades_because_a_host_answered(both_hosts) -> None:
    both_hosts["managed"].raises = ContextEngineDisabled(_SERVICE_DOWN)

    result = _list()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["id"] for row in payload["pots"]] == ["pot_l"]
    assert payload["unavailable"] == {"managed": _SERVICE_DOWN}


def test_a_reachable_host_holding_no_pots_still_reports_an_empty_listing(
    local_only,
) -> None:
    local_only["local"].pots = []

    result = _list()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pots"] == []
    assert "unavailable" not in payload


def test_the_human_listing_reports_the_failure_on_the_structured_surface(
    local_only,
) -> None:
    local_only["local"].raises = ContextEngineDisabled(_DAEMON_DOWN)
    _common.set_json(False)

    result = _list()

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    assert "daemon" in result.output
    assert "doctor" in result.output
    assert "Unexpected internal error" not in result.output


def test_naming_both_origins_lists_both_of_them(both_hosts) -> None:
    result = _list("--local", "--managed")

    assert result.exit_code == 0, result.output
    assert [row["id"] for row in json.loads(result.output)["pots"]] == [
        "pot_l",
        "pot_m",
    ]


def test_all_with_one_origin_named_twice_is_still_all(both_hosts) -> None:
    result = _list("--all", "--local")

    assert result.exit_code == 0, result.output
    assert [row["id"] for row in json.loads(result.output)["pots"]] == [
        "pot_l",
        "pot_m",
    ]


def test_naming_managed_on_an_install_without_it_states_the_missing_host(
    local_only,
) -> None:
    result = _list("--local", "--managed")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["id"] for row in payload["pots"]] == ["pot_l"]
    assert payload["unavailable"] == {"managed": "No managed host is configured."}


def test_managed_alone_on_an_install_without_it_is_not_implemented(
    local_only,
) -> None:
    result = _list("--managed")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "not_implemented"
    assert "No managed host is configured" in payload["message"]
    assert "--token -" in payload["recommended_next_action"]


def test_archived_pots_are_hidden_unless_asked_for(both_hosts) -> None:
    both_hosts["managed"].pots.append(
        PotInfo(pot_id="pot_old", name="old", archived=True)
    )

    hidden = json.loads(_list("--managed").output)
    shown = json.loads(_list("--managed", "--archived").output)

    assert [row["id"] for row in hidden["pots"]] == ["pot_m"]
    assert [(row["id"], row["archived"]) for row in shown["pots"]] == [
        ("pot_m", False),
        ("pot_old", True),
    ]
