"""`potpie use` origin selection: the flags pick the host, not the wording.

The origin the CLI says it used has to be the origin it actually used, an
explicit origin is a target rather than a starting point, and anything the CLI
cannot decide is a refusal. Pot directories are in-memory doubles.
"""

from __future__ import annotations

import json
import socket

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common
from potpie.cli.main import app
from potpie.config import host_registry
from potpie.pots.contracts import PotInfo
from tests.support.pot_directories import two_hosts


class _SocketOpenedInUnitTest(BaseException):
    """Outside ``Exception`` on purpose: the CLI swallows that hierarchy."""


@pytest.fixture(autouse=True)
def _no_sockets(monkeypatch):
    """A unit test here must never dial out, loopback most of all."""

    def _refuse(*args, **_kwargs):
        raise _SocketOpenedInUnitTest(f"unit test opened a socket: {args!r}")

    real_connect = socket.socket.connect

    def _refuse_network(sock, address, *args, **kwargs):
        if sock.family == getattr(socket, "AF_UNIX", None):
            return real_connect(sock, address, *args, **kwargs)
        _refuse(sock, address)

    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)


@pytest.fixture
def registry(monkeypatch):
    """Two configured hosts, each with a pot named ``shared``.

    The shared name is the point: it is what makes a wrong-host selection
    silent, because both hosts answer the ref and only the id differs.
    """
    host_registry.reset_for_tests()
    built = two_hosts(
        monkeypatch,
        local=[
            PotInfo(pot_id="pot_l_shared", name="shared", active=True),
            PotInfo(pot_id="pot_l2", name="notes"),
        ],
        managed=[
            PotInfo(pot_id="pot_m_shared", name="shared", active=True),
            PotInfo(pot_id="pot_m2", name="api"),
        ],
    )
    yield built
    _common.set_pot_directory_factory(None)
    host_registry.reset_for_tests()


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


def _stored_origin() -> str | None:
    path = host_registry.registry_path()
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("active_origin")


# --- the flags select the host ----------------------------------------------


def test_local_flag_selects_the_local_pot_from_a_managed_session(registry) -> None:
    host_registry.set_persisted_origin(host_registry.MANAGED)

    result = _run("--json", "use", "shared", "--local")

    assert result.exit_code == 0, result.output
    assert registry["local"].used == ["shared"]
    assert registry["managed"].used == []
    payload = json.loads(result.output)
    assert (payload["id"], payload["origin"]) == ("pot_l_shared", "local")
    assert _stored_origin() == "local"


def test_managed_flag_selects_the_managed_pot_from_a_local_session(registry) -> None:
    result = _run("--json", "use", "shared", "--managed")

    assert result.exit_code == 0, result.output
    assert registry["managed"].used == ["shared"]
    assert registry["local"].used == []
    payload = json.loads(result.output)
    assert (payload["id"], payload["origin"]) == ("pot_m_shared", "managed")
    assert _stored_origin() == "managed"


def test_the_flag_is_equivalent_to_qualifying_the_ref(registry) -> None:
    host_registry.set_persisted_origin(host_registry.MANAGED)
    flagged = json.loads(_run("--json", "use", "shared", "--local").output)
    host_registry.set_persisted_origin(host_registry.MANAGED)
    host_registry.reset_for_tests()
    qualified = json.loads(_run("--json", "use", "local:shared").output)

    assert flagged["id"] == qualified["id"] == "pot_l_shared"
    assert flagged["origin"] == qualified["origin"] == "local"


def test_a_bare_ref_selects_on_the_current_origin(registry) -> None:
    """Deliberate deviation from searching both hosts: 'shared' on both is not
    ambiguous, it is the current origin's pot."""
    host_registry.set_persisted_origin(host_registry.MANAGED)

    result = _run("--json", "use", "shared")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == "pot_m_shared"
    assert registry["local"].list_calls == 0


def test_an_explicit_origin_never_falls_through_to_the_other_host(registry) -> None:
    """`notes` exists only locally; `--managed` has to fail there."""
    result = _run("--json", "use", "notes", "--managed")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert "'local:notes'" in payload["message"]
    assert "potpie use local:notes" in payload["recommended_next_action"]
    assert registry["local"].used == []
    assert registry["managed"].used == []
    assert host_registry.persisted_origin() == "local"


def test_the_human_line_names_the_origin_that_was_resolved(registry) -> None:
    host_registry.set_persisted_origin(host_registry.MANAGED)

    result = _run("use", "shared", "--local")

    assert result.exit_code == 0, result.output
    assert "shared (local)" in result.output


# --- refusals ----------------------------------------------------------------


def test_both_flags_at_once_is_refused(registry) -> None:
    result = _run("--json", "use", "shared", "--local", "--managed")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "--local" in payload["message"] and "--managed" in payload["message"]
    assert registry["local"].used == registry["managed"].used == []


def test_a_flag_contradicting_the_qualified_ref_is_refused(registry) -> None:
    result = _run("--json", "use", "managed:shared", "--local")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "--local" in payload["message"]
    assert "managed:shared" in payload["message"]
    assert registry["local"].used == registry["managed"].used == []


def test_managed_without_a_configured_host_reports_the_configured_error(
    monkeypatch,
) -> None:
    host_registry.reset_for_tests()
    two_hosts(monkeypatch, managed_configured=False)

    result = _run("--json", "use", "shared", "--managed")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "No managed host is configured" in payload["message"]
    assert "host set" in payload["recommended_next_action"]
    assert "Traceback" not in result.output


def test_an_unreachable_host_still_lets_a_qualified_ref_through(registry) -> None:
    from potpie_context_engine.core.errors import ContextEngineDisabled

    registry["managed"].raises = ContextEngineDisabled("connection refused")
    host_registry.set_persisted_origin(host_registry.MANAGED)

    result = _run("--json", "use", "local:notes")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == "pot_l2"
    assert host_registry.persisted_origin() == "local"


def test_a_targeted_host_that_is_down_is_reported_not_rescued(registry) -> None:
    from potpie_context_engine.core.errors import ContextEngineDisabled

    registry["managed"].raises = ContextEngineDisabled("connection refused")

    result = _run("--json", "use", "managed:api")

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    payload = json.loads(result.output)
    assert payload["code"] == "unavailable"
    assert "connection refused" in payload["message"]
    assert registry["local"].used == []
    assert host_registry.persisted_origin() == "local"


def test_a_refused_selection_leaves_the_pointer_alone(registry) -> None:
    host_registry.set_persisted_origin(host_registry.MANAGED)

    result = _run("--json", "use", "notes")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert "'local:notes'" in payload["message"]
    assert registry["local"].used == registry["managed"].used == []
    assert host_registry.persisted_origin() == "managed"
