"""A graph store that refuses to serve possibly stale data must name its repair.

The embedded graph refuses to open after its server died without saving and
without a complete AOF, and carries the operator's recovery step on the error.
These pin that the step reaches the user through both the in-process CLI
contract and the daemon's typed failure, and that the daemon stops the embedded
graph server it started when it shuts down.
"""

from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

import json
from types import SimpleNamespace

import pytest
import typer

from potpie.cli.commands import _common
from potpie.daemon import __main__ as daemon_main
from potpie.runtime.local_engine import LocalEngineOperations
from potpie_context_engine import ContextIdentity, Failure, Success
from potpie_context_engine.adapters.outbound.graph import falkordb_writer
from potpie_context_engine.core.errors import (
    ContextEngineDisabled,
    GraphSubstrateUnavailable,
)
from potpie_context_engine.requests import SearchRequest

_REPAIR = "delete the graph's settings file and re-run"
_GENERIC = "check backend/daemon readiness with 'potpie doctor'"


def _refusal() -> GraphSubstrateUnavailable:
    return GraphSubstrateUnavailable(
        "the embedded graph cannot be served as complete",
        recommended_next_action=_REPAIR,
    )


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (_refusal(), _REPAIR),
        (ContextEngineDisabled("graph clients unavailable"), _GENERIC),
    ],
)
def test_cli_contract_prefers_the_errors_own_repair(
    raised: ContextEngineDisabled,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            raise raised

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.exit_code == _common.EXIT_UNAVAILABLE
    assert payload["code"] == "unavailable"
    assert payload["recommended_next_action"] == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (_refusal(), _REPAIR),
        (ContextEngineDisabled("graph clients unavailable"), _GENERIC),
    ],
)
async def test_daemon_operation_failure_carries_the_repair(
    raised: ContextEngineDisabled, expected: str
) -> None:
    def refuse(_request: object) -> object:
        raise raised

    operations = LocalEngineOperations(
        SimpleNamespace(agent_context=SimpleNamespace(search=refuse))
    )

    outcome = await operations.search(
        ContextIdentity("pot-1"), SearchRequest(query="what changed")
    )

    assert isinstance(outcome, Failure)
    assert outcome.error.category == "dependency"
    assert outcome.error.code == "unavailable"
    assert outcome.error.recommended_next_action == expected


@pytest.mark.anyio
async def test_daemon_shutdown_stops_embedded_servers_after_releasing_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        falkordb_writer,
        "shutdown_embedded_servers",
        lambda: order.append("embedded servers") or 1,
    )

    async def release() -> object:
        order.append("resources")
        return Success(None)

    outcome = await daemon_main._then_stop_embedded_graph_servers(release)()

    assert outcome == Success(None)
    assert order == ["resources", "embedded servers"]


@pytest.mark.anyio
async def test_daemon_shutdown_stops_embedded_servers_even_when_release_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[bool] = []
    monkeypatch.setattr(
        falkordb_writer,
        "shutdown_embedded_servers",
        lambda: stopped.append(True) or 1,
    )

    async def release() -> object:
        raise RuntimeError("engine close failed")

    with pytest.raises(RuntimeError, match="engine close failed"):
        await daemon_main._then_stop_embedded_graph_servers(release)()

    assert stopped == [True]


def test_stopping_embedded_servers_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> int:
        raise ConnectionError("server already gone")

    monkeypatch.setattr(falkordb_writer, "shutdown_embedded_servers", boom)

    daemon_main._stop_embedded_graph_servers()  # must not raise
