"""A daemon left running from another Potpie version is named, with its repair.

Every release that adds an operation changes the catalog fingerprint, and the
running daemon refuses the new client's handshake. The CLI must say what to do
rather than relay "catalogs do not match".
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from potpie.runtime import DaemonEngineClient, ProtocolError, RuntimeEndpoint
from potpie_context_engine import Failure
from tests.unit.test_graph_cli_contract import _Graph, _Host

pytestmark = pytest.mark.unit


def test_a_catalog_mismatch_names_the_restart_and_the_old_process(monkeypatch):
    from potpie.daemon import discovery

    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "daemon")
    monkeypatch.setattr(
        discovery,
        "load_daemon_connection",
        lambda _home: SimpleNamespace(
            discovery=SimpleNamespace(
                instance_id="instance-1",
                pid=4242,
                endpoint=RuntimeEndpoint(kind="tcp", address="127.0.0.1", port=9),
            ),
            bearer_token="t" * 43,
        ),
    )

    async def refused(self):
        return Failure(
            ProtocolError(
                code="operation_catalog_mismatch",
                message="client and daemon operation catalogs do not match",
                recommended_next_action="restart with a compatible Potpie version",
            )
        )

    monkeypatch.setattr(DaemonEngineClient, "handshake", refused)
    _common.set_runtime(_Host(_Graph()))
    _common.set_json(True)
    try:
        result = CliRunner().invoke(graph.graph_app, ["commits", "--pot", "p"])
    finally:
        _common.set_json(False)
        _common.set_runtime(None)

    error = json.loads(result.output)["error"]
    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    assert error["code"] == "operation_catalog_mismatch"
    assert "different Potpie version" in error["message"]
    next_action = json.loads(result.output)["recommended_next_action"]
    assert "potpie daemon restart" in next_action
    assert "4242" in next_action


def test_other_handshake_refusals_pass_through_unchanged():
    error = ProtocolError(code="daemon_not_ready", message="not ready")

    assert _common._actionable_handshake_error(error, pid=1) is error
