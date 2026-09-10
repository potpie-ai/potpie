"""Exercise commit recovery through the real RPC codec and CLI envelope."""

from dataclasses import replace
import json
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from potpie.cli import commit_recovery, hosts
from potpie.cli.commands import _common, graph
from potpie.daemon.client import DaemonRpcClient, RemoteSurface
from potpie.daemon.rpc import decode, encode
from potpie_context_core.graph_plans import GraphIngestionVerificationResult
from tests._rpc_fakes import install_rpc_session
from tests.unit.test_graph_cli_contract import _Host, _Graph, _commit_result


@pytest.fixture
def remote(monkeypatch):
    calls = []
    responses = {}

    def post(url, **kwargs):
        payload = kwargs["json"]
        method = payload["method"]
        args, options = decode(payload["args"]), decode(payload["kwargs"])
        calls.append((method, args, options, kwargs["timeout"], url))
        response = responses[method].pop(0)
        if isinstance(response, Exception):
            raise response
        return httpx.Response(200, json={"ok": True, "result": encode(response)})

    install_rpc_session(monkeypatch, post)
    rpc = DaemonRpcClient(
        daemon=SimpleNamespace(
            discovery=lambda: {
                "base_url": "http://managed.example",
                "token": "test-token",
            }
        )
    )
    surface = RemoteSurface(rpc, "graph_workbench")
    _common.set_host(_Host(_Graph(), graph_workbench=surface))
    _common.set_json(True)
    monkeypatch.setattr(hosts, "current_origin", lambda: "managed")
    monkeypatch.setattr(commit_recovery.time, "sleep", lambda _: None)
    yield responses, calls, rpc
    _common.set_host(None)
    _common.set_json(False)


def _invoke(*args):
    return CliRunner().invoke(graph.graph_app, ["commit", "mutation-plan:test", *args])


def test_timeout_recovers_the_exact_receipt_without_retrying_write(remote):
    responses, calls, rpc = remote
    responses["commit"] = [httpx.ReadTimeout("late response")]
    responses["commit_status"] = [
        replace(_commit_result(), ok=False, status="committing"),
        _commit_result(),
    ]
    result = _invoke("--timeout", "45")
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)["result"]
    assert body["status"] == "committed" and body["mutation_id"] == "mutation-1"
    assert [c[0] for c in calls] == ["commit", "commit_status", "commit_status"]
    assert [c[3] for c in calls] == [45, 2, 2]
    assert all(c[1] == ("mutation-plan:test",) and c[2]["pot_id"] == "p" for c in calls)
    assert all(c[4] == "http://managed.example/rpc" for c in calls)
    assert rpc.timeout_s == 30


@pytest.mark.parametrize("json_mode", [True, False])
def test_unresolved_commit_reports_unknown_and_scoped_history(remote, json_mode):
    responses, calls, _ = remote
    _common.set_json(json_mode)
    responses["commit"] = [httpx.ReadTimeout("late response")]
    responses["commit_status"] = [httpx.ReadTimeout("busy") for _ in range(3)]
    result = _invoke()
    assert result.exit_code == 2, result.output
    assert "unknown_completion" in result.output
    assert (
        "potpie --json graph history --plan mutation-plan:test --pot managed:p"
        in result.output
    )
    if json_mode:
        assert json.loads(result.output)["error"]["code"] == "unknown_completion"
    assert [c[0] for c in calls].count("commit") == 1
    assert len(calls) == 4


def test_verification_over_default_deadline_keeps_successful_receipt(remote):
    responses, calls, _ = remote
    responses["commit"] = [_commit_result()]
    # Simulates a server readback lasting 31s against the default 30s RPC
    # deadline. Service tests exercise the write/verification separation.
    responses["verify_commit"] = [
        httpx.ReadTimeout("verification still running after 31s")
    ]
    result = _invoke("--verify")
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)["result"]
    assert json.loads(result.output)["ok"] and body["status"] == "committed"
    assert body["mutation_id"] == "mutation-1"
    assert body["verification"]["status"] == "unknown_completion"
    assert body["verification"]["recommended_next_action"].endswith(
        "--verify --pot managed:p"
    )
    assert [c[0] for c in calls] == ["commit", "verify_commit"]
    assert calls[0][2]["defer_verification"] is True
    assert calls[1][3] == 30


def test_retry_while_first_commit_runs_polls_then_verifies(remote):
    responses, calls, _ = remote
    responses["commit"] = [replace(_commit_result(), ok=False, status="committing")]
    responses["commit_status"] = [_commit_result()]
    responses["verify_commit"] = [
        GraphIngestionVerificationResult(
            ok=True,
            status="ok",
            plan_id="mutation-plan:test",
            pot_id="p",
        )
    ]
    result = _invoke("--verify")
    assert result.exit_code == 0, result.output
    assert [c[0] for c in calls] == ["commit", "commit_status", "verify_commit"]
    assert json.loads(result.output)["result"]["verification"]["ok"]


def test_configured_timeout_reaches_transport_only(remote, monkeypatch):
    responses, calls, _ = remote
    responses["commit"] = [_commit_result()]
    monkeypatch.setenv("POTPIE_GRAPH_COMMIT_TIMEOUT", "90")
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert calls[0][3] == 90
    assert "timeout" not in calls[0][2]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_timeout_never_submits_commit(remote, value):
    _, calls, _ = remote
    result = _invoke("--timeout", value)
    assert result.exit_code == 1, result.output
    assert not calls


@pytest.mark.parametrize("status", ["error", "conflict"])
def test_timeout_returns_a_persisted_failure_instead_of_unknown(remote, status):
    responses, calls, _ = remote
    responses["commit"] = [httpx.ReadTimeout("late response")]
    responses["commit_status"] = [
        replace(
            _commit_result(),
            ok=False,
            status=status,
            detail="backend rejected mutation",
        )
    ]
    result = _invoke()
    assert result.exit_code == 1, result.output
    assert json.loads(result.output)["error"]["code"] == status
    assert "backend rejected mutation" in result.output
    assert len(calls) == 2
