"""Commit recovery through the real daemon transport, codec and CLI envelope.

A lost commit response is never answered by committing again: the CLI reads
the plan's durable receipt (a read that writes nothing) a bounded number of
times, and ``--verify`` runs as its own read after the receipt is durable.
"""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from typer.testing import CliRunner

from potpie.cli import commit_recovery
from potpie.cli.commands import _common, graph
from potpie.runtime import (
    ContextSelector,
    DaemonEngineClient,
    HttpDaemonTransport,
    RuntimeEndpoint,
)
from potpie.runtime.codec import decode_request, encode_response
from potpie.runtime.operations import (
    operation_capabilities,
    operation_catalog_fingerprint,
)
from potpie.runtime.protocol import (
    PROTOCOL_VERSION,
    FailureResponse,
    HandshakeRequest,
    HandshakeResult,
    SuccessResponse,
)
from potpie_context_engine import DomainError, Failure, Success
from potpie_context_engine.core.graph_plans import GraphIngestionVerificationResult
from tests.unit.test_graph_cli_contract import _commit_result, _Graph, _Host

pytestmark = pytest.mark.unit

HISTORY = "potpie --json graph history --plan mutation-plan:test --pot p"


class _Daemon:
    """A daemon reached over HTTP whose replies each test scripts by operation."""

    def __init__(self) -> None:
        self.replies: dict[str, list[object]] = {}
        self.calls: list[tuple[str, dict]] = []

    def handle(self, http_request: httpx.Request) -> httpx.Response:
        document = json.loads(http_request.content)
        decoded = decode_request(document)
        assert isinstance(decoded, Success), decoded
        request = decoded.value
        if isinstance(request, HandshakeRequest):
            value: object = HandshakeResult(
                protocol_min=PROTOCOL_VERSION,
                protocol_max=PROTOCOL_VERSION,
                instance_id="instance-1",
                lifecycle_state="ready",
                capabilities=operation_capabilities(),
                operation_catalog_fingerprint=operation_catalog_fingerprint(),
                compatibility_ticket="ticket",
            )
            return self._reply(request, Success(value))
        operation = request.operation.value
        self.calls.append((operation, document["payload"]))
        reply = self.replies[operation].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return self._reply(
            request, reply if isinstance(reply, Failure) else Success(reply)
        )

    @staticmethod
    def _reply(request, outcome) -> httpx.Response:
        response_type = (
            FailureResponse if isinstance(outcome, Failure) else SuccessResponse
        )
        return httpx.Response(
            200,
            json=encode_response(
                response_type(
                    protocol_version=PROTOCOL_VERSION,
                    request_id=request.request_id,
                    outcome=outcome,
                )
            ),
        )

    def operations(self) -> list[str]:
        return [operation for operation, _payload in self.calls]


@pytest.fixture
def daemon(monkeypatch):
    daemon = _Daemon()
    transport = HttpDaemonTransport(
        endpoint=RuntimeEndpoint(kind="tcp", address="127.0.0.1", port=9),
        bearer_token="t" * 43,
    )
    transport._client = httpx.AsyncClient(
        transport=httpx.MockTransport(daemon.handle), base_url="http://127.0.0.1:9"
    )
    client = DaemonEngineClient(
        selector=ContextSelector(kind="explicit", value="p"),
        transport=transport,
        expected_instance_id="instance-1",
    )
    _common.set_runtime(_Host(_Graph()))
    _common.set_json(True)
    assert _common.run_engine_outcome(client.handshake()).ok
    monkeypatch.setattr(graph, "get_engine_client", lambda *_a, **_k: client)
    monkeypatch.setattr(commit_recovery.time, "sleep", lambda _seconds: None)
    yield daemon
    _common.set_runtime(None)
    _common.set_json(False)


def _invoke(*args: str):
    return CliRunner().invoke(graph.graph_app, ["commit", "mutation-plan:test", *args])


def test_lost_response_recovers_the_receipt_without_committing_again(daemon):
    daemon.replies["commit"] = [httpx.ReadTimeout("late response")]
    daemon.replies["commit_status"] = [
        replace(_commit_result(), ok=False, status="committing"),
        _commit_result(),
    ]

    result = _invoke()

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)["result"]
    assert body["status"] == "committed" and body["mutation_id"] == "mutation-1"
    assert daemon.operations() == ["commit", "commit_status", "commit_status"]
    assert all(
        payload["plan_id"] == "mutation-plan:test" for _, payload in daemon.calls
    )


@pytest.mark.parametrize("json_mode", [True, False])
def test_an_unsettled_commit_keeps_its_failure_and_names_the_history_read(
    daemon, json_mode
):
    _common.set_json(json_mode)
    daemon.replies["commit"] = [httpx.ReadTimeout("late response")]
    daemon.replies["commit_status"] = [httpx.ReadTimeout("busy") for _ in range(3)]

    result = _invoke()

    # The same failure and exit status a lost commit response always had; the
    # next action now points at the read that settles it, never at a retry.
    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    assert HISTORY in " ".join(result.output.split())
    if json_mode:
        error = json.loads(result.output)["error"]
        assert error["code"] == "daemon_connection_lost"
        assert "may or may not have been applied" in error["message"]
    assert daemon.operations().count("commit") == 1
    assert len(daemon.calls) == 4


def test_a_refused_commit_is_reported_as_is_and_never_polled(daemon):
    daemon.replies["commit"] = [
        Failure(DomainError(code="plan_expired", message="plan expired"))
    ]

    result = _invoke()

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert json.loads(result.output)["error"]["code"] == "plan_expired"
    assert daemon.operations() == ["commit"]


def test_failed_verification_keeps_the_durable_receipt(daemon):
    daemon.replies["commit"] = [_commit_result()]
    daemon.replies["verify_commit"] = [httpx.ReadTimeout("readback still running")]

    result = _invoke("--verify")

    emitted = json.loads(result.output)
    body = emitted["result"]
    assert emitted["ok"] and body["status"] == "committed"
    assert body["mutation_id"] == "mutation-1"
    assert body["verification"]["status"] == "error"
    assert body["verification"]["recommended_next_action"] == (
        "potpie --json graph commit mutation-plan:test --verify --pot p"
    )
    # A verification that did not pass still exits 1, as before.
    assert result.exit_code == _common.EXIT_VALIDATION
    assert daemon.operations() == ["commit", "verify_commit"]
    assert daemon.calls[0][1]["defer_verification"] is True
    assert daemon.calls[0][1]["verify"] is True


def test_an_overlapping_commit_is_polled_then_verified(daemon):
    daemon.replies["commit"] = [
        replace(_commit_result(), ok=False, status="committing")
    ]
    daemon.replies["commit_status"] = [_commit_result()]
    daemon.replies["verify_commit"] = [
        GraphIngestionVerificationResult(
            ok=True, status="ok", plan_id="mutation-plan:test", pot_id="p"
        )
    ]

    result = _invoke("--verify")

    assert result.exit_code == 0, result.output
    assert daemon.operations() == ["commit", "commit_status", "verify_commit"]
    assert json.loads(result.output)["result"]["verification"]["ok"]


def test_a_commit_still_running_elsewhere_is_reported_with_its_history_read(daemon):
    committing = replace(_commit_result(), ok=False, status="committing")
    daemon.replies["commit"] = [committing]
    daemon.replies["commit_status"] = [committing, committing, committing]

    result = _invoke()

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    emitted = json.loads(result.output)
    assert emitted["error"]["code"] == "committing"
    assert emitted["recommended_next_action"] == HISTORY
    assert daemon.operations().count("commit") == 1


def test_commit_offers_no_client_side_deadline_option(daemon):
    """Timeouts and cancellation are not yet part of the typed protocol."""

    result = _invoke("--timeout", "45")

    assert result.exit_code == 2
    assert daemon.calls == []
