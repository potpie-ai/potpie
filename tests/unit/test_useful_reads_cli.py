"""CLI and typed-protocol checks for truthful partial read receipts."""

from __future__ import annotations

import json

import pytest
from test_graph_cli_contract import _Backend, _Host
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from potpie.runtime.codec import decode_response, encode_response
from potpie.runtime.operations import EngineOperation
from potpie.runtime.protocol import (
    PROTOCOL_VERSION,
    EngineOperationRequest,
    SuccessResponse,
)
from potpie.runtime.resource_manager import ContextSelector
from potpie_context_engine import Success
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.core.ports.graph_service import GraphReadRequest
from potpie_context_engine.requests import ReadRequest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_mode():
    yield
    _common.set_json(False)


def _service() -> DefaultGraphService:
    service = DefaultGraphService(backend=InMemoryGraphBackend())
    _common.set_runtime(_Host(service, backend=_Backend()))
    return service


def test_partial_graph_result_text_renders_the_constraint_and_context():
    _service()
    _common.set_json(False)
    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "debugging",
            "--view",
            "prior_occurrences",
            "--query",
            "timeout",
            "--since",
            "2100-01-01",
        ],
    )
    assert result.exit_code != 0
    assert "Occurrence window not applied" in result.stdout
    assert "Supplemental context" in result.stdout


def test_partial_graph_result_survives_the_typed_protocol():
    service = _service()
    receipt = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="debugging",
            view="prior_occurrences",
            query="timeout",
            query_threshold=1.0,
        )
    )
    assert receipt.ok is False and receipt.fallback_context
    request = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="read-1",
        operation=EngineOperation.READ,
        selector=ContextSelector(kind="explicit", value="p"),
        payload=ReadRequest(
            subgraph="debugging", view="prior_occurrences", query="timeout"
        ),
    )
    response = SuccessResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=request.request_id,
        outcome=Success(receipt),
    )

    decoded = decode_response(encode_response(response), request=request)

    assert isinstance(decoded, Success)
    assert decoded.value.outcome.value == receipt


def test_partial_graph_json_contains_empty_answer_and_separate_context():
    _service()
    _common.set_json(True)
    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "knowledge",
            "--view",
            "document_context",
            "--query",
            "timeout",
            "--query-threshold",
            "1",
        ],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert not payload["ok"]
    assert "fallback_context" in json.dumps(payload)
    assert "Threshold not applied" in json.dumps(payload)


def test_partial_jsonl_is_a_single_machine_readable_receipt():
    _service()
    _common.set_json(False)
    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "knowledge",
            "--view",
            "document_context",
            "--query",
            "timeout",
            "--query-threshold",
            "1",
            "--format",
            "jsonl",
        ],
    )
    assert result.exit_code != 0
    assert len(result.stdout.splitlines()) == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "partial" and not payload["items"]
    assert "fallback_context" in payload


def test_a_read_without_a_threshold_leaves_filtering_to_the_view():
    """No ``--query-threshold`` means no constraint: a view that cannot apply
    one must not answer every read with a partial receipt."""
    _service()
    _common.set_json(True)
    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "features", "--view", "feature_context"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["unsupported"] == []
