"""Passage filtering and warnings survive CLI parsing and the typed protocol."""

from __future__ import annotations

import json
from dataclasses import fields

import pytest
from test_graph_cli_contract import _Backend, _Host
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph, query
from potpie.runtime import ContextSelector
from potpie.runtime.codec import (
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)
from potpie.runtime.operations import EngineOperation
from potpie.runtime.protocol import (
    PROTOCOL_VERSION,
    EngineOperationRequest,
    SuccessResponse,
)
from potpie_context_engine import Success
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.core.ports.agent_context import SearchRequest
from potpie_context_engine.core.ports.graph_service import GraphReadRequest
from potpie_context_engine.core.ports.resource_index import ChunkHit, IndexSearchResult
from potpie_context_engine.requests import ReadRequest

pytestmark = pytest.mark.unit


def _over_the_wire(request: GraphReadRequest, answer) -> tuple[ReadRequest, object]:
    """Carry one read through the daemon's typed codec, both directions."""
    payload = ReadRequest(
        **{field.name: getattr(request, field.name) for field in fields(ReadRequest)}
    )
    envelope = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="read-1",
        operation=EngineOperation.READ,
        selector=ContextSelector(kind="explicit", value="p"),
        payload=payload,
    )
    decoded_request = decode_request(encode_request(envelope))
    assert isinstance(decoded_request, Success)
    wire_request = decoded_request.value.payload
    result = answer(
        GraphReadRequest(pot_id=request.pot_id, **wire_request.to_payload())
    )
    response = SuccessResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=envelope.request_id,
        outcome=Success(result),
    )
    decoded_response = decode_response(encode_response(response), request=envelope)
    assert isinstance(decoded_response, Success)
    return wire_request, decoded_response.value.outcome.value


@pytest.fixture
def passage_service(monkeypatch):
    class Index:
        calibrated = True

        def search(self, **kwargs):
            return IndexSearchResult(
                profile="test",
                match_mode="hybrid",
                similarity_calibrated=self.calibrated,
                hits=(
                    ChunkHit(
                        resource_id="potpie://res/qme/definitions/0000",
                        doc="qme",
                        section="definitions",
                        seq=0,
                        document_key="document:qme",
                        section_key="docsection:qme:definitions",
                        section_title="Definitions",
                        label="QME",
                        snippet="A partial QME match",
                        chars=19,
                        revision=1,
                        score=0.1,
                        rank=1,
                        match_mode="hybrid",
                        similarity=0.1,
                        lexical_rank=1,
                        term_coverage=0.25,
                    ),
                ),
            )

    index = Index()
    service = DefaultGraphService(backend=InMemoryGraphBackend(), resource_index=index)
    requests: list[ReadRequest] = []

    class WireGraph:
        def read(self, request):
            wire_request, result = _over_the_wire(request, service.read)
            requests.append(wire_request)
            return result

    _common.set_runtime(_Host(WireGraph(), backend=_Backend()))
    monkeypatch.setattr(graph, "_empty_read_warnings", lambda *args: ())
    _common.set_json(False)
    yield service, index, requests
    _common.set_json(False)


def _read(*args):
    return CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "knowledge",
            "--view",
            "document_passages",
            "--query",
            "QME full form",
            "--limit",
            "1",
            *args,
        ],
    )


@pytest.mark.parametrize("format_", ["raw", "table"])
def test_default_human_read_warns_about_a_full_page_of_weak_evidence(
    passage_service, format_
):
    result = _read("--format", format_)
    assert result.exit_code == 0, result.output
    assert passage_service[2][0].query_threshold is None
    assert "items=1" in result.output
    assert "Weak passage matches" in result.output
    assert "query_threshold=auto" in result.output
    assert "match=hybrid" in result.output


def test_explicit_threshold_removes_weak_lexical_matches_in_human_output(
    passage_service,
):
    result = _read("--query-threshold", "0.7")
    assert result.exit_code == 0, result.output
    assert passage_service[2][0].query_threshold == 0.7
    assert "(no rows)" in result.output
    assert "No passages meet" in result.output
    assert "query_threshold=0.7" in result.output


def test_jsonl_keeps_warnings_on_stderr_and_rows_parseable(passage_service):
    result = _read("--format", "jsonl")
    assert result.exit_code == 0, result.output
    assert "Weak passage matches" in result.stderr
    assert len([json.loads(line) for line in result.stdout.splitlines()]) == 1


@pytest.mark.parametrize("threshold", [None, "0.7"])
def test_json_carries_warnings_and_effective_threshold(passage_service, threshold):
    _common.set_json(True)
    result = _read(*(["--query-threshold", threshold] if threshold else []))
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["ok"] is True
    assert body["warnings"]
    metadata = body["result"]["coverage"][0]["metadata"]
    assert metadata["query_threshold"] == (float(threshold) if threshold else None)
    assert metadata["match_mode"] == "hybrid"
    assert metadata["similarity_calibrated"] is True


@pytest.mark.parametrize("as_json", [False, True])
def test_uncalibrated_threshold_is_an_actionable_error(passage_service, as_json):
    passage_service[1].calibrated = False
    _common.set_json(as_json)
    result = _read("--query-threshold", "0.7")
    assert result.exit_code == 1, result.output
    assert "cannot apply a calibrated" in result.output
    assert "Omit --query-threshold" in result.output


def test_search_also_exposes_reader_warnings(passage_service):
    envelope = passage_service[0].search(
        SearchRequest(pot_id="p", query="QME", include=("resources",))
    )
    assert "Weak passage matches" in query._envelope_human(envelope)
    assert envelope.to_dict()["metadata"]["readers"]["resources"]["warnings"]
