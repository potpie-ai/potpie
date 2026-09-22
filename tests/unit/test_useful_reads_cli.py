"""CLI/RPC checks for truthful partial read receipts."""

import json

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from potpie.daemon.rpc import decode, encode
from potpie_context_core.ports.graph_service import GraphReadRequest
from potpie_context_core.ports.resource_store import (
    ResourceBatchResult,
    format_resource_id,
)
from test_resource_cli_contract import _seed, DOC
from test_graph_cli_contract import _Host, _Backend
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_mode():
    yield
    _common.set_json(False)


@pytest.mark.parametrize("as_json", [True, False])
def test_mixed_resource_cli_keeps_text_and_exits_nonzero(tmp_path, as_json):
    host = _seed(tmp_path)
    valid, invalid = (format_resource_id(DOC, "body", seq) for seq in (0, 9))
    _common.set_json(as_json)
    from potpie.cli.commands import resource

    result = CliRunner().invoke(resource.resource_app, ["get", valid, invalid])
    assert result.exit_code == 1, result.output
    if as_json:
        payload = json.loads(result.stdout)
        assert payload["status"] == "partial" and payload["chunks"]
        assert payload["chunks"][0]["requested"] is True
        assert [outcome["resource_id"] for outcome in payload["outcomes"]] == [
            valid,
            invalid,
        ]
    else:
        assert "partial" in result.stdout and invalid in result.stdout
        assert "not_found" in result.stdout
        assert "[neighbor]" not in result.stdout
    receipt = host.resources.get(pot_id="p", resource_ids=(valid, invalid))
    assert isinstance(receipt, ResourceBatchResult)
    assert decode(encode(receipt)) == receipt


def test_partial_graph_result_survives_rpc_and_text_renders_constraint():
    service = DefaultGraphService(backend=InMemoryGraphBackend())
    _common.set_host(_Host(service, backend=_Backend()))
    args = [
        "read",
        "--subgraph",
        "debugging",
        "--view",
        "prior_occurrences",
        "--query",
        "timeout",
        "--since",
        "2100-01-01",
    ]
    _common.set_json(False)
    result = CliRunner().invoke(graph.graph_app, args)
    assert result.exit_code != 0
    assert "Occurrence window not applied" in result.stdout
    assert "Supplemental context" in result.stdout
    receipt = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="debugging",
            view="prior_occurrences",
            query="timeout",
            query_threshold=1.0,
        )
    )
    assert decode(encode(receipt)) == receipt


def test_partial_graph_json_contains_empty_answer_and_separate_context():
    service = DefaultGraphService(backend=InMemoryGraphBackend())
    _common.set_host(_Host(service, backend=_Backend()))
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
    service = DefaultGraphService(backend=InMemoryGraphBackend())
    _common.set_host(_Host(service, backend=_Backend()))
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


@pytest.mark.parametrize("as_json", [True, False])
def test_failed_ambiguous_root_is_not_labelled_as_requested_neighbor(tmp_path, as_json):
    from potpie.cli.commands import resource
    from potpie_context_engine.testing import write_import_directory

    host = _seed(tmp_path)
    source = write_import_directory(
        tmp_path / "revision-two",
        [
            {
                "slug": "body",
                "summary": "changed document",
                "chunks": [
                    {"seq": i, "label": f"chunk {i}", "text": f"changed {i}"}
                    for i in range(3)
                ],
            }
        ],
    )
    host.store.inner.import_dir(pot_id="p", slug=DOC, source_dir=source)
    ambiguous = format_resource_id(DOC, "body", 0)
    valid = format_resource_id(DOC, "body", 1, revision=1)
    _common.set_json(as_json)
    result = CliRunner().invoke(
        resource.resource_app, ["get", ambiguous, valid, "--with-neighbors"]
    )
    assert result.exit_code == 1, result.output
    if as_json:
        payload = json.loads(result.stdout)
        assert payload["outcomes"][0]["status"] == "error"
        body = next(chunk for chunk in payload["chunks"] if chunk["seq"] == 0)
        assert body["requested"] is False and body["revision"] == 1
    else:
        assert (
            f"{format_resource_id(DOC, 'body', 0, revision=1)} [neighbor]"
            in result.stdout
        )
        assert "resource_revision_ambiguous" in result.stdout
