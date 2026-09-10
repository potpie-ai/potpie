"""Passage filtering and warnings survive CLI parsing and RPC serialization."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph, query
from potpie.daemon.rpc import decode, encode
from potpie_context_core.ports.agent_context import SearchRequest
from potpie_context_core.ports.resource_index import ChunkHit, IndexSearchResult
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService


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
                        resource_id="potpie://res/pms/definitions/0000",
                        doc="pms",
                        section="definitions",
                        seq=0,
                        document_key="document:pms",
                        section_key="docsection:pms:definitions",
                        section_title="Definitions",
                        label="PMS",
                        snippet="A partial PMS match",
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
    requests = []

    class RpcGraph:
        def read(self, request):
            request = decode(encode(request))
            requests.append(request)
            return decode(encode(service.read(request)))

    monkeypatch.setattr(graph, "get_host", lambda: SimpleNamespace(graph=RpcGraph()))
    monkeypatch.setattr(graph, "resolve_pot_id", lambda host, pot: "p")
    monkeypatch.setattr(graph, "pot_scope_human", lambda host, pot_id: "pot=p")
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
            "PMS full form",
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
        SearchRequest(pot_id="p", query="PMS", include=("resources",))
    )
    assert "Weak passage matches" in query._envelope_human(envelope)
    assert envelope.to_dict()["metadata"]["readers"]["resources"]["warnings"]
