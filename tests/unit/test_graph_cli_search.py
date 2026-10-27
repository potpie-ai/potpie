"""Graph CLI search behavior."""

from __future__ import annotations

import json

import pytest
from potpie_context_engine.core.ports.graph_service import (
    GraphEntityCandidate,
    GraphEntitySearchResult,
)
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_support import _assert_graph_envelope, _Graph, _Host
from tests.graph_cli_support import graph_cli_state as graph_cli_state

pytestmark = pytest.mark.unit


def test_graph_entity_search_result_includes_summary() -> None:
    result = GraphEntitySearchResult(
        entities=(
            GraphEntityCandidate(
                key="service:web",
                labels=("Service",),
                name="web",
                summary="Web frontend service.",
                description="Web frontend service for browser clients.",
            ),
        ),
        match_mode="lexical",
        graph_contract_version="v1.5",
        ontology_version="test",
    )

    payload = result.to_dict()

    assert payload["entities"][0]["summary"] == "Web frontend service."


def test_graph_search_entities_omits_supporting_claims_by_default() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["search-entities", "payments"],
    )

    assert result.exit_code == 0
    assert graph_service.search_request.supporting_claims == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.search-entities")
    assert body["entities"][0]["supporting_claims"] == []


def test_graph_search_entities_supporting_claims_is_opt_in() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["search-entities", "payments", "--supporting-claims", "1"],
    )

    assert result.exit_code == 0
    assert graph_service.search_request.supporting_claims == 1
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.search-entities")
    assert body["entities"][0]["supporting_claims"][0]["predicate"] == "PROVIDES"


def test_graph_search_entities_threads_source_ref_filter() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["search-entities", "payments", "--source-ref", "github:pr:955"],
    )

    assert result.exit_code == 0
    assert graph_service.search_request.source_refs == ("github:pr:955",)


def test_graph_search_entities_threads_source_facets() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "search-entities",
            "payments",
            "--source-system",
            "github",
            "--source-family",
            "github",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.search_request.source_system == "github"
    assert graph_service.search_request.source_family == "github"
