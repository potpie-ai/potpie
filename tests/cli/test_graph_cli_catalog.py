"""Graph CLI catalog behavior."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_support import (
    _assert_graph_envelope,
    _Graph,
    _Host,
    _NoSelectedPots,
    _plain_cli_output,
)
from tests.graph_cli_support import graph_cli_state as graph_cli_state

pytestmark = pytest.mark.unit


def test_graph_catalog_json_advertises_v2_workbench_commands() -> None:
    _common.set_json(True)
    _common.set_runtime(_Host(_Graph()))

    result = CliRunner().invoke(graph.graph_app, ["catalog"])

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.catalog")
    assert "propose" in body["commands"]
    assert "commit" in body["commands"]
    assert "mutate" not in body["commands"]
    assert "legacy_commands" not in body
    assert body["data_plane_graph_contract_version"] == "v1.5"


def test_graph_catalog_task_ranks_relevant_views() -> None:
    _common.set_json(True)
    _common.set_runtime(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["catalog", "--task", "debug staging timeout after deployment"],
    )

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.catalog")
    ranking = body["task_ranking"]
    ranked_subgraphs = [entry["subgraph"] for entry in ranking]
    assert ranked_subgraphs.index("debugging") < ranked_subgraphs.index("features")
    assert ranked_subgraphs.index("recent_changes") < ranked_subgraphs.index("features")
    assert ranked_subgraphs.index("infra_topology") < ranked_subgraphs.index("features")
    assert ranked_subgraphs.index("decisions") < ranked_subgraphs.index("features")
    assert body["views"][0]["name"] == ranking[0]["view"]
    assert "reason" in ranking[0]


def test_graph_catalog_read_profile_returns_compact_contract() -> None:
    _common.set_json(True)
    _common.set_runtime(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["catalog", "--task", "debug staging timeout", "--profile", "read"],
    )

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.catalog")
    assert body["profile"] == "read"
    assert "read" in body["commands"]
    assert "commit" not in body["commands"]
    assert "mutation_operations" not in body
    assert "entity_types" not in body
    assert body["views"]
    assert set(body["views"][0]) <= {
        "name",
        "subgraph",
        "view",
        "backed",
        "description",
        "result_shape",
        "required_scope",
        "required_any_scope",
        "supported_filters",
        "next_read",
        "next_read_is_template",
    }
    assert body["task_ranking"][0]["rank"] == 1
    assert "reason" in body["task_ranking"][0]
    assert "matched_terms" in body["task_ranking"][0]


def test_graph_catalog_table_format_uses_compact_human_output() -> None:
    _common.set_json(False)
    _common.set_runtime(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["catalog", "--profile", "read", "--format", "table"],
    )

    assert result.exit_code == 0
    assert "graph catalog profile=read" in result.output
    assert "view | backed | requires | filters" in result.output


def test_graph_catalog_table_format_shows_task_ranking_context() -> None:
    _common.set_json(False)
    _common.set_runtime(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "catalog",
            "--task",
            "debug staging timeout",
            "--profile",
            "read",
            "--format",
            "table",
        ],
    )

    assert result.exit_code == 0
    assert "rank | score | view | reason" in result.output
    assert "Matched" in result.output


def test_graph_catalog_unknown_subgraph_uses_error_envelope() -> None:
    _common.set_json(True)
    _common.set_runtime(
        _Host(_Graph(catalog_error=ValueError("unknown graph subgraph")))
    )

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--subgraph", "missing"])

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.catalog", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "unknown graph subgraph" in emitted["error"]["message"]


def test_graph_catalog_include_guess_error_carries_did_you_mean() -> None:
    # Audit item 17 first-contact path: an include family typed where a
    # subgraph is expected gets the same migration guidance as read.
    from potpie_context_engine.core.graph_views import (
        UnknownGraphViewError,
        include_guess_guidance,
    )

    _common.set_json(True)
    guidance = include_guess_guidance("docs", None)
    _common.set_runtime(
        _Host(
            _Graph(
                catalog_error=UnknownGraphViewError(
                    "unknown graph subgraph 'docs'",
                    did_you_mean=guidance,
                    recommended_next_action=guidance["read_command"],
                )
            )
        )
    )

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--subgraph", "docs"])

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.catalog", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    did_you_mean = emitted["error"]["detail"]["did_you_mean"]
    assert did_you_mean["view"] == "knowledge.document_context"
    assert did_you_mean["matched_include"] == "docs"
    assert emitted["recommended_next_action"] == (
        "potpie graph read --subgraph knowledge --view document_context"
    )


def test_graph_describe_returns_executable_view_contract() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    host = _Host(graph_service)
    host.pots = _NoSelectedPots()
    _common.set_runtime(host)

    result = CliRunner().invoke(
        graph.graph_app,
        ["describe", "debugging", "--view", "prior_occurrences", "--examples"],
    )

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.describe")
    assert body["contract_kind"] == "graph_workbench_ontology"
    assert body["view"]["name"] == "debugging.prior_occurrences"
    assert body["view"]["result_shape"] == "entity_relations"
    assert "REPRODUCES" in body["view"]["inline_relations"]
    assert body["view"]["examples"][0]["command"].startswith("potpie graph read")
    # Examples are authored in the ontology and are templates: their scope
    # values are illustrative, and they say so.
    assert body["view"]["examples"][0]["template"] is True
    # The next read is templated over the view's own selector rule
    # (prior_occurrences needs one of query/service/repo), instead of a bare
    # command that fails on the missing selector.
    assert emitted["recommended_next_action"] == (
        "Fill the placeholders, then run `potpie graph read --subgraph debugging "
        "--view prior_occurrences --query '<query>' --json`."
    )
    # The CLI is a thin client: the contract must be answered through the
    # service request, never a CLI-local domain call.
    assert graph_service.describe_request is not None
    assert graph_service.describe_request.subgraph == "debugging"
    assert graph_service.describe_request.view == "prior_occurrences"
    assert graph_service.describe_request.include_examples is True


def test_graph_describe_requires_subgraph_before_dispatch() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(graph.graph_app, ["describe"])

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.describe", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert emitted["error"]["message"] == "subgraph is required"
    assert graph_service.describe_request is None


def test_graph_describe_unknown_view_uses_error_envelope() -> None:
    _common.set_json(True)
    _common.set_runtime(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["describe", "debugging", "--view", "missing"],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.describe", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "unknown graph view" in emitted["error"]["message"]


@pytest.mark.parametrize("format_", ["auto", "table"])
def test_full_catalog_exposes_the_ontology_for_ingestion(format_) -> None:
    from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
        InMemoryGraphBackend,
    )
    from potpie_context_engine.application.services.graph_service import (
        DefaultGraphService,
    )
    from potpie_context_engine.core.ontology import EDGE_TYPES, ENTITY_TYPES

    service = DefaultGraphService(backend=InMemoryGraphBackend())

    class CatalogGraph(_Graph):
        def catalog(self, request):
            return service.catalog(request)

    _common.set_runtime(_Host(CatalogGraph()))
    _common.set_json(False)
    result = CliRunner().invoke(
        graph.graph_app, ["catalog", "--profile", "full", "--format", format_]
    )
    assert result.exit_code == 0, result.output
    output = " ".join(_plain_cli_output(result.output).split())
    for spec in ENTITY_TYPES.values():
        if spec.public:
            assert spec.label in output
            assert " ".join(spec.identity_policy.split()) in output
            assert " ".join(spec.description.split()) in output
    for spec in EDGE_TYPES.values():
        if spec.public:
            assert spec.edge_type in output
            assert " ".join(spec.description.split()) in output
            for subject, object_ in spec.allowed_pairs:
                assert f"{subject} -> {object_}" in output


def test_read_catalog_stays_compact_and_points_writers_to_full_catalog() -> None:
    _common.set_runtime(_Host(_Graph()))
    _common.set_json(False)
    result = CliRunner().invoke(graph.graph_app, ["catalog", "--profile", "read"])
    assert result.exit_code == 0
    output = " ".join(_plain_cli_output(result.output).split())
    assert "graph catalog --profile full" in output
    assert "allowed subject -> object" not in output
