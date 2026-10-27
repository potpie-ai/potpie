"""Graph CLI read validation, rendering, and timeline/nudge aliases."""

from __future__ import annotations

import json

import pytest
from potpie_context_engine.core.ports.graph_service import (
    GraphReadResult,
)
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_support import (
    _assert_graph_envelope,
    _Graph,
    _Host,
    _non_timeline_env,
    _Nudge,
    _plain_cli_output,
    _timeline_env,
    _unsupported_query_read_result,
)

pytestmark = pytest.mark.unit


def _invoke_timeline_read_text(*args: str) -> str:
    _common.set_json(False)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_runtime(_Host(graph_service))
    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "recent_changes", "--view", "timeline", *args],
    )
    assert result.exit_code == 0
    return _plain_cli_output(result.output)


@pytest.mark.parametrize("scope", ["service", "service:"])
def test_graph_read_rejects_malformed_scope_before_service_call(scope: str) -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "debugging",
            "--view",
            "prior_occurrences",
            "--scope",
            scope,
        ],
    )

    assert result.exit_code == 1
    assert graph_service.read_called is False
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "invalid --scope entry" in emitted["error"]["message"]


def test_graph_read_unknown_view_uses_error_envelope() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_error=ValueError("unknown graph view 'missing.view'"))
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "missing", "--view", "view"],
    )

    assert result.exit_code == 1
    assert graph_service.read_called is True
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "unknown graph view" in emitted["error"]["message"]


def test_graph_read_missing_required_scope_result_is_error_envelope() -> None:
    _common.set_json(True)
    graph_service = _Graph(
        read_result=GraphReadResult(
            graph_contract_version="v1.5",
            ontology_version="2026-06-graph",
            view="features.feature_context",
            subgraph="features",
            ok=False,
            status="missing_required_scope",
            message=(
                "graph read view 'features.feature_context' requires one of "
                "scope, service, repo, anchor_entity_key, query"
            ),
            coverage=(
                {
                    "view": "features.feature_context",
                    "status": "unsupported",
                    "candidate_pool": 0,
                },
            ),
            quality={"status": "unsupported", "reason": "missing_required_scope"},
            unsupported=(
                {
                    "name": "features.feature_context",
                    "reason": "missing_required_scope",
                },
            ),
        )
    )
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "features", "--view", "feature_context"],
    )

    assert result.exit_code == 1
    assert graph_service.read_called is True
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "missing_required_scope"
    assert emitted["unsupported"][0]["reason"] == "missing_required_scope"
    assert emitted["error"]["detail"]["quality"]["reason"] == "missing_required_scope"


def test_graph_read_include_guess_error_carries_did_you_mean() -> None:
    # Audit item 17: a failed include-family guess returns machine-readable
    # migration guidance in the error envelope (never accepted as input).
    from potpie_context_engine.core.graph_views import (
        UnknownGraphViewError,
        include_guess_guidance,
    )

    _common.set_json(True)
    guidance = include_guess_guidance("docs", "relevant")
    graph_service = _Graph(
        read_error=UnknownGraphViewError(
            "unknown graph view 'docs.relevant'",
            did_you_mean=guidance,
            recommended_next_action=guidance["read_command"],
        )
    )
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "docs", "--view", "relevant"],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    did_you_mean = emitted["error"]["detail"]["did_you_mean"]
    assert did_you_mean["view"] == "knowledge.document_context"
    assert did_you_mean["matched_include"] == "docs"
    assert emitted["recommended_next_action"] == (
        "potpie graph read --subgraph knowledge --view document_context"
    )


def test_graph_read_rejects_conflicting_fully_qualified_view_before_service_call() -> (
    None
):
    # A qualified --view that names a *different* subgraph than --subgraph is
    # a conflict: both targets are named and nothing is read. (An agreeing or
    # absent --subgraph is an exact alias and executes once; see
    # test_read_adjustments_cli.py.)
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "decisions", "--view", "debugging.prior_occurrences"],
    )

    assert result.exit_code == 1
    assert graph_service.read_called is False
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "'debugging'" in emitted["error"]["message"]
    assert "'decisions'" in emitted["error"]["message"]
    assert emitted["recommended_next_action"] == (
        "potpie graph read --subgraph debugging --view prior_occurrences"
    )


def test_graph_read_timeline_defaults_to_deduped_event_json() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "recent_changes",
            "--view",
            "timeline",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.read_request.limit == 40
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.read")
    assert "subgraph_versions" not in body
    assert "unsupported" not in body
    assert body["read_shape"] == "events"
    assert body["event_count"] == 1
    assert body["events"][0]["source_refs"] == ["github:pr:2"]
    assert body["freshness"]["local_worktree_included"] is False
    assert "items" not in body


def test_graph_read_threads_source_ref_filter() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "recent_changes",
            "--view",
            "timeline",
            "--source-ref",
            "github:pr:2",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.read_request.source_refs == ("github:pr:2",)


def test_graph_read_raw_json_defaults_to_compact_relations() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "recent_changes",
            "--view",
            "timeline",
            "--format",
            "raw",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.read_request.detail == "compact"
    assert graph_service.read_request.relations == "summary"
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.read")
    assert body["detail"] == "compact"
    assert body["relations_detail"] == "summary"
    assert "relations" not in body["items"][0]
    assert body["items"][0]["relation_count"] == 2


def test_graph_read_full_detail_preserves_relation_payload() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "recent_changes",
            "--view",
            "timeline",
            "--format",
            "raw",
            "--detail",
            "full",
            "--relations",
            "full",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.read_request.detail == "full"
    assert graph_service.read_request.relations == "full"
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.read")
    assert body["detail"] == "full"
    assert body["items"][0]["relations"][0]["predicate"] == "TOUCHED"


def test_timeline_recent_passes_project_scope_and_time_window() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.timeline_app,
        ["--time-window", "7d", "--limit", "2"],
    )

    assert result.exit_code == 0
    req = graph_service.read_request
    assert req.subgraph == "recent_changes"
    assert req.view == "timeline"
    assert req.scope == {}
    assert req.limit == 40
    assert req.since is not None
    emitted = json.loads(result.output)
    assert emitted["event_count"] == 2


def test_graph_read_table_format_renders_pipe_table() -> None:
    output = _invoke_timeline_read_text("--format", "table", "--limit", "2")
    assert "occurred_at |" in output
    assert "--- | ---" in output
    data_lines = [
        line
        for line in output.splitlines()
        if line.strip()
        and not line.startswith("view=")
        and not line.startswith("scope=")
        and not line.startswith("occurred_at |")
        and not line.startswith("--- |")
    ]
    assert data_lines
    assert not any(line.lstrip().startswith("•") for line in data_lines)


def test_graph_read_events_format_uses_bullets_not_table() -> None:
    output = _invoke_timeline_read_text("--format", "events", "--limit", "2")
    assert "  • " in output
    assert "--- | ---" not in output


def test_graph_read_table_vs_events_produce_different_output() -> None:
    events_output = _invoke_timeline_read_text("--format", "events", "--limit", "2")
    table_output = _invoke_timeline_read_text("--format", "table", "--limit", "2")
    assert events_output != table_output


def test_graph_read_text_detail_full_shows_claim_or_breakdown() -> None:
    compact_output = _invoke_timeline_read_text(
        "--format", "events", "--detail", "compact", "--limit", "1"
    )
    full_output = _invoke_timeline_read_text(
        "--format", "events", "--detail", "full", "--limit", "1"
    )
    assert "truth=" in full_output
    assert "truth=" not in compact_output


def test_graph_read_text_relations_summary_shows_counts() -> None:
    output = _invoke_timeline_read_text(
        "--format", "events", "--relations", "summary", "--limit", "1"
    )
    assert "relations:" in output
    assert "TOUCHED" in output


def test_graph_read_text_relations_full_lists_edges() -> None:
    output = _invoke_timeline_read_text(
        "--format",
        "events",
        "--relations",
        "full",
        "--detail",
        "full",
        "--limit",
        "1",
    )
    assert "↳ TOUCHED" in output
    assert "↳ PERFORMED" in output


def test_timeline_recent_table_format() -> None:
    _common.set_json(False)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.timeline_app,
        ["--format", "table", "--limit", "2"],
    )

    assert result.exit_code == 0
    output = _plain_cli_output(result.output)
    assert "occurred_at |" in output
    assert "--- | ---" in output


def test_graph_read_unsupported_filter_result_is_error_envelope() -> None:
    # A view that cannot apply a filter refuses the read; an exit-0 empty page
    # would read as a genuine empty answer.
    _common.set_json(True)
    graph_service = _Graph(read_result=_unsupported_query_read_result())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "admin",
            "--view",
            "inspection_slice",
            "--query",
            "QME",
            "--limit",
            "20",
            "--detail",
            "full",
        ],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "unsupported_filter"
    assert "does not support filter query" in emitted["error"]["message"]
    assert emitted["unsupported"][0]["name"] == "query"
    assert emitted["unsupported"][0]["detail"]["supported_filters"] == ["source_ref"]
    assert emitted["error"]["detail"]["quality"]["reason"] == "unsupported_filter"


def test_graph_read_unsupported_filter_human_output_names_filter_and_exits() -> None:
    _common.set_json(False)
    graph_service = _Graph(read_result=_unsupported_query_read_result())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "admin",
            "--view",
            "inspection_slice",
            "--query",
            "QME",
        ],
    )

    assert result.exit_code == 1
    assert "does not support filter query" in result.output
    assert "supported filters: source_ref" in result.output
    # Never dressed up as an ordinary empty page.
    assert "(no rows)" not in result.output


@pytest.mark.parametrize("format_", ["table", "events"])
def test_graph_read_json_keeps_items_for_non_timeline_formats(format_: str) -> None:
    # Only the timeline has an event projection; every other view keeps its
    # items in the machine body whatever the human layout.
    _common.set_json(True)
    graph_service = _Graph(read_result=_non_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "infra_topology",
            "--view",
            "service_neighborhood",
            "--scope",
            "service:payments-api",
            "--format",
            format_,
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.read")
    assert body["items"], f"--format {format_} --json returned no data"
    assert body["items"][0]["entity_key"] == "service:payments-api"


def test_graph_read_rejects_an_unknown_format() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_result=_non_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "infra_topology",
            "--view",
            "service_neighborhood",
            "--format",
            "csv",
        ],
    )

    assert result.exit_code == _common.EXIT_VALIDATION
    emitted = json.loads(result.output)
    assert emitted["error"]["code"] == "validation_error"
    assert "--format must be one of" in emitted["error"]["message"]
    assert graph_service.read_called is False


def test_graph_nudge_accepts_dash_event_alias() -> None:
    _common.set_json(True)
    nudge_service = _Nudge()
    _common.set_runtime(_Host(_Graph(), nudge_service=nudge_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["nudge", "--event", "pre-edit", "--session", "sess-1", "--path", "src/app.py"],
    )

    assert result.exit_code == 0
    assert nudge_service.request.event == "pre-edit"
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.nudge")
    assert body["event"] == "pre_edit"
    # Harnesses call this command directly; it does not steer them elsewhere.
    assert not any("legacy" in warning for warning in emitted["warnings"])


from tests.graph_cli_support import graph_cli_state as graph_cli_state
