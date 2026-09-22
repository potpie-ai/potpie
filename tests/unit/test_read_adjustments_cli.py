"""Useful results with disclosed adjustments (agent-experience audit, batch 1).

Every case here was a retry, a silent reinterpretation, or a confident empty
answer before: ``--depth 100`` ran at 4 with no notice, ``--type repository``
returned zero rows, ``--format json`` failed after the graph work, and the
compact catalog advertised commands that refused on a missing selector.

The rules under test: a locally decidable bad input costs zero host calls; an
accepted exact alias costs exactly one execution; every adjustment is carried
as ``status="adjusted"`` + ``adjustments`` in JSON and as a ``~`` line in
text, once; and nothing here ever picks a near miss or moves the pot.
"""

from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone

import pytest
from potpie_context_core.cli_commands import selector_tokens
from potpie_context_core.graph_views import views_for_catalog
from potpie_context_core.graph_workbench_ontology import ontology_contract
from potpie_context_core.ports.graph_service import GraphCatalogResult
from test_graph_cli_contract import (
    _assert_graph_envelope,
    _Backend,
    _Graph,
    _Host,
    _non_timeline_env,
    _plain_cli_output,
    _timeline_env,
)
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_json_mode():
    yield
    _common.set_json(False)


class _CountingGraph(_Graph):
    """Records catalog requests so preflight tests can prove zero host calls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.catalog_requests = []

    def catalog(self, request):
        self.catalog_requests.append(request)
        return super().catalog(request)


def _read(graph_service, *args: str, as_json: bool = True):
    _common.set_json(as_json)
    _common.set_host(_Host(graph_service, backend=_Backend()))
    return CliRunner().invoke(graph.graph_app, ["read", *args])


def _neighborhood_args(*extra: str) -> list[str]:
    return [
        "--subgraph",
        "infra_topology",
        "--view",
        "service_neighborhood",
        "--scope",
        "service:payments-api",
        *extra,
    ]


def _adjustment_lines(output: str) -> list[str]:
    return [
        line for line in _plain_cli_output(output).splitlines() if line.startswith("~ ")
    ]


# --- AX10: numeric adjustments ---------------------------------------------


def test_depth_over_the_advertised_maximum_is_capped_once_and_disclosed() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(service, *_neighborhood_args("--depth", "100"))

    assert result.exit_code == 0, result.output
    assert service.read_request.depth == 4
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read")
    assert emitted["status"] == "adjusted"
    assert emitted["adjustments"] == [
        {
            "field": "depth",
            "requested": 100,
            "effective": 4,
            "reason": "maximum_supported",
            "message": "Depth 100 exceeds the supported maximum 4; returned depth-4 context.",
            "max_supported": 4,
        }
    ]
    # No rerun suggestion: the command already handled it.
    assert emitted["recommended_next_action"] is None


def test_depth_adjustment_is_one_text_line() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(service, *_neighborhood_args("--depth", "100"), as_json=False)

    assert result.exit_code == 0, result.output
    assert _adjustment_lines(result.output) == [
        "~ Depth 100 exceeds the supported maximum 4; returned depth-4 context."
    ]


def test_depth_within_bounds_carries_no_adjustment_keys() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(service, *_neighborhood_args("--depth", "3"))

    assert result.exit_code == 0, result.output
    assert service.read_request.depth == 3
    emitted = json.loads(result.output)
    assert "status" not in emitted
    assert "adjustments" not in emitted


@pytest.mark.parametrize("depth", ["0", "-1"])
def test_depth_below_one_is_refused_before_the_host_is_asked(depth: str) -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(service, *_neighborhood_args("--depth", depth))

    assert result.exit_code == _common.EXIT_VALIDATION
    assert service.read_called is False
    emitted = json.loads(result.output)
    assert emitted["error"]["code"] == "validation_error"
    assert "--depth must be >= 1 (this view walks 1..4)" in emitted["error"]["message"]


def test_depth_on_a_view_without_a_budget_is_left_to_the_service() -> None:
    # Not every view walks. The service reports ``depth`` as an unsupported
    # filter for those; the CLI must not invent a cap for them.
    service = _Graph(read_result=_timeline_env())
    result = _read(
        service,
        "--subgraph",
        "recent_changes",
        "--view",
        "timeline",
        "--depth",
        "50",
    )

    assert result.exit_code == 0, result.output
    assert service.read_request.depth == 50
    assert "adjustments" not in json.loads(result.output)


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_non_positive_limit_is_refused_before_the_host_is_asked(limit: str) -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(service, *_neighborhood_args("--limit", limit))

    assert result.exit_code == _common.EXIT_VALIDATION
    assert service.read_called is False
    assert "--limit must be >= 1" in json.loads(result.output)["error"]["message"]


# --- AX18: exact selector aliases -------------------------------------------


def test_case_variants_of_subgraph_and_view_execute_once() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(
        service,
        "--subgraph",
        "Decisions",
        "--view",
        "Preferences_For_Scope",
        "--scope",
        "service:payments-api",
    )

    assert result.exit_code == 0, result.output
    assert service.read_request.subgraph == "decisions"
    assert service.read_request.view == "preferences_for_scope"
    emitted = json.loads(result.output)
    assert [(a["field"], a["reason"]) for a in emitted["adjustments"]] == [
        ("subgraph", "canonical_case"),
        ("view", "canonical_case"),
    ]


def test_fully_qualified_view_with_agreeing_subgraph_executes_once() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(
        service,
        "--subgraph",
        "debugging",
        "--view",
        "debugging.prior_occurrences",
        "--query",
        "timeout",
    )

    assert result.exit_code == 0, result.output
    assert (service.read_request.subgraph, service.read_request.view) == (
        "debugging",
        "prior_occurrences",
    )
    adjustments = json.loads(result.output)["adjustments"]
    assert adjustments[0]["reason"] == "canonical_alias"
    assert adjustments[0]["requested"] == "debugging.prior_occurrences"


def test_fully_qualified_view_alone_supplies_the_subgraph() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(
        service, "--view", "debugging.prior_occurrences", "--query", "timeout"
    )

    assert result.exit_code == 0, result.output
    assert service.read_request.subgraph == "debugging"


def test_exact_include_alias_reads_the_canonical_view() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(
        service, "--subgraph", "knowledge", "--view", "docs", "--query", "potpie"
    )

    assert result.exit_code == 0, result.output
    assert service.read_request.view == "document_context"
    adjustment = json.loads(result.output)["adjustments"][0]
    assert adjustment["effective"] == "knowledge.document_context"
    assert adjustment["reason"] == "canonical_alias"


def test_near_miss_view_is_never_corrected() -> None:
    # The CLI passes a token it cannot exactly resolve through untouched so
    # the host's unknown-view error (with candidates) is what the caller
    # sees; it must not guess ``preferences_for_scope``.
    service = _Graph(read_result=_non_timeline_env())
    result = _read(
        service,
        "--subgraph",
        "decisions",
        "--view",
        "preferences_for_scop",
        "--scope",
        "service:x",
    )

    assert result.exit_code == 0  # the fake host answers; the real one refuses
    assert service.read_request.view == "preferences_for_scop"
    assert "adjustments" not in json.loads(result.output)


# --- AX21: presentation preflight ------------------------------------------


def test_format_json_emits_the_machine_envelope_without_the_root_flag() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(service, *_neighborhood_args("--format", "json"), as_json=False)

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read")
    assert emitted["status"] == "adjusted"
    assert emitted["adjustments"][0]["field"] == "format"
    assert emitted["adjustments"][0]["reason"] == "machine_json"
    assert emitted["adjustments"][0]["effective"] == "auto"


def test_detail_summary_alias_executes_once_as_compact() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(service, *_neighborhood_args("--detail", "summary"))

    assert result.exit_code == 0, result.output
    assert service.read_request.detail == "compact"
    adjustment = json.loads(result.output)["adjustments"][0]
    assert (adjustment["field"], adjustment["effective"]) == ("detail", "compact")


@pytest.mark.parametrize(
    ("flag", "value", "fragment"),
    [
        ("--format", "yaml", "--format must be one of"),
        ("--detail", "verbose", "--detail must be one of"),
        ("--relations", "all", "--relations must be one of"),
        ("--sort", "newest", "--sort must be one of"),
        ("--dedupe", "smart", "--dedupe must be one of"),
    ],
)
def test_unknown_presentation_values_cost_zero_host_calls(
    flag: str, value: str, fragment: str
) -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(service, *_neighborhood_args(flag, value))

    assert result.exit_code == _common.EXIT_VALIDATION
    assert service.read_called is False
    assert fragment in json.loads(result.output)["error"]["message"]


def test_neighborhood_detail_compact_is_the_summary_alias() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph(), backend=_Backend()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["neighborhood", "--entity", "service:web", "--detail", "compact"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.neighborhood")
    assert body["detail"] == "summary"
    assert emitted["status"] == "adjusted"
    assert emitted["adjustments"][0]["field"] == "detail"


# --- AX20: explicit time parsing -------------------------------------------


def _timeline(*extra: str) -> list[str]:
    return ["--subgraph", "recent_changes", "--view", "timeline", *extra]


def test_explicit_since_wins_and_the_unused_window_is_disclosed() -> None:
    service = _Graph(read_result=_timeline_env())
    result = _read(service, *_timeline("--since", "2026-09-01", "--time-window", "1h"))

    assert result.exit_code == 0, result.output
    assert service.read_request.since == datetime(2026, 9, 1, tzinfo=timezone.utc)
    adjustments = json.loads(result.output)["adjustments"]
    assert len(adjustments) == 1
    assert adjustments[0]["field"] == "time_window"
    assert adjustments[0]["reason"] == "explicit_since"
    assert adjustments[0]["requested"] == "1h"
    assert "2026-09-01T00:00:00+00:00" in adjustments[0]["message"]


def test_reversed_bounds_are_refused_before_the_host_is_asked() -> None:
    service = _Graph(read_result=_timeline_env())
    result = _read(
        service, *_timeline("--since", "2026-09-22", "--until", "2026-09-01")
    )

    assert result.exit_code == _common.EXIT_VALIDATION
    assert service.read_called is False
    message = json.loads(result.output)["error"]["message"]
    assert "2026-09-22T00:00:00+00:00" in message
    assert "2026-09-01T00:00:00+00:00" in message
    assert "not reordered" in message


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("7days", "7d"), ("2 weeks", "2w"), ("24hours", "24h"), ("30minutes", "30m")],
)
def test_spelled_out_units_match_their_canonical_window(
    alias: str, canonical: str
) -> None:
    # A fixed --until pins the window end, so both spellings are compared on
    # identical bounds without a clock.
    canonical_service = _Graph(read_result=_timeline_env())
    _read(
        canonical_service,
        *_timeline("--time-window", canonical, "--until", "2026-09-22T00:00:00Z"),
    )
    alias_service = _Graph(read_result=_timeline_env())
    result = _read(
        alias_service,
        *_timeline("--time-window", alias, "--until", "2026-09-22T00:00:00Z"),
    )

    assert result.exit_code == 0, result.output
    assert alias_service.read_request.since == canonical_service.read_request.since
    assert alias_service.read_request.until == canonical_service.read_request.until
    adjustment = json.loads(result.output)["adjustments"][0]
    assert (adjustment["field"], adjustment["effective"], adjustment["reason"]) == (
        "time_window",
        canonical,
        "unit_alias",
    )


def test_unknown_duration_unit_stays_a_refusal() -> None:
    service = _Graph(read_result=_timeline_env())
    result = _read(service, *_timeline("--time-window", "2fortnights"))

    assert result.exit_code == _common.EXIT_VALIDATION
    assert service.read_called is False


def test_timeline_recent_carries_adjustments_in_its_payload() -> None:
    _common.set_json(True)
    service = _Graph(read_result=_timeline_env())
    _common.set_host(_Host(service))

    result = CliRunner().invoke(
        graph.timeline_app,
        ["--time-window", "2weeks", "--until", "2026-09-22T00:00:00Z", "--limit", "2"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    assert emitted["status"] == "adjusted"
    assert emitted["adjustments"][0]["effective"] == "2w"
    assert emitted["event_count"] == 2


def test_jsonl_keeps_stdout_rows_only_and_notices_aside() -> None:
    service = _Graph(read_result=_timeline_env())
    result = _read(
        service,
        *_timeline(
            "--format",
            "jsonl",
            "--time-window",
            "7days",
            "--until",
            "2026-09-22T00:00:00Z",
        ),
        as_json=False,
    )

    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines if line.startswith("{")]
    assert rows, result.output
    assert any(line.startswith("~ ") for line in lines)


# --- AX19: canonical ontology filters ------------------------------------------


def _search(service, *args: str, as_json: bool = True):
    _common.set_json(as_json)
    _common.set_host(_Host(service))
    return CliRunner().invoke(graph.graph_app, ["search-entities", *args])


def test_case_variant_type_and_predicate_execute_once_as_canonical() -> None:
    service = _CountingGraph()
    result = _search(
        service,
        "potpie",
        "--type",
        "repository",
        "--predicate",
        "policy-applies-to",
        "--subgraph",
        "Decisions",
    )

    assert result.exit_code == 0, result.output
    assert service.search_request.type == "Repository"
    assert service.search_request.predicate == "POLICY_APPLIES_TO"
    assert service.search_request.subgraph == "decisions"
    # Known locally: the catalog is not consulted for an exact variant.
    assert service.catalog_requests == []
    emitted = json.loads(result.output)
    assert [a["field"] for a in emitted["adjustments"]] == [
        "type",
        "predicate",
        "subgraph",
    ]
    assert all(a["reason"] == "canonical_case" for a in emitted["adjustments"])


def test_case_variant_filters_are_one_text_line_each() -> None:
    service = _Graph()
    result = _search(service, "potpie", "--type", "repository", as_json=False)

    assert result.exit_code == 0, result.output
    assert _adjustment_lines(result.output) == [
        "~ --type 'repository' read as 'Repository'"
    ]


def test_unknown_type_is_refused_before_retrieval_with_candidates() -> None:
    service = _CountingGraph()
    result = _search(service, "potpie", "--type", "Repositry")

    assert result.exit_code == _common.EXIT_VALIDATION
    assert service.search_request is None
    # The host's advertised vocabulary was consulted once before refusing,
    # so a valid server extension would have been accepted.
    assert len(service.catalog_requests) == 1
    emitted = json.loads(result.output)
    assert emitted["error"]["code"] == "unsupported_filter"
    detail = emitted["error"]["detail"]
    assert detail["argument"] == "--type"
    assert detail["candidates"][0] == "Repository"
    assert len(detail["candidates"]) <= 6
    assert "Repository" in emitted["recommended_next_action"]


def test_unknown_predicate_is_refused_before_retrieval() -> None:
    service = _Graph()
    result = _search(service, "potpie", "--predicate", "POLICY_APPLIES")

    assert result.exit_code == _common.EXIT_VALIDATION
    assert service.search_request is None
    assert (
        "POLICY_APPLIES_TO"
        in json.loads(result.output)["error"]["detail"]["candidates"]
    )


def test_server_extension_type_is_accepted_from_the_catalog() -> None:
    class _ExtendedGraph(_Graph):
        def catalog(self, request):
            base = super().catalog(request)
            return GraphCatalogResult(
                **{
                    **{
                        key: getattr(base, key)
                        for key in (
                            "graph_contract_version",
                            "ontology_version",
                            "commands",
                            "truth_classes",
                            "mutation_operations",
                            "review_required_operations",
                            "deferred_operations",
                            "views",
                            "predicates",
                            "match_mode",
                        )
                    },
                    "entity_types": ({"label": "LoadBalancer"},),
                }
            )

    service = _ExtendedGraph()
    result = _search(service, "edge", "--type", "loadbalancer")

    assert result.exit_code == 0, result.output
    assert service.search_request.type == "LoadBalancer"


def test_search_entities_non_positive_limit_is_refused() -> None:
    service = _Graph()
    result = _search(service, "potpie", "--limit", "0")

    assert result.exit_code == _common.EXIT_VALIDATION
    assert service.search_request is None


# --- AX05 / AX07: executable, pot-scoped discovery -------------------------


def test_compact_catalog_views_carry_requirements_and_pot_scoped_templates() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--profile", "read"])

    assert result.exit_code == 0, result.output
    body = _assert_graph_envelope(json.loads(result.output), "graph.catalog")
    assert body["views"]
    for view in body["views"]:
        contract = ontology_contract().view(view["name"])
        assert contract is not None, view["name"]
        assert view["required_any_scope"] == list(contract.required_any_scope)
        assert view["required_scope"] == list(contract.required_scope)
        assert view["supported_filters"] == list(contract.supported_filters)
        tokens = shlex.split(view["next_read"])
        assert tokens[-2:] == ["--pot", "p"], view["next_read"]
        needs_selector = bool(contract.required_any_scope or contract.required_scope)
        assert view["next_read_is_template"] is needs_selector, view["next_read"]
        if contract.required_any_scope:
            flag, value = selector_tokens(contract.required_any_scope)
            assert flag in tokens and value in tokens, view["next_read"]
            assert "<" in value
        else:
            assert "<" not in view["next_read"]


def test_compact_catalog_text_shows_requirements() -> None:
    _common.set_json(False)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--profile", "read"])

    assert result.exit_code == 0, result.output
    output = _plain_cli_output(result.output)
    assert "view | backed | requires | filters" in output
    assert "one of query, service, repo" in output


def test_catalog_invalid_profile_costs_zero_host_calls() -> None:
    _common.set_json(True)
    service = _CountingGraph()
    _common.set_host(_Host(service))

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--profile", "nope"])

    assert result.exit_code == _common.EXIT_VALIDATION
    assert service.catalog_requests == []


def test_catalog_format_json_emits_the_machine_envelope() -> None:
    _common.set_json(False)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--format", "json"])

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.catalog")
    assert emitted["status"] == "adjusted"


def test_catalog_subgraph_case_is_canonicalized_before_the_host_call() -> None:
    _common.set_json(True)
    service = _CountingGraph()
    _common.set_host(_Host(service))

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--subgraph", "Debugging"])

    assert result.exit_code == 0, result.output
    assert service.catalog_requests[0].subgraph == "debugging"
    assert json.loads(result.output)["adjustments"][0]["field"] == "subgraph"


def test_describe_examples_render_in_text_and_name_the_requirements() -> None:
    _common.set_json(False)
    _common.set_host(_Host(_Graph()))

    without = CliRunner().invoke(
        graph.graph_app, ["describe", "debugging", "--view", "prior_occurrences"]
    )
    with_examples = CliRunner().invoke(
        graph.graph_app,
        ["describe", "debugging", "--view", "prior_occurrences", "--examples"],
    )

    assert without.exit_code == 0 and with_examples.exit_code == 0
    plain_without = _plain_cli_output(without.output)
    plain_with = _plain_cli_output(with_examples.output)
    assert "requires: one of query, service, repo" in plain_without
    assert "examples (templates" not in plain_without
    assert "examples (templates" in plain_with
    assert (
        "potpie graph read --subgraph debugging --view prior_occurrences" in plain_with
    )
    assert "--pot p" in plain_with


def test_describe_case_variants_execute_once() -> None:
    _common.set_json(True)
    service = _Graph()
    _common.set_host(_Host(service))

    result = CliRunner().invoke(
        graph.graph_app, ["describe", "Debugging", "--view", "Prior_Occurrences"]
    )

    assert result.exit_code == 0, result.output
    assert service.describe_request.subgraph == "debugging"
    assert service.describe_request.view == "prior_occurrences"
    assert json.loads(result.output)["status"] == "adjusted"


def test_describe_subgraph_alone_is_never_redirected_to_a_view() -> None:
    _common.set_json(True)
    service = _Graph()
    _common.set_host(_Host(service))

    result = CliRunner().invoke(graph.graph_app, ["describe", "Debugging"])

    assert result.exit_code == 0, result.output
    assert service.describe_request.subgraph == "debugging"
    assert service.describe_request.view is None


# --- AX14: one truthful mutation-template route ----------------------------


def test_mutation_template_no_longer_warns_toward_a_missing_surface() -> None:
    _common.set_json(True)
    result = CliRunner().invoke(
        graph.graph_app, ["mutation-template", "--kind", "repo-baseline"]
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    assert emitted["warnings"] == []
    assert "mutation examples" not in emitted["recommended_next_action"]
    assert "graph propose" in emitted["recommended_next_action"]


# --- AX17: one contract for combined adjustments ----------------------------


def test_combined_adjustments_are_ordered_and_deduplicated() -> None:
    service = _Graph(read_result=_non_timeline_env())
    result = _read(
        service,
        "--subgraph",
        "Infra_Topology",
        "--view",
        "service_neighborhood",
        "--scope",
        "service:payments-api",
        "--depth",
        "9",
        "--detail",
        "summary",
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    assert [a["field"] for a in emitted["adjustments"]] == [
        "detail",
        "subgraph",
        "depth",
    ]
    assert service.read_request.depth == 4
    assert service.read_request.subgraph == "infra_topology"
    assert service.read_request.detail == "compact"


def test_text_and_json_report_the_same_adjustments() -> None:
    args = _neighborhood_args("--depth", "9", "--detail", "summary")
    as_json = _read(_Graph(read_result=_non_timeline_env()), *args)
    as_text = _read(_Graph(read_result=_non_timeline_env()), *args, as_json=False)

    messages = [a["message"] for a in json.loads(as_json.output)["adjustments"]]
    assert _adjustment_lines(as_text.output) == [f"~ {message}" for message in messages]
    assert len(messages) == 2


def test_catalog_read_profile_templates_validate_against_selector_rules() -> None:
    # Every advertised view either runs unchanged (no selector needed) or is
    # visibly a template naming one accepted selector.
    for entry in views_for_catalog():
        view = graph._compact_catalog_view(entry, pot_id="pot_x")
        contract = ontology_contract().view(entry["name"])
        assert contract is not None
        if contract.required_any_scope:
            assert view["next_read_is_template"] is True
        tokens = shlex.split(view["next_read"])
        assert tokens[:3] == ["potpie", "graph", "read"]
        assert tokens[-2:] == ["--pot", "pot_x"]
