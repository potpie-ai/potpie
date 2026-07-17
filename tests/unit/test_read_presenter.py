"""Unit tests for graph read human presentation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from potpie.cli.read_presenter import (
    ReadPresentationContext,
    _escape_table_cell,
    _format_relations_summary,
    prepare_items,
    render_items_table,
    render_timeline_table,
)
from potpie_context_core.domain.ports.services.graph_service import GraphReadResult

pytestmark = pytest.mark.unit


def _timeline_result() -> GraphReadResult:
    return GraphReadResult(
        graph_contract_version="v1.5",
        ontology_version="2026-06-graph",
        view="recent_changes.timeline",
        subgraph="recent_changes",
        read_shape="entity_relations",
        quality={"status": "ok"},
        items=(
            {
                "entity_key": "activity:github:pr-2",
                "entity_type": "Activity",
                "score": 0.9,
                "summary": 'PR #2 "newer" was merged into acme/widgets.',
                "source_refs": ["github:pr:2"],
                "truth": "timeline_event",
                "relations": [
                    {
                        "predicate": "TOUCHED",
                        "from_key": "activity:github:pr-2",
                        "to_key": "repo:github.com/acme/widgets",
                        "fact": "PR #2 merged",
                        "source_refs": ["github:pr:2"],
                        "truth": "timeline_event",
                    },
                    {
                        "predicate": "PERFORMED",
                        "from_key": "person:bob",
                        "to_key": "activity:github:pr-2",
                        "fact": "PR #2 merged",
                        "source_refs": ["github:pr:2"],
                        "truth": "timeline_event",
                    },
                ],
            },
        ),
    )


def test_escape_table_cell_replaces_pipes_and_newlines() -> None:
    assert _escape_table_cell("a|b\nc") == "a\\|b c"


def test_format_relations_summary_includes_count_predicates_and_keys() -> None:
    item = {
        "relation_count": 2,
        "relation_predicates": ["PERFORMED", "TOUCHED"],
        "related_keys": ["repo:github.com/acme/widgets", "person:bob"],
    }
    summary = _format_relations_summary(item)
    assert summary.startswith("2 [")
    assert "TOUCHED" in summary
    assert "repo:github.com/acme/widgets" in summary


def test_prepare_items_applies_detail_and_relations_shaping() -> None:
    result = replace(
        _timeline_result(),
        detail="compact",
        relations="summary",
    )
    items = prepare_items(result)
    assert items[0]["relation_count"] == 2
    assert "relations" not in items[0]


def test_render_timeline_table_includes_headers_and_relations_column() -> None:
    result = _timeline_result()
    shaped = prepare_items(result)
    ctx = ReadPresentationContext(
        view="recent_changes.timeline",
        detail="compact",
        relations="summary",
        format_mode="table",
        sort="occurred_at",
        dedupe="source_ref",
        event_limit=5,
    )
    events = [
        {
            "activity_key": "activity:github:pr-2",
            "occurred_at": "2026-06-08",
            "source_refs": ["github:pr:2"],
            "fact": "PR #2 merged",
            "score": 0.9,
            "predicate": "TOUCHED",
        }
    ]
    output = render_timeline_table(result, events, shaped, ctx)
    assert "occurred_at | source_ref | activity | fact | score | relations" in output
    assert "TOUCHED" in output
    assert "scope=applied graph-read scope" in output
    assert "project-wide" not in output


def test_render_items_table_handles_empty_rows() -> None:
    ctx = ReadPresentationContext(
        view="decisions.preferences_for_scope",
        detail="compact",
        relations="summary",
        format_mode="table",
        sort="auto",
        dedupe="auto",
        event_limit=10,
    )
    output = render_items_table([], ctx)
    assert "score | type | entity_key | summary | relations" in output
    assert "(no rows)" in output
