"""Graph CLI inbox behavior."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_support import _assert_graph_envelope, _Graph, _Host
from tests.graph_cli_support import graph_cli_state as graph_cli_state
from tests.graph_cli_workbench_doubles import InboxRecorder, _inbox_result

pytestmark = pytest.mark.unit


def test_graph_inbox_add_returns_pending_item_envelope() -> None:
    _common.set_json(True)
    workbench = InboxRecorder(inbox_result=_inbox_result())
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "add",
            "--summary",
            "Possible graph update",
            "--evidence",
            "github:pr:955",
            "--source-ref",
            "github:pr:955",
            "--subgraph",
            "debugging",
            "--created-by",
            "codex",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.inbox.add")
    assert body["action"] == "add"
    assert body["item"]["item_id"] == "graph-inbox:test"
    assert body["item"]["status"] == "pending"
    assert workbench.inbox_calls == [
        (
            "add",
            {
                "pot_id": "p",
                "summary": "Possible graph update",
                "details": None,
                "evidence": ("github:pr:955",),
                "source_refs": ("github:pr:955",),
                "suspected_subgraphs": ("debugging",),
                "created_by": {"surface": "cli", "actor": "codex"},
            },
        )
    ]


def test_graph_inbox_list_passes_filters() -> None:
    _common.set_json(True)
    workbench = InboxRecorder(inbox_result=_inbox_result(action="list"))
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "list",
            "--status",
            "pending,claimed",
            "--claimed-by",
            "user:alice",
            "--subgraph",
            "debugging",
            "--source-ref",
            "github:pr:955",
            "--limit",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.inbox.list")
    assert body["item_count"] == 1
    assert workbench.inbox_calls[0] == (
        "list",
        {
            "pot_id": "p",
            "status": ("pending,claimed",),
            "claimed_by": "user:alice",
            "suspected_subgraph": "debugging",
            "source_ref": "github:pr:955",
            "since": None,
            "until": None,
            "limit": 10,
        },
    )


def test_graph_inbox_claim_passes_actor() -> None:
    _common.set_json(True)
    workbench = InboxRecorder(
        inbox_result=_inbox_result(action="claim", status="claimed")
    )
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["inbox", "claim", "graph-inbox:test", "--by", "user:alice"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.inbox.claim")
    assert body["item"]["status"] == "claimed"
    assert workbench.inbox_calls == [
        (
            "claim",
            {
                "pot_id": "p",
                "item_id": "graph-inbox:test",
                "claimed_by": "user:alice",
            },
        )
    ]


def test_graph_inbox_mark_applied_passes_plan_and_mutation() -> None:
    _common.set_json(True)
    workbench = InboxRecorder(
        inbox_result=_inbox_result(action="mark-applied", status="applied")
    )
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "mark-applied",
            "graph-inbox:test",
            "--plan",
            "mutation-plan:test",
            "--mutation",
            "mutation-1",
            "--by",
            "user:alice",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.inbox.mark-applied")
    assert body["item"]["status"] == "applied"
    assert workbench.inbox_calls == [
        (
            "mark-applied",
            {
                "pot_id": "p",
                "item_id": "graph-inbox:test",
                "closed_by": "user:alice",
                "linked_plan_id": "mutation-plan:test",
                "linked_mutation_id": "mutation-1",
            },
        )
    ]


def test_graph_inbox_mark_rejected_and_close_record_reasons() -> None:
    _common.set_json(True)
    workbench = InboxRecorder(
        inbox_result=_inbox_result(action="mark-rejected", status="rejected")
    )
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    rejected = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "mark-rejected",
            "graph-inbox:test",
            "--reason",
            "not enough evidence",
            "--by",
            "user:alice",
        ],
    )

    assert rejected.exit_code == 0, rejected.output
    emitted = json.loads(rejected.output)
    _assert_graph_envelope(emitted, "graph.inbox.mark-rejected")
    assert workbench.inbox_calls == [
        (
            "mark-rejected",
            {
                "pot_id": "p",
                "item_id": "graph-inbox:test",
                "closed_by": "user:alice",
                "rejection_reason": "not enough evidence",
            },
        )
    ]

    workbench = InboxRecorder(
        inbox_result=_inbox_result(action="close", status="closed")
    )
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))
    closed = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "close",
            "graph-inbox:test",
            "--reason",
            "superseded",
            "--by",
            "user:alice",
        ],
    )

    assert closed.exit_code == 0, closed.output
    emitted = json.loads(closed.output)
    _assert_graph_envelope(emitted, "graph.inbox.close")
    assert workbench.inbox_calls[0][1]["rejection_reason"] == "superseded"
