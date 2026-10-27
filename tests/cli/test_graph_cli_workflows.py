"""Graph CLI workflows persist plans and inbox changes through a real workbench."""

import json

import pytest
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from typer.testing import CliRunner

from potpie.cli.commands import graph
from tests.graph_cli_support import (
    _assert_graph_envelope,
    _bulk_mutation_payload,
    _valid_mutation_payload,
)
from tests.graph_workbench_support import (
    _graph_workbench_runtime as _graph_workbench_runtime,
)

pytestmark = pytest.mark.unit


def _run(*args: str) -> dict:
    result = CliRunner().invoke(graph.graph_app, list(args))
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.output)
    assert envelope["ok"] is True, envelope
    return envelope


def test_propose_commit_and_history_share_the_persisted_plan(
    graph_workbench_runtime, tmp_path
) -> None:
    payload = _valid_mutation_payload()
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(payload), encoding="utf-8")

    proposal = _assert_graph_envelope(
        _run("propose", "--file", str(payload_file), "--ttl", "30m"),
        "graph.propose",
    )
    plan_id = proposal["plan_id"]
    stored = graph_workbench_runtime.plans.get(pot_id="p", plan_id=plan_id)
    assert stored is not None
    assert stored.status == "validated"
    assert stored.original_payload["operations"] == payload["operations"]
    assert (stored.expires_at - stored.created_at).total_seconds() == 1800
    assert (
        graph_workbench_runtime.backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id="p")
        )
        == []
    )

    committed = _assert_graph_envelope(_run("commit", plan_id), "graph.commit")
    assert committed["status"] == "committed"
    stored = graph_workbench_runtime.plans.get(pot_id="p", plan_id=plan_id)
    assert stored.status == "committed"
    assert stored.mutation_id == committed["mutation_id"]
    assert (
        len(
            graph_workbench_runtime.backend.claim_query.find_claims(
                ClaimQueryFilter(pot_id="p")
            )
        )
        == 1
    )

    history = _assert_graph_envelope(
        _run("history", "--plan", plan_id), "graph.history"
    )
    entry = next(row for row in history["entries"] if row["plan_id"] == plan_id)
    assert entry["status"] == "committed"
    assert entry["mutation_id"] == committed["mutation_id"]


@pytest.mark.parametrize("dry_run", [False, True])
def test_bulk_chunks_persist_their_plans_and_only_apply_when_requested(
    graph_workbench_runtime, tmp_path, dry_run
) -> None:
    payload_file = tmp_path / "bulk.json"
    payload_file.write_text(json.dumps(_bulk_mutation_payload(3)), encoding="utf-8")
    flags = ["--dry-run"] if dry_run else []

    body = _assert_graph_envelope(
        _run("bulk", "apply", "--file", str(payload_file), "--chunk-size", "2", *flags),
        "graph.bulk.apply",
    )

    assert body["status"] == ("validated" if dry_run else "committed")
    assert body["chunks_total"] == body["chunks_attempted"] == 2
    plans = graph_workbench_runtime.plans.list(pot_id="p")
    assert len(plans) == 2
    assert {record.status for record in plans} == {
        "validated" if dry_run else "committed"
    }
    assert {record.original_payload["idempotency_key"] for record in plans} == {
        "bulk:test:chunk-0001",
        "bulk:test:chunk-0002",
    }
    assert sorted(len(record.original_payload["operations"]) for record in plans) == [
        1,
        2,
    ]
    claims = graph_workbench_runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p")
    )
    assert len(claims) == (0 if dry_run else 3)
    assert body["chunks_committed"] == (0 if dry_run else 2)


def test_inbox_claim_and_apply_persist_the_same_item(graph_workbench_runtime) -> None:
    added = _run(
        "inbox",
        "add",
        "--summary",
        "Possible graph update",
        "--source-ref",
        "github:pr:955",
        "--subgraph",
        "debugging",
        "--created-by",
        "user:alice",
    )["result"]["item"]
    item_id = added["item_id"]
    assert (
        graph_workbench_runtime.inbox.get(pot_id="p", item_id=item_id).status
        == "pending"
    )

    claimed = _run("inbox", "claim", item_id, "--by", "user:alice")["result"]["item"]
    assert claimed["status"] == "claimed"
    listing = _run(
        "inbox",
        "list",
        "--status",
        "claimed",
        "--claimed-by",
        "user:alice",
        "--subgraph",
        "debugging",
        "--source-ref",
        "github:pr:955",
    )["result"]
    assert [item["item_id"] for item in listing["items"]] == [item_id]

    _run(
        "inbox",
        "mark-applied",
        item_id,
        "--plan",
        "mutation-plan:applied",
        "--mutation",
        "mutation-applied",
        "--by",
        "user:alice",
    )
    stored = graph_workbench_runtime.inbox.get(pot_id="p", item_id=item_id)
    assert stored.status == "applied"
    assert stored.claimed_by == stored.closed_by == "user:alice"
    assert stored.linked_plan_id == "mutation-plan:applied"
    assert stored.linked_mutation_id == "mutation-applied"
    assert _run("inbox", "list", "--status", "pending")["result"]["items"] == []


@pytest.mark.parametrize(
    ("command", "status", "reason"),
    [
        ("mark-rejected", "rejected", "not enough evidence"),
        ("close", "closed", "superseded"),
    ],
)
def test_inbox_terminal_transitions_persist_the_actor_and_reason(
    graph_workbench_runtime, command, status, reason
) -> None:
    added = _run("inbox", "add", "--summary", "Review this update")["result"]["item"]
    item_id = added["item_id"]

    _run("inbox", command, item_id, "--reason", reason, "--by", "user:alice")

    stored = graph_workbench_runtime.inbox.get(pot_id="p", item_id=item_id)
    assert stored.status == status
    assert stored.closed_by == "user:alice"
    assert stored.rejection_reason == reason
    shown = _run("inbox", "show", item_id)["result"]["item"]
    assert shown["status"] == status
    assert shown["rejection_reason"] == reason
