"""Graph CLI proposal, commit, bulk ingestion and history contracts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest
from potpie_context_engine.core.graph_history import (
    GraphHistoryEntry,
    GraphHistoryResult,
)
from potpie_context_engine.core.graph_plans import (
    GraphIngestionVerificationResult,
)
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_results import _commit_result
from tests.graph_cli_support import (
    _assert_graph_envelope,
    _Graph,
    _Host,
    _plain_cli_output,
)
from tests.graph_cli_workbench_doubles import (
    CommitRecorder,
    ProposalRecorder,
    _proposal,
)

pytestmark = pytest.mark.unit


def test_graph_mutate_rejection_emits_result_and_exits_nonzero(tmp_path) -> None:
    _common.set_json(True)
    proposal = _proposal(
        ok=False,
        status="invalid",
        issues=(
            {
                "code": "invalid_endpoints",
                "message": "invalid endpoint pair",
                "severity": "error",
                "op_index": 0,
            },
        ),
    )
    workbench = ProposalRecorder(proposal=proposal)
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(_valid_mutation_payload()), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        ["mutate", "--file", str(payload_file)],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    detail = _assert_graph_envelope(emitted, "graph.mutate", ok=False)
    assert detail == {}
    assert emitted["error"]["code"] == "invalid_endpoints"
    assert emitted["error"]["detail"]["status"] == "invalid"
    assert emitted["error"]["detail"]["issues"][0]["code"] == "invalid_endpoints"
    assert "legacy transition command" in emitted["warnings"][0]
    assert workbench.commit_calls == []


def test_graph_propose_returns_persisted_plan_envelope(tmp_path) -> None:
    _common.set_json(True)
    workbench = ProposalRecorder(proposal=_proposal())
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(_valid_mutation_payload()), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        ["propose", "--file", str(payload_file), "--ttl", "30m"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.propose")
    assert body["plan_id"] == "mutation-plan:test"
    assert body["status"] == "validated"
    assert workbench.propose_calls[0][1:] == ("p", 1800)


def test_graph_propose_passes_approved_by_through(tmp_path) -> None:
    _common.set_json(True)
    workbench = ProposalRecorder(proposal=_proposal(status="validated", risk="medium"))
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(_valid_mutation_payload()), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        ["propose", "--file", str(payload_file), "--approved-by", "user:alice"],
    )

    assert result.exit_code == 0, result.output
    assert workbench.propose_approvals == ["user:alice"]


def test_graph_propose_without_the_flag_sends_no_approval(tmp_path) -> None:
    _common.set_json(True)
    workbench = ProposalRecorder(proposal=_proposal())
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(_valid_mutation_payload()), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app, ["propose", "--file", str(payload_file)]
    )

    assert result.exit_code == 0, result.output
    assert workbench.propose_approvals == [None]


def test_graph_propose_help_names_pre_approval() -> None:
    result = CliRunner().invoke(graph.graph_app, ["propose", "--help"])

    assert result.exit_code == 0
    text = " ".join(re.sub(r"[│╭╮╰╯─]", " ", _plain_cli_output(result.output)).split())
    assert "--approved-by" in text
    assert "pre-approves" in text


def test_graph_commit_applies_plan_id_only() -> None:
    _common.set_json(True)
    workbench = CommitRecorder(commit_result=_commit_result())
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.commit")
    assert body["status"] == "committed"
    assert body["mutation_id"] == "mutation-1"
    assert workbench.commit_calls == [("mutation-plan:test", "p", None, False)]


def test_graph_commit_verify_passes_hard_gate_flag() -> None:
    _common.set_json(True)
    workbench = CommitRecorder(commit_result=_commit_result())
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test", "--verify"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.commit")
    assert body["status"] == "committed"
    assert body["verification"]["status"] == "ok"
    assert workbench.commit_calls == [("mutation-plan:test", "p", None, True)]
    # The receipt comes back before the readback, which runs as its own read.
    assert workbench.deferred_verification == [True]
    assert workbench.verify_calls == [("mutation-plan:test", "p")]


def test_graph_commit_verify_exits_nonzero_when_gate_fails() -> None:
    _common.set_json(True)
    verification = GraphIngestionVerificationResult(
        ok=False,
        status="degraded",
        plan_id="mutation-plan:test",
        pot_id="p",
        claim_keys=("claim:test",),
        missing_claim_keys=("claim:test",),
        quality_status="ok",
        detail="committed plan did not read back all expected claim keys",
        recommended_next_action="Inspect graph history for the plan.",
    )
    workbench = CommitRecorder(commit_result=_commit_result(verification=verification))
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test", "--verify"],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.commit")
    assert body["status"] == "committed"
    assert body["verification"]["ok"] is False
    assert body["verification"]["missing_claim_keys"] == ["claim:test"]


def test_graph_commit_rejects_raw_payload_option() -> None:
    _common.set_json(True)
    workbench = CommitRecorder(commit_result=_commit_result())
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test", "--file", "mutation.json"],
    )

    assert result.exit_code != 0
    assert workbench.commit_calls == []


def test_graph_commit_verify_exits_zero_when_only_quality_regressed() -> None:
    """The commit landed (`ok: true`); a quality regression is a warning, not a
    failed write, so a script never retries a commit that went through."""
    _common.set_json(True)
    verification = GraphIngestionVerificationResult(
        ok=False,
        status="degraded",
        plan_id="mutation-plan:test",
        pot_id="p",
        claim_keys=("claim:test",),
        readback_claim_keys=("claim:test",),
        readback_count=1,
        quality_status="watch",
        quality_regressions={"conflicting_claims": {"before": 0, "after": 1}},
        detail="quality findings increased after commit",
        recommended_next_action="Run the affected graph quality reports.",
    )
    workbench = CommitRecorder(commit_result=_commit_result(verification=verification))
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test", "--verify"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.commit")
    assert body["status"] == "committed"
    assert body["verification"]["status"] == "degraded"
    assert body["verification"]["ok"] is False
    assert any(
        "post-commit verification degraded: quality findings increased" in warning
        for warning in emitted["warnings"]
    )


def test_graph_commit_verify_exits_nonzero_when_a_committed_claim_is_missing() -> None:
    _common.set_json(True)
    verification = GraphIngestionVerificationResult(
        ok=False,
        status="failed",
        plan_id="mutation-plan:test",
        pot_id="p",
        claim_keys=("claim:test",),
        missing_claim_keys=("claim:test",),
        readback_count=0,
        detail="committed claim did not read back",
    )
    workbench = CommitRecorder(commit_result=_commit_result(verification=verification))
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test", "--verify"],
    )

    assert result.exit_code == _common.EXIT_VALIDATION, result.output


def test_graph_commit_verify_help_says_what_exits_nonzero() -> None:
    result = CliRunner().invoke(graph.graph_app, ["commit", "--help"])

    assert result.exit_code == 0
    # Rich wraps option help inside a bordered panel, so the border glyphs
    # land mid-sentence; drop them before looking for the phrase.
    text = " ".join(re.sub(r"[│╭╮╰╯─]", " ", _plain_cli_output(result.output)).split())
    assert "a quality regression alone is reported" in text


class BulkRecorder(ProposalRecorder, CommitRecorder):
    def __init__(self, *, proposal, commit_result):
        ProposalRecorder.__init__(self, proposal=proposal)
        CommitRecorder.__init__(self, commit_result=commit_result)


def test_graph_bulk_apply_chunks_and_commits(tmp_path) -> None:
    _common.set_json(True)
    workbench = BulkRecorder(proposal=_proposal(), commit_result=_commit_result())
    graph_service = _Graph()
    _common.set_runtime(_Host(graph_service, graph_workbench=workbench))
    payload_file = tmp_path / "bulk.json"
    manifest_file = tmp_path / "manifest.json"
    payload_file.write_text(json.dumps(_bulk_mutation_payload(3)), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "bulk",
            "apply",
            "--file",
            str(payload_file),
            "--chunk-size",
            "2",
            "--manifest",
            str(manifest_file),
            "--verify",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.bulk.apply")
    assert body["status"] == "committed"
    assert body["chunks_total"] == 2
    assert body["chunks_attempted"] == 2
    assert body["chunks_committed"] == 2
    assert body["operations_committed"] == 3
    assert body["verification"]["counts"] == {"claims": 3}
    assert len(workbench.propose_calls) == 2
    assert len(workbench.commit_calls) == 2
    assert len(workbench.propose_calls[0][0]["operations"]) == 2
    assert len(workbench.propose_calls[1][0]["operations"]) == 1
    assert workbench.propose_calls[0][0]["idempotency_key"] == "bulk:test:chunk-0001"
    assert workbench.propose_calls[1][0]["idempotency_key"] == "bulk:test:chunk-0002"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["status"] == "committed"
    assert manifest["chunks_committed"] == 2


def test_graph_bulk_apply_dry_run_does_not_commit(tmp_path) -> None:
    _common.set_json(True)
    workbench = BulkRecorder(proposal=_proposal(), commit_result=_commit_result())
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "bulk.ndjson"
    lines = [json.dumps(op) for op in _bulk_mutation_payload(2)["operations"]]
    payload_file.write_text("\n".join(lines), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "bulk",
            "apply",
            "--file",
            str(payload_file),
            "--chunk-size",
            "1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.bulk.apply")
    assert body["status"] == "validated"
    assert body["chunks_validated"] == 2
    assert body["operations_validated"] == 2
    assert workbench.commit_calls == []


def test_graph_bulk_apply_stops_on_failed_proposal(tmp_path) -> None:
    _common.set_json(True)
    proposal = _proposal(
        ok=False,
        status="invalid",
        issues=(
            {
                "code": "invalid_endpoints",
                "message": "invalid endpoint pair",
                "severity": "error",
                "op_index": 0,
            },
        ),
    )
    workbench = BulkRecorder(proposal=proposal, commit_result=_commit_result())
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "bulk.json"
    payload_file.write_text(json.dumps(_bulk_mutation_payload(3)), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "bulk",
            "apply",
            "--file",
            str(payload_file),
            "--chunk-size",
            "1",
        ],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.bulk.apply", ok=False)
    assert body == {}
    assert emitted["error"]["code"] == "invalid_endpoints"
    assert emitted["error"]["detail"]["status"] == "failed"
    assert emitted["error"]["detail"]["chunks_attempted"] == 1
    assert workbench.commit_calls == []


def test_graph_bulk_verify_reports_content_loss_separately_from_committed_count(
    tmp_path,
) -> None:
    _common.set_json(True)
    verification = GraphIngestionVerificationResult(
        ok=False,
        status="degraded",
        plan_id="mutation-plan:test",
        pot_id="p",
        content_readback={
            "checked_entities": ["service:fixture"],
            "mismatches": [
                {
                    "entity_key": "service:fixture",
                    "properties": ["description"],
                }
            ],
        },
        detail="content readback differs from the committed plan",
    )
    workbench = BulkRecorder(
        proposal=_proposal(), commit_result=_commit_result(verification=verification)
    )
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "bulk.json"
    payload_file.write_text(json.dumps(_bulk_mutation_payload(3)), encoding="utf-8")
    result = CliRunner().invoke(
        graph.graph_app,
        ["bulk", "apply", "--file", str(payload_file), "--chunk-size", "2", "--verify"],
    )
    assert result.exit_code != 0
    _assert_graph_envelope(json.loads(result.output), "graph.bulk.apply", ok=False)
    body = json.loads(result.output)["error"]["detail"]
    assert body["chunks_committed"] == 1
    assert body["chunks_attempted"] == 1
    assert body["chunks"][0]["commit"]["verification"]["content_readback"][
        "mismatches"
    ][0]["properties"] == ["description"]
    assert body["issues"][0]["code"] == "verification_failed"
    # Every chunk commit was asked to verify itself.
    assert all(call[3] is True for call in workbench.commit_calls)


class HistoryRecorder:
    def __init__(self, *, history_result):
        self.history_result = history_result
        self.history_calls = []

    def history(self, **kwargs):
        self.history_calls.append(kwargs)
        return self.history_result


def _history_result() -> GraphHistoryResult:
    return GraphHistoryResult(
        ok=True,
        pot_id="p",
        filters={"plan": "mutation-plan:test", "limit": 50},
        entries=(
            GraphHistoryEntry(
                kind="plan",
                id="mutation-plan:test",
                status="committed",
                plan_id="mutation-plan:test",
                mutation_id="mutation-1",
                occurred_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
                source_refs=("repo:manifest",),
                summary="mutation plan committed",
            ),
        ),
        subgraph_versions={"_global": 1},
    )


def test_graph_history_plan_returns_envelope() -> None:
    _common.set_json(True)
    workbench = HistoryRecorder(history_result=_history_result())
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["history", "--plan", "mutation-plan:test"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.history")
    assert emitted["subgraph_versions"] == {"_global": 1}
    assert body["entry_count"] == 1
    assert body["entries"][0]["plan_id"] == "mutation-plan:test"
    assert workbench.history_calls == [
        {
            "pot_id": "p",
            "entity_key": None,
            "claim_key": None,
            "subgraph": None,
            "plan_id": "mutation-plan:test",
            "mutation_id": None,
            "since": None,
            "until": None,
            "limit": 50,
            "include_claims": True,
        }
    ]


from tests.graph_cli_support import _bulk_mutation_payload, _valid_mutation_payload
from tests.graph_cli_support import graph_cli_state as graph_cli_state
