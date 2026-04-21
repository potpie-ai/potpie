"""Reconciliation plan, compat planner, and apply."""

from unittest.mock import MagicMock

import pytest

from adapters.outbound.reconciliation.github_pr_compat import (
    GitHubPrMergedCompatAgent,
    build_github_pr_merged_compatibility_plan,
)
from application.use_cases.apply_reconciliation_plan import apply_reconciliation_plan
from application.use_cases.reconcile_event import reconcile_event
from application.use_cases.agent_work_capture import bind_agent_work_recorder
from application.use_cases.reconciliation_validation import validate_reconciliation_plan
from domain.context_events import ContextEvent, EventRef, EventScope
from domain.errors import ReconciliationPlanValidationError
from domain.graph_mutations import EntityUpsert
from domain.ports.ingestion_ledger import LedgerScope
from adapters.outbound.reconciliation.llm_plan_convert import (
    llm_plan_to_reconciliation_plan,
)
from adapters.outbound.reconciliation.llm_plan_schema import (
    LlmEdgeDelete,
    LlmEdgeUpsert,
    LlmEntityUpsert,
    LlmEpisodeDraft,
    LlmEvidenceRef,
    LlmInvalidationOp,
    LlmReconciliationPlan,
)
from domain.reconciliation import ReconciliationPlan, ReconciliationRequest


def test_validate_compat_mutually_exclusive_with_generic() -> None:
    ref = EventRef(event_id="e1", source_system="github", pot_id="p1")
    plan = ReconciliationPlan(
        event_ref=ref,
        summary="s",
        episodes=[],
        entity_upserts=[
            EntityUpsert(entity_key="k", labels=("Entity",), properties={})
        ],
        compat_github_pr_merged=MagicMock(),
    )
    with pytest.raises(ReconciliationPlanValidationError):
        validate_reconciliation_plan(plan, "p1")


def test_compat_plan_build_roundtrip() -> None:
    ref = EventRef(event_id="e1", source_system="github", pot_id="pot-a")
    pr_data = {
        "number": 7,
        "title": "T",
        "author": "a",
        "merged_at": "2024-01-01T00:00:00Z",
        "body": "x",
        "files": [],
        "labels": [],
        "milestone": None,
    }
    plan = build_github_pr_merged_compatibility_plan(
        event_ref=ref,
        repo_name="o/r",
        pr_data=pr_data,
        commits=[],
        review_threads=[],
        linked_issues=[],
        issue_comments=[],
    )
    validate_reconciliation_plan(plan, "pot-a")
    assert plan.compat_github_pr_merged is not None
    assert len(plan.episodes) == 1
    assert plan.episodes[0].name == "pr_7_merged"


def test_apply_reconciliation_plan_compat_calls_stamp() -> None:
    ref = EventRef(event_id="e1", source_system="github", pot_id="p1")
    plan = build_github_pr_merged_compatibility_plan(
        event_ref=ref,
        repo_name="o/r",
        pr_data={
            "number": 2,
            "title": "t",
            "author": "u",
            "merged_at": "2024-01-02T00:00:00Z",
            "body": "",
            "files": [],
            "labels": [],
            "milestone": None,
        },
        commits=[],
        review_threads=[],
        linked_issues=[],
        issue_comments=[],
    )
    episodic = MagicMock()
    episodic.write_episode_drafts.return_value = ["uuid-1"]
    structural = MagicMock()
    structural.stamp_pr_entities.return_value = {"stamped_pr": 1}

    out = apply_reconciliation_plan(episodic, structural, plan, expected_pot_id="p1")

    assert out.ok
    assert out.episode_uuids == ["uuid-1"]
    structural.stamp_pr_entities.assert_called_once()
    episodic.write_episode_drafts.assert_called_once()


def test_compat_agent_from_payload() -> None:
    agent = GitHubPrMergedCompatAgent(repo_name="o/r")
    ev = ContextEvent(
        event_id="evt-1",
        source_system="github",
        event_type="pull_request",
        action="merged",
        pot_id="p1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
        source_id="pr_3_merged",
        payload={
            "pr_data": {
                "number": 3,
                "title": "x",
                "author": "a",
                "merged_at": "2024-01-01T00:00:00Z",
                "body": "",
                "files": [],
                "labels": [],
                "milestone": None,
            },
            "commits": [],
            "review_threads": [],
            "linked_issues": [],
            "issue_comments": [],
        },
    )
    req = ReconciliationRequest(event=ev, pot_id="p1", repo_name="o/r")
    plan = agent.run_reconciliation(req)
    assert plan.compat_github_pr_merged is not None


def test_reconcile_event_with_mock_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_RECONCILIATION_ENABLED", "1")
    episodic = MagicMock()
    episodic.write_episode_drafts.return_value = ["u1"]
    structural = MagicMock()
    structural.stamp_pr_entities.return_value = {}

    ev = ContextEvent(
        event_id="evt-2",
        source_system="github",
        event_type="pull_request",
        action="merged",
        pot_id="p1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
        source_id="pr_1_merged",
        payload={
            "pr_data": {
                "number": 1,
                "title": "t",
                "author": "a",
                "merged_at": "2024-01-01T00:00:00Z",
                "body": "",
                "files": [],
                "labels": [],
                "milestone": None,
            },
            "commits": [],
            "review_threads": [],
            "linked_issues": [],
            "issue_comments": [],
        },
    )
    agent = GitHubPrMergedCompatAgent(repo_name="o/r")
    req = ReconciliationRequest(event=ev, pot_id="p1", repo_name="o/r")

    ledger = MagicMock()
    ledger.claim_event_for_processing.return_value = True
    ledger.next_attempt_number.return_value = 1
    ledger.start_reconciliation_run.return_value = "run-1"

    result = reconcile_event(
        episodic,
        structural,
        agent,
        req,
        reco_ledger=ledger,
    )
    assert result.ok
    ledger.record_run_success.assert_called_once_with("run-1")
    ledger.record_event_reconciled.assert_called_once_with("evt-2")


def test_event_scope_matches_context_event() -> None:
    ev = ContextEvent(
        event_id="x",
        source_system="github",
        event_type="pull_request",
        action="merged",
        pot_id="p",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
        source_id="pr_1_merged",
    )
    scope = EventScope(
        pot_id="p",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
    )
    assert ev.pot_id == scope.pot_id


def test_agent_work_recorder_binds_to_supported_agent() -> None:
    class AgentWithRecorder:
        def __init__(self) -> None:
            self.recorder = None

        def set_work_event_recorder(self, recorder) -> None:  # type: ignore[no-untyped-def]
            self.recorder = recorder

    ledger = MagicMock()
    agent = AgentWithRecorder()

    bind_agent_work_recorder(agent, ledger, "run-1")  # type: ignore[arg-type]
    assert agent.recorder is not None
    agent.recorder.record("thought", title="t", body="b", payload={"x": 1})

    ledger.record_run_work_event.assert_called_once_with(
        "run-1",
        event_kind="thought",
        title="t",
        body="b",
        payload={"x": 1},
    )


def test_ledger_scope_compatible_with_event_scope() -> None:
    lr = LedgerScope(
        pot_id="p",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
    )
    es = EventScope(
        pot_id=lr.pot_id,
        provider=lr.provider,
        provider_host=lr.provider_host,
        repo_name=lr.repo_name,
    )
    assert es.pot_id == "p"


def test_llm_plan_convert_roundtrip() -> None:
    from datetime import datetime, timezone

    ref = EventRef(event_id="e1", source_system="github", pot_id="pot-a")
    llm = LlmReconciliationPlan(
        summary="test plan",
        episodes=[
            LlmEpisodeDraft(
                name="n",
                episode_body="b",
                source_description="s",
                reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        ],
        entity_upserts=[
            LlmEntityUpsert(
                entity_key="source-ref:github:pr:1",
                labels=["Entity", "SourceReference"],
                properties={
                    "source_system": "github",
                    "external_id": "pr:1",
                    "ref_type": "pull_request",
                },
            ),
            LlmEntityUpsert(
                entity_key="source-system:github",
                labels=["Entity", "SourceSystem"],
                properties={"name": "GitHub", "source_type": "source_control"},
            ),
        ],
        edge_upserts=[
            LlmEdgeUpsert(
                edge_type="FROM_SOURCE",
                from_entity_key="source-ref:github:pr:1",
                to_entity_key="source-system:github",
                properties={},
            )
        ],
        edge_deletes=[
            LlmEdgeDelete(
                edge_type="FROM_SOURCE",
                from_entity_key="source-ref:github:pr:1",
                to_entity_key="source-system:github",
            )
        ],
        invalidations=[
            LlmInvalidationOp(reason="r", target_entity_key="k"),
            LlmInvalidationOp(
                reason="r2",
                edge_type="FROM_SOURCE",
                from_entity_key="source-ref:github:pr:1",
                to_entity_key="source-system:github",
            ),
        ],
        evidence=[LlmEvidenceRef(kind="payload", ref="ref1")],
        confidence=0.9,
        warnings=["w"],
    )
    plan = llm_plan_to_reconciliation_plan(llm, event_ref=ref)
    validate_reconciliation_plan(plan, "pot-a")
    assert plan.summary == "test plan"
    assert len(plan.episodes) == 1
    assert plan.entity_upserts[0].labels == ("Entity", "SourceReference")
    assert plan.invalidations[1].target_edge == (
        "FROM_SOURCE",
        "source-ref:github:pr:1",
        "source-system:github",
    )


def test_llm_plan_convert_accepts_dict_output() -> None:
    ref = EventRef(event_id="e1", source_system="manual", pot_id="pot-a")
    plan = llm_plan_to_reconciliation_plan(
        {
            "summary": "record manual note",
            "episodes": [
                {
                    "name": "manual_note",
                    "episode_body": "deepesh deleted the database",
                    "source_description": "manual raw note",
                }
            ],
        },
        event_ref=ref,
    )

    validate_reconciliation_plan(plan, "pot-a")
    assert plan.summary == "record manual note"
    assert len(plan.episodes) == 1
    assert plan.invalidations == []
