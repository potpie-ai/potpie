"""Deterministic Linear planner: entities, edges, evidence, action-specific extras."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from adapters.outbound.connectors.linear.plan import build_linear_issue_plan
from domain.context_events import EventRef
from adapters.outbound.connectors.linear.events import (
    LinearComment,
    LinearIssueEvent,
    LinearPerson,
    LinearState,
    linear_issue_from_payload,
)

pytestmark = pytest.mark.unit

DATA = Path(__file__).resolve().parent.parent / "data" / "linear"


def _detail() -> dict[str, Any]:
    return json.loads((DATA / "issue_detail.json").read_text(encoding="utf-8"))


def _ref() -> EventRef:
    return EventRef(event_id="evt-1", source_system="linear", pot_id="pot-1")


def test_plan_builds_issue_team_source_and_person_entities() -> None:
    issue = linear_issue_from_payload(_detail())
    plan = build_linear_issue_plan(
        event_ref=_ref(),
        event=LinearIssueEvent(action="update", issue=issue),
    )
    entity_keys = {e.entity_key for e in plan.entity_upserts}
    assert "linear:issue:ENG-42" in entity_keys
    assert "linear:team:team_eng" in entity_keys
    assert "source-ref:linear:issue:ENG-42" in entity_keys
    assert "linear:user:user_nandan" in entity_keys
    assert "linear:user:user_priya" in entity_keys
    assert "linear:label:team_eng:label_ctx" in entity_keys


def test_plan_builds_canonical_edges() -> None:
    issue = linear_issue_from_payload(_detail())
    plan = build_linear_issue_plan(
        event_ref=_ref(),
        event=LinearIssueEvent(action="update", issue=issue),
    )
    edge_triples = {
        (e.edge_type, e.from_entity_key, e.to_entity_key) for e in plan.edge_upserts
    }
    assert ("EVIDENCED_BY", "linear:issue:ENG-42", "source-ref:linear:issue:ENG-42") in edge_triples
    assert ("BELONGS_TO_TEAM", "linear:issue:ENG-42", "linear:team:team_eng") in edge_triples
    assert ("CREATED_BY", "linear:issue:ENG-42", "linear:user:user_nandan") in edge_triples
    assert ("ASSIGNED_TO", "linear:issue:ENG-42", "linear:user:user_priya") in edge_triples
    assert ("HAS_LABEL", "linear:issue:ENG-42", "linear:label:team_eng:label_ctx") in edge_triples


def test_plan_attaches_evidence_ref() -> None:
    issue = linear_issue_from_payload(_detail())
    plan = build_linear_issue_plan(
        event_ref=_ref(),
        event=LinearIssueEvent(action="create", issue=issue),
    )
    assert len(plan.evidence) == 1
    ev = plan.evidence[0]
    assert ev.kind == "source_ref"
    assert ev.ref == "source-ref:linear:issue:ENG-42"
    assert ev.metadata["provider"] == "linear"
    assert ev.metadata["identifier"] == "ENG-42"


def test_comment_added_event_adds_comment_entity_and_edges() -> None:
    issue = linear_issue_from_payload(_detail())
    comment = LinearComment(
        id="comment_01",
        body="Confirmed bug.",
        author=LinearPerson(id="user_priya", name="priya"),
    )
    plan = build_linear_issue_plan(
        event_ref=_ref(),
        event=LinearIssueEvent(action="comment_added", issue=issue, comment=comment),
    )
    entity_keys = {e.entity_key for e in plan.entity_upserts}
    assert "linear:comment:comment_01" in entity_keys
    edge_triples = {
        (e.edge_type, e.from_entity_key, e.to_entity_key) for e in plan.edge_upserts
    }
    assert (
        "HAS_COMMENT",
        "linear:issue:ENG-42",
        "linear:comment:comment_01",
    ) in edge_triples
    assert (
        "AUTHORED_BY",
        "linear:comment:comment_01",
        "linear:user:user_priya",
    ) in edge_triples


def test_state_change_summary_mentions_transition() -> None:
    issue = linear_issue_from_payload(_detail())
    plan = build_linear_issue_plan(
        event_ref=_ref(),
        event=LinearIssueEvent(
            action="state_change",
            issue=issue,
            previous_state=LinearState(name="Triage", type="unstarted"),
        ),
    )
    assert "Triage" in plan.summary
    assert "In Progress" in plan.summary
    # Episode body also surfaces the transition so semantic search can find it.
    assert "Triage" in plan.episodes[0].episode_body


def test_dedupes_entities_by_key() -> None:
    payload = _detail()
    payload["creator"] = payload["assignee"]  # same person as creator + assignee
    issue = linear_issue_from_payload(payload)
    plan = build_linear_issue_plan(
        event_ref=_ref(),
        event=LinearIssueEvent(action="update", issue=issue),
    )
    # The single user must appear only once in entity upserts.
    keys = [e.entity_key for e in plan.entity_upserts]
    assert keys.count("linear:user:user_priya") == 1


def test_benchmark_dataset_exposes_linear_issues() -> None:
    from benchmarks.models import DEFAULT_DATASET
    from benchmarks.dataset import load_dataset

    dataset = load_dataset(DEFAULT_DATASET)
    assert len(dataset.linear_issues) >= 2
    idents = {entry["issue"]["identifier"] for entry in dataset.linear_issues}
    assert {"ENG-42", "ENG-128"}.issubset(idents)
