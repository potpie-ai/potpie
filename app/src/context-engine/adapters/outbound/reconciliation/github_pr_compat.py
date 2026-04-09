"""Deterministic GitHub merged-PR plan (regression baseline for reconciliation)."""

from __future__ import annotations

from typing import Any

from domain.context_events import EventRef
from domain.episode_formatters import build_pr_episode
from domain.reconciliation import (
    EpisodeDraft,
    GitHubPrMergedCompat,
    ReconciliationPlan,
    ReconciliationRequest,
)


def build_github_pr_merged_compatibility_plan(
    *,
    event_ref: EventRef,
    repo_name: str,
    pr_data: dict[str, Any],
    commits: list[dict[str, Any]],
    review_threads: list[dict[str, Any]],
    linked_issues: list[dict[str, Any]],
    issue_comments: list[dict[str, Any]] | None = None,
) -> ReconciliationPlan:
    """Build the same episodic + structural intent as legacy ``ingest_merged_pull_request``."""
    ep = build_pr_episode(
        pr_data=pr_data,
        commits=commits,
        review_threads=review_threads,
        linked_issues=linked_issues,
        issue_comments=issue_comments,
    )
    pr_number = pr_data.get("number")
    summary = f"compat merged GitHub PR #{pr_number} ({repo_name})"
    draft = EpisodeDraft(
        name=ep["name"],
        episode_body=ep["episode_body"],
        source_description=ep["source_description"],
        reference_time=ep["reference_time"],
    )
    return ReconciliationPlan(
        event_ref=event_ref,
        summary=summary,
        episodes=[draft],
        compat_github_pr_merged=GitHubPrMergedCompat(
            repo_name=repo_name,
            pr_data=pr_data,
            commits=commits,
            review_threads=review_threads,
            linked_issues=linked_issues,
            issue_comments=issue_comments,
        ),
    )


class GitHubPrMergedCompatAgent:
    """``ReconciliationAgentPort`` that ignores LLMs and builds the compatibility plan from payload."""

    def __init__(self, repo_name: str) -> None:
        self._repo_name = repo_name

    def run_reconciliation(self, request: ReconciliationRequest) -> ReconciliationPlan:
        p = request.event.payload
        ref = EventRef(
            event_id=request.event.event_id,
            source_system=request.event.source_system,
            pot_id=request.pot_id,
        )
        return build_github_pr_merged_compatibility_plan(
            event_ref=ref,
            repo_name=self._repo_name,
            pr_data=p["pr_data"],
            commits=list(p.get("commits") or []),
            review_threads=list(p.get("review_threads") or []),
            linked_issues=list(p.get("linked_issues") or []),
            issue_comments=list(p.get("issue_comments") or []) or None,
        )

    def capability_metadata(self) -> dict[str, Any]:
        return {"agent": "github_pr_merged_compat", "version": "1"}
