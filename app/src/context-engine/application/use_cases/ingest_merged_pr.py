"""Ingest a merged PR into Graphiti + ledger."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from adapters.outbound.reconciliation.github_pr_compat import build_github_pr_merged_compatibility_plan
from application.use_cases.apply_reconciliation_plan import apply_reconciliation_plan
from application.use_cases.reconciliation_validation import validate_reconciliation_plan
from domain.context_events import EventRef
from domain.errors import ReconciliationApplyError
from domain.ingestion import IngestionResult
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort, LedgerScope
from domain.ports.structural_graph import StructuralGraphPort
from domain.reconciliation_flags import compat_pr_reconciler_enabled

logger = logging.getLogger(__name__)

SOURCE_TYPE = "github"


def ingest_merged_pull_request(
    ledger: IngestionLedgerPort,
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    scope: LedgerScope,
    repo_name: str,
    pr_data: dict[str, Any],
    commits: list[dict[str, Any]],
    review_threads: list[dict[str, Any]],
    linked_issues: list[dict[str, Any]],
    issue_comments: list[dict[str, Any]] | None = None,
) -> IngestionResult:
    pr_number = pr_data.get("number")
    source_id = f"pr_{pr_number}_merged"
    pr_entity_key = f"github:pr:{repo_name}:{pr_number}"

    existing = ledger.get_ingestion_log(scope, SOURCE_TYPE, source_id)
    if existing:
        logger.info(
            "Skipping already-ingested source %s for pot %s repo %s",
            source_id,
            scope.pot_id,
            scope.repo_name,
        )
        return IngestionResult(
            episode_uuid=existing.graphiti_episode_uuid,
            pr_entity_key=pr_entity_key,
            already_existed=True,
        )

    payload = {
        "pr_data": pr_data,
        "commits": commits,
        "review_threads": review_threads,
        "linked_issues": linked_issues,
        "issue_comments": issue_comments or [],
    }

    if compat_pr_reconciler_enabled():
        event_ref = EventRef(
            event_id=str(uuid4()),
            source_system="github",
            pot_id=scope.pot_id,
        )
        plan = build_github_pr_merged_compatibility_plan(
            event_ref=event_ref,
            repo_name=repo_name,
            pr_data=pr_data,
            commits=commits,
            review_threads=review_threads,
            linked_issues=linked_issues,
            issue_comments=issue_comments,
        )
        validate_reconciliation_plan(plan, scope.pot_id)
        try:
            result = apply_reconciliation_plan(
                episodic,
                structural,
                plan,
                expected_pot_id=scope.pot_id,
            )
        except ReconciliationApplyError:
            logger.exception("reconciliation apply failed for merged PR ingest")
            raise
        episode_uuid = result.episode_uuids[0] if result.episode_uuids else None
        stamp_counts = result.mutation_summary.stamp_counts
    else:
        from domain.episode_formatters import build_pr_episode

        episode = build_pr_episode(
            pr_data=pr_data,
            commits=commits,
            review_threads=review_threads,
            linked_issues=linked_issues,
            issue_comments=issue_comments,
        )

        episode_uuid = episodic.add_episode(
            pot_id=scope.pot_id,
            name=episode["name"],
            episode_body=episode["episode_body"],
            source_description=episode["source_description"],
            reference_time=episode["reference_time"],
        )

        stamp_counts = structural.stamp_pr_entities(
            pot_id=scope.pot_id,
            episode_uuid=episode_uuid or "",
            repo_name=repo_name,
            pr_number=pr_number,
            commits=commits,
            review_threads=review_threads,
            pr_data=pr_data,
            author=pr_data.get("author"),
            pr_title=pr_data.get("title"),
            issue_comments=issue_comments or [],
        )

    ok = ledger.try_append_ingestion_and_raw_event(
        scope=scope,
        source_type=SOURCE_TYPE,
        source_id=source_id,
        graphiti_episode_uuid=episode_uuid,
        payload=payload,
    )
    if not ok:
        after = ledger.get_ingestion_log(scope, SOURCE_TYPE, source_id)
        return IngestionResult(
            episode_uuid=after.graphiti_episode_uuid if after else episode_uuid,
            pr_entity_key=pr_entity_key,
            already_existed=True,
            stamp_counts=stamp_counts,
        )

    return IngestionResult(
        episode_uuid=episode_uuid,
        pr_entity_key=pr_entity_key,
        stamp_counts=stamp_counts,
    )
