"""Ingest a merged PR into Graphiti + ledger."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from adapters.outbound.reconciliation.github_pr_plan import build_github_pr_merged_plan
from application.use_cases.reconciliation_validation import validate_reconciliation_plan
from domain.context_events import EventRef
from domain.errors import ReconciliationApplyError
from domain.ingestion import IngestionResult
from domain.ports.context_graph import ContextGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort, LedgerScope

logger = logging.getLogger(__name__)

SOURCE_TYPE = "github"


def ingest_merged_pull_request(
    ledger: IngestionLedgerPort,
    context_graph: ContextGraphPort,
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

    event_ref = EventRef(
        event_id=str(uuid4()),
        source_system="github",
        pot_id=scope.pot_id,
    )
    plan = build_github_pr_merged_plan(
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
        from datetime import datetime, timezone

        from domain.graph_mutations import ProvenanceContext

        prov_ctx = ProvenanceContext(
            source_kind="pull_request",
            source_ref=source_id,
            event_received_at=datetime.now(timezone.utc),
            created_by_agent="github_pr_merged_planner",
        )
        result = context_graph.apply_plan(
            plan,
            expected_pot_id=scope.pot_id,
            provenance_context=prov_ctx,
        )
    except ReconciliationApplyError:
        logger.exception("reconciliation apply failed for merged PR ingest")
        raise
    episode_uuid = result.episode_uuids[0] if result.episode_uuids else None
    stamp_counts = {
        "entity_upserts_applied": result.mutation_summary.entity_upserts_applied,
        "edge_upserts_applied": result.mutation_summary.edge_upserts_applied,
        "edge_deletes_applied": result.mutation_summary.edge_deletes_applied,
        "invalidations_applied": result.mutation_summary.invalidations_applied,
    }

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
