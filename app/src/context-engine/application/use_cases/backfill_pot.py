"""Backfill merged PRs for one pot (one primary repo per run)."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from application.services.pr_bundle import fetch_full_pr
from application.use_cases.ingest_merged_pr import ingest_merged_pull_request
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort, ledger_scope_from_pot_repo
from domain.ports.pot_resolution import PotResolutionPort
from domain.ports.settings import ContextEngineSettingsPort
from domain.ports.source_control import SourceControlPort
from domain.ports.structural_graph import StructuralGraphPort

logger = logging.getLogger(__name__)

SOURCE_TYPE = "github_pr"


def backfill_pot_context(
    settings: ContextEngineSettingsPort,
    pots: PotResolutionPort,
    source: SourceControlPort,
    ledger: IngestionLedgerPort,
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    pot_id: str,
    rate_limit_sleep_s: float = 0.5,
) -> dict[str, Any]:
    if not settings.is_enabled():
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "context_graph_disabled",
        }

    resolved = pots.resolve_pot(pot_id)
    primary = resolved.primary_repo() if resolved else None
    if not resolved or not primary or not primary.repo_name:
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "pot_not_found_or_missing_repo",
        }
    if not resolved.ready or not primary.ready:
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "pot_not_ready",
        }

    repo_name = primary.repo_name
    scope = ledger_scope_from_pot_repo(primary)
    sync = ledger.get_or_create_sync_state(scope, SOURCE_TYPE)
    cursor = sync.last_synced_at
    latest_merged_at: datetime | None = cursor
    ledger.update_sync_state_running(scope, SOURCE_TYPE)

    max_per_run = settings.backfill_max_prs_per_run()
    ingested = 0
    skipped = 0
    failed = 0

    try:
        for pr in source.iter_closed_pulls(repo_name):
            if ingested >= max_per_run:
                break
            if not pr.merged_at:
                continue
            if cursor is not None and pr.merged_at <= cursor:
                skipped += 1
                continue

            source_id = f"pr_{pr.number}_merged"
            try:
                payload = fetch_full_pr(source, repo_name, pr.number)
                result = ingest_merged_pull_request(
                    ledger=ledger,
                    episodic=episodic,
                    structural=structural,
                    scope=scope,
                    repo_name=repo_name,
                    pr_data=payload["pr_data"],
                    commits=payload["commits"],
                    review_threads=payload["review_threads"],
                    linked_issues=payload["linked_issues"],
                    issue_comments=payload["issue_comments"],
                )

                merged_at = pr.merged_at.isoformat() if pr.merged_at else None
                bridge_result = structural.write_bridges(
                    pot_id=pot_id,
                    pr_entity_key=result.pr_entity_key,
                    pr_number=pr.number,
                    repo_name=repo_name,
                    files_with_patches=payload["pr_data"].get("files", []),
                    review_threads=payload["review_threads"],
                    merged_at=merged_at,
                    is_live=False,
                )
                ledger.update_bridge_status(
                    scope,
                    SOURCE_TYPE,
                    source_id,
                    entity_key=result.pr_entity_key,
                    bridge_result=bridge_result,
                    error=None,
                )
                ingested += 1
                latest_merged_at = pr.merged_at
                time.sleep(rate_limit_sleep_s)
            except Exception as exc:
                failed += 1
                ledger.update_bridge_status(
                    scope,
                    SOURCE_TYPE,
                    source_id,
                    entity_key=f"github:pr:{repo_name}:{pr.number}",
                    bridge_result=None,
                    error=str(exc),
                )
                logger.exception(
                    "Failed ingesting PR #%s for pot %s",
                    pr.number,
                    pot_id,
                )

        ledger.update_sync_state_success(scope, SOURCE_TYPE, latest_merged_at)
        return {
            "status": "success",
            "pot_id": pot_id,
            "repo_name": repo_name,
            "ingested": ingested,
            "skipped": skipped,
            "failed": failed,
            "max_prs_per_run": max_per_run,
            "last_synced_at": latest_merged_at.isoformat() if latest_merged_at else None,
        }
    except Exception as exc:
        ledger.update_sync_state_error(scope, SOURCE_TYPE, str(exc))
        logger.exception("Context graph backfill failed for pot %s", pot_id)
        return {
            "status": "error",
            "pot_id": pot_id,
            "repo_name": repo_name,
            "error": str(exc),
            "ingested": ingested,
            "failed": failed,
        }
