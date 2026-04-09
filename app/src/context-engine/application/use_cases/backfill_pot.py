"""Backfill merged PRs for one pot (all attached repos, or one repo when specified)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from application.services.pr_bundle import fetch_full_pr
from application.use_cases.ingest_merged_pr import ingest_merged_pull_request
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort, ledger_scope_from_pot_repo
from domain.ports.pot_resolution import PotResolutionPort, ResolvedPotRepo, resolve_write_repo
from domain.ports.settings import ContextEngineSettingsPort
from domain.ports.source_control import SourceControlPort
from domain.ports.structural_graph import StructuralGraphPort

logger = logging.getLogger(__name__)

SOURCE_TYPE = "github"


def _backfill_one_repo(
    *,
    settings: ContextEngineSettingsPort,
    source: SourceControlPort,
    ledger: IngestionLedgerPort,
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    pot_id: str,
    primary: ResolvedPotRepo,
    rate_limit_sleep_s: float,
) -> dict[str, Any]:
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
                    "Failed ingesting PR #%s for pot %s repo %s",
                    pr.number,
                    pot_id,
                    repo_name,
                )

        ledger.update_sync_state_success(scope, SOURCE_TYPE, latest_merged_at)
        return {
            "status": "success",
            "repo_name": repo_name,
            "ingested": ingested,
            "skipped": skipped,
            "failed": failed,
            "max_prs_per_run": max_per_run,
            "last_synced_at": latest_merged_at.isoformat() if latest_merged_at else None,
        }
    except Exception as exc:
        ledger.update_sync_state_error(scope, SOURCE_TYPE, str(exc))
        logger.exception(
            "Context graph backfill failed for pot %s repo %s", pot_id, repo_name
        )
        return {
            "status": "error",
            "repo_name": repo_name,
            "error": str(exc),
            "ingested": ingested,
            "failed": failed,
        }


def backfill_pot_context(
    settings: ContextEngineSettingsPort,
    pots: PotResolutionPort,
    source_for_repo: Callable[[str], SourceControlPort],
    ledger: IngestionLedgerPort,
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    target_repo_name: str | None = None,
    rate_limit_sleep_s: float = 0.5,
) -> dict[str, Any]:
    if not settings.is_enabled():
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "context_graph_disabled",
        }

    resolved = pots.resolve_pot(pot_id)
    if not resolved:
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "pot_not_found_or_missing_repo",
        }
    if not resolved.repos:
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "pot_not_found_or_missing_repo",
        }

    if target_repo_name and target_repo_name.strip():
        selected = resolve_write_repo(resolved, repo_name=target_repo_name)
        if selected is None:
            return {
                "status": "skipped",
                "pot_id": pot_id,
                "reason": "repo_not_in_pot",
                "repo_name": target_repo_name,
            }
        repos = [selected]
    else:
        repos = list(resolved.repos)

    if not resolved.ready:
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "pot_not_ready",
        }

    repo_results: list[dict[str, Any]] = []
    for rr in repos:
        if not rr.ready:
            repo_results.append(
                {
                    "status": "skipped",
                    "repo_name": rr.repo_name,
                    "reason": "repo_not_ready",
                }
            )
            continue
        src = source_for_repo(rr.repo_name)
        one = _backfill_one_repo(
            settings=settings,
            source=src,
            ledger=ledger,
            episodic=episodic,
            structural=structural,
            pot_id=pot_id,
            primary=rr,
            rate_limit_sleep_s=rate_limit_sleep_s,
        )
        repo_results.append(one)

    total_ingested = sum(int(r.get("ingested", 0)) for r in repo_results)
    total_failed = sum(int(r.get("failed", 0)) for r in repo_results)
    any_err = any(r.get("status") == "error" for r in repo_results)
    out_status = "error" if any_err else "success"
    out: dict[str, Any] = {
        "status": out_status,
        "pot_id": pot_id,
        "repo_results": repo_results,
        "ingested": total_ingested,
        "failed": total_failed,
    }
    if len(repo_results) == 1:
        r0 = repo_results[0]
        out["repo_name"] = r0.get("repo_name")
        out["skipped"] = r0.get("skipped")
        out["max_prs_per_run"] = r0.get("max_prs_per_run")
        out["last_synced_at"] = r0.get("last_synced_at")
    return out
