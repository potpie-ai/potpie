"""Ingest one merged PR (e.g. webhook)."""

from __future__ import annotations

import logging
from typing import Any

from application.services.pr_bundle import fetch_full_pr
from application.use_cases.ingest_merged_pr import ingest_merged_pull_request
from domain.ports.context_graph import ContextGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort, ledger_scope_from_pot_repo
from domain.ports.pot_resolution import PotResolutionPort, resolve_write_repo
from domain.ports.settings import ContextEngineSettingsPort
from domain.ports.source_control import SourceControlPort

logger = logging.getLogger(__name__)

SOURCE_TYPE = "github"


def run_merged_pr_ingest_core(
    settings: ContextEngineSettingsPort,
    pots: PotResolutionPort,
    source: SourceControlPort,
    ledger: IngestionLedgerPort,
    context_graph: ContextGraphPort,
    pot_id: str,
    pr_number: int,
    *,
    repo_name: str | None = None,
    is_live_bridge: bool = True,
) -> dict[str, Any]:
    """Fetch merged PR data and apply its generic graph mutations."""
    del is_live_bridge

    if not settings.is_enabled():
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "pr_number": pr_number,
            "reason": "context_graph_disabled",
        }

    resolved = pots.resolve_pot(pot_id)
    if not resolved or not resolved.repos:
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "pr_number": pr_number,
            "reason": "pot_not_found_or_missing_repo",
        }
    primary = resolve_write_repo(resolved, repo_name=repo_name)
    if primary is None:
        reason = (
            "repo_not_in_pot"
            if (repo_name or "").strip()
            else "ambiguous_repo"
        )
        out: dict[str, Any] = {
            "status": "skipped",
            "pot_id": pot_id,
            "pr_number": pr_number,
            "reason": reason,
        }
        if reason == "ambiguous_repo":
            out["message"] = "Pot has multiple repositories; pass repo_name (owner/repo)."
        elif (repo_name or "").strip():
            out["repo_name"] = repo_name.strip()
        return out

    repo_name = primary.repo_name
    scope = ledger_scope_from_pot_repo(primary)
    source_id = f"pr_{pr_number}_merged"

    try:
        payload = fetch_full_pr(source, repo_name, pr_number)
        result = ingest_merged_pull_request(
            ledger=ledger,
            context_graph=context_graph,
            scope=scope,
            repo_name=repo_name,
            pr_data=payload["pr_data"],
            commits=payload["commits"],
            review_threads=payload["review_threads"],
            linked_issues=payload["linked_issues"],
            issue_comments=payload["issue_comments"],
        )

        ledger.update_bridge_status(
            scope,
            SOURCE_TYPE,
            source_id,
            entity_key=result.pr_entity_key,
            bridge_result=None,
            error=None,
        )
        return {
            "status": "success",
            "pot_id": pot_id,
            "repo_name": repo_name,
            "pr_number": pr_number,
            "mutations": dict(result.stamp_counts),
        }
    except Exception as exc:
        ledger.update_bridge_status(
            scope,
            SOURCE_TYPE,
            source_id,
            entity_key=f"github:pr:{repo_name}:{pr_number}",
            bridge_result=None,
            error=str(exc),
        )
        logger.exception(
            "Single PR ingest failed for pot=%s pr=%s",
            pot_id,
            pr_number,
        )
        return {
            "status": "error",
            "pot_id": pot_id,
            "pr_number": pr_number,
            "error": str(exc),
        }


def ingest_single_pull_request(
    settings: ContextEngineSettingsPort,
    pots: PotResolutionPort,
    source: SourceControlPort,
    ledger: IngestionLedgerPort,
    context_graph: ContextGraphPort,
    pot_id: str,
    pr_number: int,
    is_live_bridge: bool = True,
    *,
    repo_name: str | None = None,
) -> dict[str, Any]:
    return run_merged_pr_ingest_core(
        settings,
        pots,
        source,
        ledger,
        context_graph,
        pot_id,
        pr_number,
        repo_name=repo_name,
        is_live_bridge=is_live_bridge,
    )
