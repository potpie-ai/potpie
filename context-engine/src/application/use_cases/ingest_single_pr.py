"""Ingest one merged PR (e.g. webhook)."""

from __future__ import annotations

import logging
from typing import Any

from application.services.pr_bundle import fetch_full_pr
from application.use_cases.ingest_merged_pr import ingest_merged_pull_request
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort
from domain.ports.project_resolution import ProjectResolutionPort
from domain.ports.settings import ContextEngineSettingsPort
from domain.ports.source_control import SourceControlPort
from domain.ports.structural_graph import StructuralGraphPort

logger = logging.getLogger(__name__)

SOURCE_TYPE = "github_pr"


def ingest_single_pull_request(
    settings: ContextEngineSettingsPort,
    projects: ProjectResolutionPort,
    source: SourceControlPort,
    ledger: IngestionLedgerPort,
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    project_id: str,
    pr_number: int,
    is_live_bridge: bool = True,
) -> dict[str, Any]:
    if not settings.is_enabled():
        return {
            "status": "skipped",
            "project_id": project_id,
            "pr_number": pr_number,
            "reason": "context_graph_disabled",
        }

    resolved = projects.resolve(project_id)
    if not resolved or not resolved.repo_name:
        return {
            "status": "skipped",
            "project_id": project_id,
            "pr_number": pr_number,
            "reason": "project_not_found_or_missing_repo",
        }

    repo_name = resolved.repo_name
    source_id = f"pr_{pr_number}_merged"

    try:
        payload = fetch_full_pr(source, repo_name, pr_number)
        result = ingest_merged_pull_request(
            ledger=ledger,
            episodic=episodic,
            structural=structural,
            project_id=project_id,
            repo_name=repo_name,
            pr_data=payload["pr_data"],
            commits=payload["commits"],
            review_threads=payload["review_threads"],
            linked_issues=payload["linked_issues"],
            issue_comments=payload["issue_comments"],
        )

        merged_at = payload["pr_data"].get("merged_at")
        bridge_result = structural.write_bridges(
            project_id=project_id,
            pr_entity_key=result.pr_entity_key,
            pr_number=pr_number,
            repo_name=repo_name,
            files_with_patches=payload["pr_data"].get("files", []),
            review_threads=payload["review_threads"],
            merged_at=merged_at,
            is_live=is_live_bridge,
        )
        ledger.update_bridge_status(
            project_id,
            SOURCE_TYPE,
            source_id,
            entity_key=result.pr_entity_key,
            bridge_result=bridge_result,
            error=None,
        )
        return {
            "status": "success",
            "project_id": project_id,
            "repo_name": repo_name,
            "pr_number": pr_number,
            "bridges": bridge_result.as_dict(),
        }
    except Exception as exc:
        ledger.update_bridge_status(
            project_id,
            SOURCE_TYPE,
            source_id,
            entity_key=f"github:pr:{repo_name}:{pr_number}",
            bridge_result=None,
            error=str(exc),
        )
        logger.exception(
            "Single PR ingest failed for project=%s pr=%s",
            project_id,
            pr_number,
        )
        return {
            "status": "error",
            "project_id": project_id,
            "pr_number": pr_number,
            "error": str(exc),
        }
