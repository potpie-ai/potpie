"""Ingest a merged PR into Graphiti + ledger."""

from __future__ import annotations

import logging
from typing import Any

from domain.episode_formatters import build_pr_episode
from domain.ingestion import IngestionResult
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort
from domain.ports.structural_graph import StructuralGraphPort

logger = logging.getLogger(__name__)

SOURCE_TYPE = "github_pr"


def ingest_merged_pull_request(
    ledger: IngestionLedgerPort,
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    project_id: str,
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

    existing = ledger.get_ingestion_log(project_id, SOURCE_TYPE, source_id)
    if existing:
        logger.info(
            "Skipping already-ingested source %s for project %s",
            source_id,
            project_id,
        )
        return IngestionResult(
            episode_uuid=existing.graphiti_episode_uuid,
            pr_entity_key=pr_entity_key,
            already_existed=True,
        )

    episode = build_pr_episode(
        pr_data=pr_data,
        commits=commits,
        review_threads=review_threads,
        linked_issues=linked_issues,
        issue_comments=issue_comments,
    )

    episode_uuid = episodic.add_episode(
        project_id=project_id,
        name=episode["name"],
        episode_body=episode["episode_body"],
        source_description=episode["source_description"],
        reference_time=episode["reference_time"],
    )

    # Always stamp structural graph (PR entity, review threads, conversation) even if Graphiti
    # returned no episode UUID — otherwise write_bridges runs with no Decision / PR linkage data.
    stamp_counts = structural.stamp_pr_entities(
        project_id=project_id,
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

    payload = {
        "pr_data": pr_data,
        "commits": commits,
        "review_threads": review_threads,
        "linked_issues": linked_issues,
        "issue_comments": issue_comments or [],
    }

    ok = ledger.try_append_ingestion_and_raw_event(
        project_id=project_id,
        source_type=SOURCE_TYPE,
        source_id=source_id,
        graphiti_episode_uuid=episode_uuid,
        payload=payload,
    )
    if not ok:
        after = ledger.get_ingestion_log(project_id, SOURCE_TYPE, source_id)
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
