"""Context graph ingestion service for GitHub events."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.modules.context_graph.entity_key_stamper import stamp_pr_entities
from app.modules.context_graph.episode_formatters import build_pr_episode
from app.modules.context_graph.graphiti_client import ContextGraphClient
from app.modules.context_graph.models import ContextIngestionLog, RawEvent
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class IngestionResult:
    """Structured return value so callers have everything needed for bridge writing."""

    __slots__ = (
        "episode_uuid",
        "pr_entity_key",
        "already_existed",
        "stamp_counts",
    )

    def __init__(
        self,
        episode_uuid: str | None,
        pr_entity_key: str,
        already_existed: bool = False,
        stamp_counts: dict[str, int] | None = None,
    ):
        self.episode_uuid = episode_uuid
        self.pr_entity_key = pr_entity_key
        self.already_existed = already_existed
        self.stamp_counts = stamp_counts or {}


def ingest_pr(
    db: Session,
    project_id: str,
    repo_name: str,
    pr_data: dict[str, Any],
    commits: list[dict[str, Any]],
    review_threads: list[dict[str, Any]],
    linked_issues: list[dict[str, Any]],
    issue_comments: list[dict[str, Any]] | None = None,
) -> IngestionResult:
    source_type = "github_pr"
    pr_number = pr_data.get("number")
    source_id = f"pr_{pr_number}_merged"
    pr_entity_key = f"github:pr:{repo_name}:{pr_number}"

    existing = (
        db.query(ContextIngestionLog)
        .filter(
            ContextIngestionLog.project_id == project_id,
            ContextIngestionLog.source_type == source_type,
            ContextIngestionLog.source_id == source_id,
        )
        .first()
    )
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

    client = ContextGraphClient()
    episode = build_pr_episode(
        pr_data=pr_data,
        commits=commits,
        review_threads=review_threads,
        linked_issues=linked_issues,
        issue_comments=issue_comments,
    )

    episode_uuid = client.add_episode(
        project_id=project_id,
        name=episode["name"],
        episode_body=episode["episode_body"],
        source_description=episode["source_description"],
        reference_time=episode["reference_time"],
    )

    stamp_counts: dict[str, int] = {}
    if episode_uuid:
        stamp_counts = stamp_pr_entities(
            project_id=project_id,
            episode_uuid=episode_uuid,
            repo_name=repo_name,
            pr_number=pr_number,
            commits=commits,
            review_threads=review_threads,
            author=pr_data.get("author"),
        )

    try:
        raw_event = RawEvent(
            id=str(uuid4()),
            project_id=project_id,
            source_type=source_type,
            source_id=source_id,
            payload={
                "pr_data": pr_data,
                "commits": commits,
                "review_threads": review_threads,
                "linked_issues": linked_issues,
                "issue_comments": issue_comments or [],
            },
            received_at=datetime.utcnow(),
            processed_at=datetime.utcnow(),
        )
        db.add(raw_event)

        ingestion_log = ContextIngestionLog(
            id=str(uuid4()),
            project_id=project_id,
            source_type=source_type,
            source_id=source_id,
            graphiti_episode_uuid=episode_uuid,
            bridge_written=False,
        )
        db.add(ingestion_log)
        db.commit()
        return IngestionResult(
            episode_uuid=episode_uuid,
            pr_entity_key=pr_entity_key,
            stamp_counts=stamp_counts,
        )
    except IntegrityError:
        db.rollback()
        existing_after = (
            db.query(ContextIngestionLog)
            .filter(
                ContextIngestionLog.project_id == project_id,
                ContextIngestionLog.source_type == source_type,
                ContextIngestionLog.source_id == source_id,
            )
            .first()
        )
        return IngestionResult(
            episode_uuid=existing_after.graphiti_episode_uuid if existing_after else episode_uuid,
            pr_entity_key=pr_entity_key,
            already_existed=True,
            stamp_counts=stamp_counts,
        )
    except Exception:
        db.rollback()
        logger.exception("Failed ingesting PR %s for project %s", source_id, project_id)
        raise
