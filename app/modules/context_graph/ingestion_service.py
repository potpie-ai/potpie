"""Ingestion service: dedup via context_ingestion_log, format episode, add to Graphiti, log."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config_provider import config_provider
from app.modules.context_graph.episode_formatters import (
    format_github_commit_episode,
    format_github_pr_episode,
)
from app.modules.context_graph.graphiti_client import ContextGraphClient
from app.modules.context_graph.models import ContextIngestionLog
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


async def ingest_episode(
    db: AsyncSession,
    project_id: str,
    source_type: str,
    source_id: str,
    payload: dict,
    event_type: Optional[str] = None,
    branch_name: Optional[str] = None,
) -> Optional[str]:
    """Ensure one episode is in the context graph and in the ingestion log (idempotent).

    Args:
        db: Async SQLAlchemy session (e.g. from BaseTask.async_db()).
        project_id: Project UUID.
        source_type: "github_pr" or "github_commit".
        source_id: Dedup key e.g. "pr_42_merged", "commit_abc123".
        payload: Raw PR or commit dict for the formatter.
        event_type: For github_pr, "opened" or "merged".
        branch_name: For github_commit, the branch name (required).

    Returns:
        Graphiti episode UUID if ingested, None if disabled or already present.
    """
    cfg = config_provider.get_context_graph_config()
    if not cfg.get("enabled"):
        return None

    # Idempotent: already ingested
    result = await db.execute(
        select(ContextIngestionLog.graphiti_episode_uuid).where(
            ContextIngestionLog.project_id == project_id,
            ContextIngestionLog.source_type == source_type,
            ContextIngestionLog.source_id == source_id,
        )
    )
    existing_uuid = result.scalars().first()
    if existing_uuid is not None:
        return existing_uuid

    # Format episode
    if source_type == "github_pr":
        if not event_type:
            event_type = "merged"
        formatted = format_github_pr_episode(payload, event_type)
        ref_time = payload.get("updated_at") or payload.get("merged_at") or payload.get("created_at")
    elif source_type == "github_commit":
        if not branch_name:
            branch_name = payload.get("branch_name") or "main"
        formatted = format_github_commit_episode(payload, branch_name)
        commit = payload.get("commit")
        author = (commit or {}).get("author") if isinstance(commit, dict) else {}
        ref_time = (author or {}).get("date") if isinstance(author, dict) else payload.get("created_at")
    else:
        logger.warning("Unknown source_type for context graph: %s", source_type)
        return None

    # Parse reference_time to datetime
    if isinstance(ref_time, datetime):
        reference_time = ref_time
    elif isinstance(ref_time, str):
        try:
            reference_time = datetime.fromisoformat(ref_time.replace("Z", "+00:00"))
        except Exception:
            reference_time = datetime.utcnow()
    else:
        reference_time = datetime.utcnow()

    # Add to Graphiti
    client = ContextGraphClient()
    episode_uuid = await client.add_episode(
        project_id=project_id,
        name=formatted["name"],
        episode_body=formatted["episode_body"],
        source_description=formatted["source_description"],
        reference_time=reference_time,
    )
    if not episode_uuid:
        return None

    # Log and commit
    log_entry = ContextIngestionLog(
        project_id=project_id,
        source_type=source_type,
        source_id=formatted["source_id"],
        graphiti_episode_uuid=episode_uuid,
    )
    db.add(log_entry)
    await db.commit()
    logger.info(
        "Context graph ingested episode project_id=%s source_type=%s source_id=%s",
        project_id, source_type, formatted["source_id"],
    )
    return episode_uuid
