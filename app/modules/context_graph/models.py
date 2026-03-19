"""SQLAlchemy models for context graph ETL state and ingestion log."""

from sqlalchemy import Column, Integer, Text, TIMESTAMP, UniqueConstraint, func

from app.core.base_model import Base


class ContextSyncState(Base):
    """ETL cursor and status per project + source_type."""

    __tablename__ = "context_sync_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Text, nullable=False)
    source_type = Column(Text, nullable=False)  # e.g. "github_pr", "github_commit"
    last_synced_at = Column(TIMESTAMP(timezone=True), nullable=True)
    status = Column(Text, default="idle", nullable=False)  # idle | running | failed
    error = Column(Text, nullable=True)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("project_id", "source_type", name="uq_context_sync_state_project_source"),)


class ContextIngestionLog(Base):
    """Deduplication log: one row per ingested (project_id, source_type, source_id)."""

    __tablename__ = "context_ingestion_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Text, nullable=False)
    source_type = Column(Text, nullable=False)  # e.g. "github_pr", "github_commit"
    source_id = Column(Text, nullable=False)  # e.g. "pr_42_merged", "commit_abc123"
    graphiti_episode_uuid = Column(Text, nullable=False)
    ingested_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "project_id", "source_type", "source_id",
            name="uq_context_ingestion_log_project_source_source_id",
        ),
    )
