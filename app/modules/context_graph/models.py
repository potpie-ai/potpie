from sqlalchemy import TIMESTAMP, Boolean, Column, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from app.core.base_model import Base


class ContextSyncState(Base):
    __tablename__ = "context_sync_state"

    id = Column(String(255), primary_key=True)
    project_id = Column(Text, nullable=False, index=True)
    source_type = Column(String(64), nullable=False)
    last_synced_at = Column(TIMESTAMP(timezone=True), nullable=True)
    status = Column(String(32), nullable=False, default="idle")
    error = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("project_id", "source_type", name="uq_context_sync_project_source"),
    )


class ContextIngestionLog(Base):
    __tablename__ = "context_ingestion_log"

    id = Column(String(255), primary_key=True)
    project_id = Column(Text, nullable=False, index=True)
    source_type = Column(String(64), nullable=False)
    source_id = Column(String(255), nullable=False)
    graphiti_episode_uuid = Column(String(255), nullable=True)
    entity_key = Column(String(512), nullable=True)

    bridge_written = Column(Boolean, nullable=False, default=False)
    bridge_status = Column(String(32), nullable=False, default="pending")
    bridge_error = Column(Text, nullable=True)
    bridge_touched_by = Column(Integer, nullable=False, default=0)
    bridge_modified_in = Column(Integer, nullable=False, default=0)
    bridge_has_decision = Column(Integer, nullable=False, default=0)

    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "source_type",
            "source_id",
            name="uq_context_ingestion_project_source_id",
        ),
        Index(
            "ix_context_ingestion_project_source_id",
            "project_id",
            "source_type",
            "source_id",
        ),
    )


class RawEvent(Base):
    __tablename__ = "raw_events"

    id = Column(String(255), primary_key=True)
    project_id = Column(Text, nullable=False, index=True)
    source_type = Column(String(64), nullable=False)
    source_id = Column(String(255), nullable=False)
    payload = Column(JSONB, nullable=False)
    received_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    processed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "source_type", "source_id", name="uq_raw_events_source"),
        Index("ix_raw_events_project_source_id", "project_id", "source_type", "source_id"),
    )
