"""SQLAlchemy models for context graph ledger tables (shared schema with Potpie)."""

from sqlalchemy import TIMESTAMP, Boolean, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ContextSyncState(Base):
    __tablename__ = "context_sync_state"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    project_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    last_synced_at: Mapped[object | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="idle")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("project_id", "source_type", name="uq_context_sync_project_source"),
    )


class ContextIngestionLog(Base):
    __tablename__ = "context_ingestion_log"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    project_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    graphiti_episode_uuid: Mapped[str | None] = mapped_column(String(255), nullable=True)
    entity_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    bridge_written: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    bridge_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    bridge_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    bridge_touched_by: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bridge_modified_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bridge_has_decision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
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

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    project_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    received_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    processed_at: Mapped[object | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "source_type", "source_id", name="uq_raw_events_source"),
        Index("ix_raw_events_project_source_id", "project_id", "source_type", "source_id"),
    )
