"""SQLAlchemy models for context graph ledger tables (shared schema with Potpie)."""

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ContextSyncState(Base):
    __tablename__ = "context_sync_state"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    pot_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="github")
    provider_host: Mapped[str] = mapped_column(
        String(255), nullable=False, default="github.com"
    )
    repo_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    last_synced_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="idle")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "pot_id",
            "provider",
            "provider_host",
            "repo_name",
            "source_type",
            name="uq_context_sync_pot_repo_source",
        ),
    )


class ContextEventModel(Base):
    """Canonical inbound events for reconciliation (agent lifecycle) and merged-PR ledger rows."""

    __tablename__ = "context_events"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    pot_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="github")
    provider_host: Mapped[str] = mapped_column(
        String(255), nullable=False, default="github.com"
    )
    repo_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_system: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_event_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    occurred_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    received_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="received")
    ingestion_kind: Mapped[str] = mapped_column(
        String(64), nullable=False, default="agent_reconciliation"
    )
    job_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_channel: Mapped[str] = mapped_column(
        String(64), nullable=False, default="unknown"
    )
    dedup_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    step_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    step_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    step_error: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    completed_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    event_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_metadata: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    graphiti_episode_uuid: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    entity_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    bridge_written: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    bridge_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )
    bridge_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    bridge_touched_by: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bridge_modified_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bridge_has_decision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    raw_processed_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "pot_id",
            "provider",
            "provider_host",
            "repo_name",
            "source_system",
            "source_id",
            name="uq_context_events_dedupe",
        ),
        Index(
            "ix_context_events_pot_status",
            "pot_id",
            "status",
        ),
    )


class ContextEpisodeStepModel(Base):
    """Durable episode apply step for async ingestion (replay / bulk apply)."""

    __tablename__ = "context_episode_steps"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    pot_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("context_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    step_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    step_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    applied_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    queued_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    started_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    completed_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    run_id: Mapped[str | None] = mapped_column(
        String(255),
        ForeignKey("context_reconciliation_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "sequence",
            name="uq_context_episode_steps_event_sequence",
        ),
    )


class ContextReconciliationRun(Base):
    """One attempt to reconcile a context event."""

    __tablename__ = "context_reconciliation_runs"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("context_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    agent_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    agent_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    toolset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    plan_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    episode_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    entity_mutation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    edge_mutation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=True
    )
    completed_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    plan_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "attempt_number",
            name="uq_context_reconciliation_runs_event_attempt",
        ),
    )


class ContextReconciliationWorkEvent(Base):
    """Ordered agent-working event captured during a reconciliation run."""

    __tablename__ = "context_reconciliation_work_events"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("context_reconciliation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("context_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "sequence",
            name="uq_context_reconciliation_work_events_run_sequence",
        ),
        Index(
            "ix_context_reconciliation_work_events_event_run",
            "event_id",
            "run_id",
        ),
    )
