"""SQLAlchemy models for context graph ledger tables (shared schema with Potpie)."""

from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    Numeric,
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
    actor_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    actor_surface: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actor_client_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    actor_auth_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
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
        Index(
            "ix_context_events_pot_actor",
            "pot_id",
            "actor_user_id",
        ),
        Index(
            "ix_context_events_pot_surface",
            "pot_id",
            "actor_surface",
        ),
    )


class ContextReconciliationBatchModel(Base):
    """Batch of context events processed as one reconciliation-agent run."""

    __tablename__ = "context_reconciliation_batches"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    pot_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    # ``next_ready_at`` is a legacy column from the debounced-dispatcher era;
    # the migration ``ctx_drop_debounce_20260512`` made it nullable. Kept on
    # the model for rollback safety; not read or written by application code.
    next_ready_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    claimed_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    completed_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    __table_args__ = (
        Index(
            "ix_context_reconciliation_batches_pot_status",
            "pot_id",
            "status",
        ),
        Index(
            "uq_context_reconciliation_batches_open_per_pot",
            "pot_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )


class ContextReconciliationBatchEventModel(Base):
    """Membership of context events in a reconciliation batch."""

    __tablename__ = "context_reconciliation_batch_events"

    batch_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("context_reconciliation_batches.id", ondelete="CASCADE"),
        primary_key=True,
    )
    event_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("context_events.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    added_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at: Mapped[object | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )


class ContextPotIngestionConfigModel(Base):
    """Per-pot ingestion mode + window.

    Phase 4: events admitted on a windowed pot accumulate without enqueuing.
    A scheduled task flushes the open batch once it's older than
    ``window_minutes`` (or above ``min_batch_size``). All pots are
    initialised to ``windowed/5min`` via migration so the platform default
    is the documented behaviour, but operators / pot owners can switch any
    pot to ``immediate`` (legacy behaviour) at any time.
    """

    __tablename__ = "context_pot_ingestion_config"

    pot_id: Mapped[str] = mapped_column(Text, primary_key=True)
    mode: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'windowed'")
    )
    window_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("5")
    )
    # NULL = no early-flush trigger (only the time window applies).
    min_batch_size: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_by_user_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )


class ContextAgentCheckpointModel(Base):
    """Persisted agent message history for crash resumption + observability."""

    __tablename__ = "context_agent_checkpoints"

    batch_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("context_reconciliation_batches.id", ondelete="CASCADE"),
        primary_key=True,
    )
    messages_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    tool_call_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # Durable execution bookkeeping (Phase: durable agent execution log).
    # ``completed_event_ids`` lets a resumed run skip events the crashed run
    # already finished; ``last_seq`` continues the append-only execution log
    # without colliding with already-streamed records; ``chunk_index`` is the
    # 0-based chunk the crashed run was on (chunked batches).
    completed_event_ids: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    last_seq: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class ContextAgentExecutionLogModel(Base):
    """Append-only durable log of one batch's agent run.

    This is *both* the live stream the events screen tails and the durable
    history. One row per discrete record (tool call, mutation, …) or one
    row per coalesced model part (``text`` / ``thinking``) grown in place.

    ``seq`` is a per-batch monotonic cursor. Discrete records take the next
    seq and never change it. A coalesced part keeps its ``part_id`` but has
    its ``seq`` bumped to the latest on every flush, so a cursor-based tail
    (``seq > cursor``) re-delivers the growing part for free.
    """

    __tablename__ = "context_agent_execution_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("context_reconciliation_batches.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    record_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Set for coalesced model parts (text / thinking); NULL for discrete
    # records. Unique per batch so upserts grow the right row.
    part_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    done: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    # NULL = batch-level record; set when a record is attributable to one
    # event (mutation_applied / event_processed) so the per-event stream can
    # project it.
    event_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # Cursor-ordered replay/tail for one batch + idempotent discrete
        # append (a resumed run that re-issues a seq is a conflict no-op).
        UniqueConstraint(
            "batch_id",
            "seq",
            name="uq_context_agent_execution_log_batch_seq",
        ),
        # Grow-the-right-part upsert. Postgres treats NULLs as distinct so
        # this does not constrain discrete (part_id IS NULL) records.
        UniqueConstraint(
            "batch_id",
            "part_id",
            name="uq_context_agent_execution_log_batch_part",
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


class ContextEngineCostEvent(Base):
    """One LLM call attributable to the engine — cost telemetry (Phase 5)."""

    __tablename__ = "context_engine_cost_events"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    pot_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    event_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cost_usd: Mapped[object | None] = mapped_column(Numeric(12, 6), nullable=True)
    occurred_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    __table_args__ = (
        Index("ix_context_engine_cost_pot_kind_time", "pot_id", "kind", "occurred_at"),
    )


class ContextEngineDriftSnapshot(Base):
    """Per-pot drift signal aggregate (Phase 5)."""

    __tablename__ = "context_engine_drift_snapshots"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    pot_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    source_ref_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stale_ref_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    needs_verification_ref_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    verification_failed_ref_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    source_access_gap_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    missing_coverage_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    fallback_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    open_conflicts_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    captured_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    __table_args__ = (
        Index(
            "ix_context_engine_drift_pot_time", "pot_id", "captured_at"
        ),
    )
