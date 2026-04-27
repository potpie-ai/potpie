"""Context graph ledger: final schema (pot-scoped sync, context_events, runs, episode steps).

Single revision replacing former 20260324–20260403 + lean-ledger migrations.
Creates: context_sync_state, context_events (incl. merged-PR / bridge columns),
context_reconciliation_runs (with plan_json), context_episode_steps.

Revision ID: ctx_graph_ledger_v1
Revises: 20260226_add_tool_calls_thinking
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "ctx_graph_ledger_v1"
down_revision: Union[str, None] = "20260226_add_tool_calls_thinking"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "context_sync_state",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False, server_default="github"),
        sa.Column("provider_host", sa.String(length=255), nullable=False, server_default="github.com"),
        sa.Column("repo_name", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("last_synced_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="idle"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "pot_id",
            "provider",
            "provider_host",
            "repo_name",
            "source_type",
            name="uq_context_sync_pot_repo_source",
        ),
    )
    op.create_index(op.f("ix_context_sync_state_pot_id"), "context_sync_state", ["pot_id"], unique=False)

    op.create_table(
        "context_events",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False, server_default="github"),
        sa.Column("provider_host", sa.String(length=255), nullable=False, server_default="github.com"),
        sa.Column("repo_name", sa.Text(), nullable=False),
        sa.Column("source_system", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("source_event_id", sa.String(length=512), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("occurred_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "received_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="received"),
        sa.Column(
            "ingestion_kind",
            sa.String(length=64),
            nullable=False,
            server_default="agent_reconciliation",
        ),
        sa.Column("job_id", sa.String(length=255), nullable=True),
        sa.Column("correlation_id", sa.String(length=255), nullable=True),
        sa.Column("idempotency_key", sa.String(length=512), nullable=True),
        sa.Column("source_channel", sa.String(length=64), nullable=False, server_default="unknown"),
        sa.Column("dedup_key", sa.String(length=512), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("step_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("step_done", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("step_error", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("event_error", sa.Text(), nullable=True),
        sa.Column(
            "event_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("graphiti_episode_uuid", sa.String(length=255), nullable=True),
        sa.Column("entity_key", sa.String(length=512), nullable=True),
        sa.Column("bridge_written", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("bridge_status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("bridge_error", sa.Text(), nullable=True),
        sa.Column("bridge_touched_by", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("bridge_modified_in", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("bridge_has_decision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("raw_processed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "pot_id",
            "provider",
            "provider_host",
            "repo_name",
            "source_system",
            "source_id",
            name="uq_context_events_dedupe",
        ),
    )
    op.create_index("ix_context_events_pot_status", "context_events", ["pot_id", "status"], unique=False)

    op.create_table(
        "context_reconciliation_runs",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("event_id", sa.String(length=255), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="running"),
        sa.Column("agent_name", sa.String(length=255), nullable=True),
        sa.Column("agent_version", sa.String(length=64), nullable=True),
        sa.Column("toolset_version", sa.String(length=64), nullable=True),
        sa.Column("plan_summary", sa.Text(), nullable=True),
        sa.Column("episode_count", sa.Integer(), nullable=True),
        sa.Column("entity_mutation_count", sa.Integer(), nullable=True),
        sa.Column("edge_mutation_count", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("plan_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["event_id"], ["context_events.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id", "attempt_number", name="uq_context_reconciliation_runs_event_attempt"),
    )
    op.create_index(
        "ix_context_reconciliation_runs_event_id",
        "context_reconciliation_runs",
        ["event_id"],
        unique=False,
    )

    op.create_table(
        "context_episode_steps",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("event_id", sa.String(length=255), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("step_kind", sa.String(length=64), nullable=False),
        sa.Column("step_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("applied_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("queued_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("run_id", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["event_id"], ["context_events.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["context_reconciliation_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id", "sequence", name="uq_context_episode_steps_event_sequence"),
    )
    op.create_index(
        "ix_context_episode_steps_event_id",
        "context_episode_steps",
        ["event_id"],
        unique=False,
    )
    op.create_index(
        "ix_context_episode_steps_pot_id",
        "context_episode_steps",
        ["pot_id"],
        unique=False,
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_context_episode_steps_pot_status "
        "ON context_episode_steps (pot_id, status)"
    )

    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_context_events_pot_idempotency
        ON context_events (pot_id, ingestion_kind, idempotency_key)
        WHERE idempotency_key IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_context_events_pot_dedup_kind
        ON context_events (pot_id, ingestion_kind, dedup_key)
        WHERE dedup_key IS NOT NULL
        """
    )

    # Match ORM: no server defaults on bridge counters / booleans after table exists.
    for col, typ in (
        ("bridge_written", sa.Boolean()),
        ("bridge_status", sa.String(length=32)),
        ("bridge_touched_by", sa.Integer()),
        ("bridge_modified_in", sa.Integer()),
        ("bridge_has_decision", sa.Integer()),
    ):
        op.alter_column(
            "context_events",
            col,
            server_default=None,
            existing_type=typ,
            existing_nullable=False,
        )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_context_events_pot_dedup_kind")
    op.execute("DROP INDEX IF EXISTS uq_context_events_pot_idempotency")
    op.execute("DROP INDEX IF EXISTS ix_context_episode_steps_pot_status")
    op.drop_index("ix_context_episode_steps_pot_id", table_name="context_episode_steps")
    op.drop_index("ix_context_episode_steps_event_id", table_name="context_episode_steps")
    op.drop_table("context_episode_steps")
    op.drop_index("ix_context_reconciliation_runs_event_id", table_name="context_reconciliation_runs")
    op.drop_table("context_reconciliation_runs")
    op.drop_index("ix_context_events_pot_status", table_name="context_events")
    op.drop_table("context_events")
    op.drop_index(op.f("ix_context_sync_state_pot_id"), table_name="context_sync_state")
    op.drop_table("context_sync_state")
