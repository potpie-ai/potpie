"""Single-agent ingestion: drop episode_steps, add batches + checkpoints

Revision ID: ctx_reco_batches_20260506
Revises: ctx_events_actor_20260422
Create Date: 2026-05-06

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "ctx_reco_batches_20260506"
down_revision: Union[str, None] = "ctx_events_actor_20260422"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("context_episode_steps")

    op.add_column(
        "context_graph_pots",
        sa.Column(
            "reconciliation_debounce_seconds",
            sa.Integer(),
            server_default=sa.text("10"),
            nullable=False,
        ),
    )

    op.create_table(
        "context_reconciliation_batches",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("next_ready_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("claimed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_context_reconciliation_batches_pot_status",
        "context_reconciliation_batches",
        ["pot_id", "status"],
    )
    op.create_index(
        "ix_context_reconciliation_batches_due",
        "context_reconciliation_batches",
        ["next_ready_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "uq_context_reconciliation_batches_open_per_pot",
        "context_reconciliation_batches",
        ["pot_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending','claimed','running')"),
    )

    op.create_table(
        "context_reconciliation_batch_events",
        sa.Column("batch_id", sa.String(length=255), nullable=False),
        sa.Column("event_id", sa.String(length=255), nullable=False),
        sa.Column(
            "added_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["context_reconciliation_batches.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["event_id"], ["context_events.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("batch_id", "event_id"),
    )
    op.create_index(
        "ix_context_reconciliation_batch_events_event_id",
        "context_reconciliation_batch_events",
        ["event_id"],
    )

    op.create_table(
        "context_agent_checkpoints",
        sa.Column("batch_id", sa.String(length=255), nullable=False),
        sa.Column(
            "messages_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "tool_call_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["context_reconciliation_batches.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("batch_id"),
    )


def downgrade() -> None:
    op.drop_table("context_agent_checkpoints")
    op.drop_index(
        "ix_context_reconciliation_batch_events_event_id",
        table_name="context_reconciliation_batch_events",
    )
    op.drop_table("context_reconciliation_batch_events")
    op.drop_index(
        "uq_context_reconciliation_batches_open_per_pot",
        table_name="context_reconciliation_batches",
    )
    op.drop_index(
        "ix_context_reconciliation_batches_due",
        table_name="context_reconciliation_batches",
    )
    op.drop_index(
        "ix_context_reconciliation_batches_pot_status",
        table_name="context_reconciliation_batches",
    )
    op.drop_table("context_reconciliation_batches")
    op.drop_column("context_graph_pots", "reconciliation_debounce_seconds")

    op.create_table(
        "context_episode_steps",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("event_id", sa.String(length=255), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("step_kind", sa.String(length=64), nullable=False),
        sa.Column(
            "step_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("applied_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
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
        sa.ForeignKeyConstraint(
            ["event_id"], ["context_events.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["context_reconciliation_runs.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "event_id",
            "sequence",
            name="uq_context_episode_steps_event_sequence",
        ),
    )
