"""Durable agent execution log + resume bookkeeping.

Revision ID: ctx_agent_exec_log_20260515
Revises: ctx_pot_ingest_config_20260515
Create Date: 2026-05-15

Introduces ``context_agent_execution_log`` — the single append-only,
durable record of a reconciliation batch's agent run. It is *both* the
live stream the events screen tails (text / thinking / tool calls / tool
results / graph mutations) and the durable history.

Also extends ``context_agent_checkpoints`` with durable execution
bookkeeping (``completed_event_ids``, ``last_seq``, ``chunk_index``) so a
worker crash mid-run resumes with full context instead of restarting or
redoing completed events.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "ctx_agent_exec_log_20260515"
down_revision: Union[str, None] = "ctx_pot_ingest_config_20260515"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "context_agent_execution_log",
        sa.Column(
            "id", sa.BigInteger(), primary_key=True, autoincrement=True
        ),
        sa.Column(
            "batch_id",
            sa.String(length=255),
            sa.ForeignKey(
                "context_reconciliation_batches.id", ondelete="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("record_type", sa.String(length=32), nullable=False),
        sa.Column("part_id", sa.String(length=128), nullable=True),
        sa.Column(
            "done",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("event_id", sa.String(length=255), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        # Cursor-ordered replay/tail + idempotent discrete append.
        sa.UniqueConstraint(
            "batch_id",
            "seq",
            name="uq_context_agent_execution_log_batch_seq",
        ),
        # Grow-the-right-part upsert (NULLs distinct → discrete rows
        # unconstrained).
        sa.UniqueConstraint(
            "batch_id",
            "part_id",
            name="uq_context_agent_execution_log_batch_part",
        ),
    )

    with op.batch_alter_table("context_agent_checkpoints") as batch:
        batch.add_column(
            sa.Column(
                "completed_event_ids",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'[]'::jsonb"),
            )
        )
        batch.add_column(
            sa.Column(
                "last_seq",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch.add_column(
            sa.Column(
                "chunk_index",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("context_agent_checkpoints") as batch:
        batch.drop_column("chunk_index")
        batch.drop_column("last_seq")
        batch.drop_column("completed_event_ids")

    op.drop_table("context_agent_execution_log")
