"""Capture reconciliation agent work events

Revision ID: ctx_reco_work_events_20260421
Revises: pot_slug_unique_20260420
Create Date: 2026-04-21

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "ctx_reco_work_events_20260421"
down_revision: Union[str, None] = "pot_slug_unique_20260420"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "context_reconciliation_work_events",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("event_id", sa.String(length=255), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_kind", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
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
            ["run_id"], ["context_reconciliation_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "sequence",
            name="uq_context_reconciliation_work_events_run_sequence",
        ),
    )
    op.create_index(
        "ix_context_reconciliation_work_events_event_id",
        "context_reconciliation_work_events",
        ["event_id"],
        unique=False,
    )
    op.create_index(
        "ix_context_reconciliation_work_events_run_id",
        "context_reconciliation_work_events",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_context_reconciliation_work_events_event_run",
        "context_reconciliation_work_events",
        ["event_id", "run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_context_reconciliation_work_events_event_run",
        table_name="context_reconciliation_work_events",
    )
    op.drop_index(
        "ix_context_reconciliation_work_events_run_id",
        table_name="context_reconciliation_work_events",
    )
    op.drop_index(
        "ix_context_reconciliation_work_events_event_id",
        table_name="context_reconciliation_work_events",
    )
    op.drop_table("context_reconciliation_work_events")
