"""Drop debounce artifacts: next_ready_at NOT NULL, due-index, per-pot debounce column

Revision ID: ctx_drop_debounce_20260512
Revises: ctx_reco_batches_20260506
Create Date: 2026-05-12

The reconciliation dispatcher is now event-triggered (``jobs.enqueue_batch``
fires from ``admit_event``), so:

- The partial index ``ix_context_reconciliation_batches_due`` over
  ``next_ready_at`` is no longer queried.
- ``next_ready_at`` is no longer read or written — make the column nullable
  rather than drop it outright (keeps the rollback path cheap).
- ``context_graph_pots.reconciliation_debounce_seconds`` is no longer read.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "ctx_drop_debounce_20260512"
down_revision: Union[str, None] = "ctx_reco_batches_20260506"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index(
        "ix_context_reconciliation_batches_due",
        table_name="context_reconciliation_batches",
    )
    op.alter_column(
        "context_reconciliation_batches",
        "next_ready_at",
        existing_type=sa.TIMESTAMP(timezone=True),
        nullable=True,
    )
    op.drop_column("context_graph_pots", "reconciliation_debounce_seconds")


def downgrade() -> None:
    op.add_column(
        "context_graph_pots",
        sa.Column(
            "reconciliation_debounce_seconds",
            sa.Integer(),
            server_default=sa.text("10"),
            nullable=False,
        ),
    )
    op.alter_column(
        "context_reconciliation_batches",
        "next_ready_at",
        existing_type=sa.TIMESTAMP(timezone=True),
        nullable=False,
    )
    op.create_index(
        "ix_context_reconciliation_batches_due",
        "context_reconciliation_batches",
        ["next_ready_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )
