"""Narrow ``uq_context_reconciliation_batches_open_per_pot`` to ``status='pending'`` only.

Revision ID: ctx_narrow_open_batch_20260512
Revises: ctx_drop_debounce_20260512
Create Date: 2026-05-12

The original index covered ``status IN ('pending','claimed','running')``, which
contradicts the documented contract in ``admit_event`` /
``upsert_open_batch_for_pot``: events landing while a batch is *claimed* or
*running* are supposed to coalesce into a fresh *pending* batch. The wider
predicate makes that INSERT fail with a unique violation as soon as the
worker claims the first batch, so every subsequent ingest 500s.

The right invariant is "at most one *pending* batch per pot." A claimed or
running batch has already snapshotted its event list and must not block new
events from queuing up behind it.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "ctx_narrow_open_batch_20260512"
down_revision: Union[str, None] = "ctx_drop_debounce_20260512"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index(
        "uq_context_reconciliation_batches_open_per_pot",
        table_name="context_reconciliation_batches",
    )
    op.create_index(
        "uq_context_reconciliation_batches_open_per_pot",
        "context_reconciliation_batches",
        ["pot_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_context_reconciliation_batches_open_per_pot",
        table_name="context_reconciliation_batches",
    )
    op.create_index(
        "uq_context_reconciliation_batches_open_per_pot",
        "context_reconciliation_batches",
        ["pot_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending','claimed','running')"),
    )
