"""Per-pot ingestion config (windowed batching by default).

Revision ID: ctx_pot_ingest_config_20260515
Revises: ctx_narrow_open_batch_20260512
Create Date: 2026-05-15

Adds ``context_pot_ingestion_config`` so each pot can opt into immediate or
windowed batching. Default is ``windowed`` with a 5-minute window — this is
a *behaviour change*: events ingest in 5-minute batches now instead of
firing the agent on every event. The "Queued: N ⚡" UI control gives the
user a one-click force-flush at any time, and switching a pot back to
``immediate`` is a single ``PUT /pots/{pot_id}/ingestion-config`` call.

Existing pots are seeded with the platform default so legacy behaviour
isn't preserved by accident — operators who want zero-latency ingestion
should set ``mode='immediate'`` per pot.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "ctx_pot_ingest_config_20260515"
down_revision: Union[str, None] = "drop_project_sources_20260514"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "context_pot_ingestion_config",
        sa.Column("pot_id", sa.Text(), primary_key=True),
        sa.Column(
            "mode",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'windowed'"),
        ),
        sa.Column(
            "window_minutes",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("5"),
        ),
        sa.Column("min_batch_size", sa.Integer(), nullable=True),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("updated_by_user_id", sa.String(length=255), nullable=True),
        sa.CheckConstraint(
            "mode IN ('immediate', 'windowed')",
            name="ck_context_pot_ingestion_config_mode",
        ),
        sa.CheckConstraint(
            "window_minutes >= 1 AND window_minutes <= 1440",
            name="ck_context_pot_ingestion_config_window_minutes",
        ),
        sa.CheckConstraint(
            "min_batch_size IS NULL OR min_batch_size >= 1",
            name="ck_context_pot_ingestion_config_min_batch_size",
        ),
    )

    # Index for the windowed-flush join — narrow predicate to avoid
    # bloating the index with immediate-mode pots that the flusher never
    # touches.
    op.create_index(
        "ix_context_pot_ingestion_config_windowed",
        "context_pot_ingestion_config",
        ["pot_id"],
        postgresql_where=sa.text("mode = 'windowed'"),
    )

    # Seed every existing pot at the platform default. We can't reach the
    # ``pots`` table from this migration generically (some hosts use a
    # different table name); instead we insert one row per *distinct
    # pot_id* observed in ``context_events`` (the source of truth for
    # active pots). New pots created after this point pick up the default
    # via lazy creation in the adapter.
    op.execute(
        """
        INSERT INTO context_pot_ingestion_config (pot_id)
        SELECT DISTINCT pot_id
        FROM context_events
        ON CONFLICT (pot_id) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index(
        "ix_context_pot_ingestion_config_windowed",
        table_name="context_pot_ingestion_config",
    )
    op.drop_table("context_pot_ingestion_config")
