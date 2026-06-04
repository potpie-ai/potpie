"""Add actor_* columns to context_events

Revision ID: ctx_events_actor_20260422
Revises: ctx_reco_work_events_20260421
Create Date: 2026-04-22

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "ctx_events_actor_20260422"
down_revision: Union[str, None] = "ctx_reco_work_events_20260421"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "context_events",
        sa.Column("actor_user_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "context_events",
        sa.Column("actor_surface", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "context_events",
        sa.Column("actor_client_name", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "context_events",
        sa.Column("actor_auth_method", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_context_events_pot_actor",
        "context_events",
        ["pot_id", "actor_user_id"],
    )
    op.create_index(
        "ix_context_events_pot_surface",
        "context_events",
        ["pot_id", "actor_surface"],
    )


def downgrade() -> None:
    op.drop_index("ix_context_events_pot_surface", table_name="context_events")
    op.drop_index("ix_context_events_pot_actor", table_name="context_events")
    op.drop_column("context_events", "actor_auth_method")
    op.drop_column("context_events", "actor_client_name")
    op.drop_column("context_events", "actor_surface")
    op.drop_column("context_events", "actor_user_id")
