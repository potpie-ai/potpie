"""context_graph_pots: user-owned pots independent of projects

Revision ID: ctx_pots_20260406
Revises: 20260406_project_sources
Create Date: 2026-04-06

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "ctx_pots_20260406"
down_revision = "20260406_project_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "context_graph_pots",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("primary_repo_name", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.uid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_context_graph_pots_user_id",
        "context_graph_pots",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_context_graph_pots_user_id", table_name="context_graph_pots")
    op.drop_table("context_graph_pots")
