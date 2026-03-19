"""Add context_sync_state and context_ingestion_log tables for context graph Phase 1

Revision ID: 20260312_context_graph_sync
Revises: 20260226_add_tool_calls_thinking
Create Date: 2026-03-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260312_context_graph_sync"
down_revision: Union[str, None] = "20260226_add_tool_calls_thinking"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "context_sync_state",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("last_synced_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="idle"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "source_type", name="uq_context_sync_state_project_source"),
    )
    op.create_table(
        "context_ingestion_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("graphiti_episode_uuid", sa.Text(), nullable=False),
        sa.Column("ingested_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "source_type", "source_id",
            name="uq_context_ingestion_log_project_source_source_id",
        ),
    )


def downgrade() -> None:
    op.drop_table("context_ingestion_log")
    op.drop_table("context_sync_state")
