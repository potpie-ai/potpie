"""Add context graph state and ingestion tables

Revision ID: 20260324_context_graph_tables
Revises: 20260226_add_tool_calls_thinking
Create Date: 2026-03-24 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260324_context_graph_tables"
down_revision: Union[str, None] = "20260226_add_tool_calls_thinking"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "context_sync_state",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("project_id", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("last_synced_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
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
            "project_id",
            "source_type",
            name="uq_context_sync_project_source",
        ),
    )
    op.create_index(
        op.f("ix_context_sync_state_project_id"),
        "context_sync_state",
        ["project_id"],
        unique=False,
    )

    op.create_table(
        "context_ingestion_log",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("project_id", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("graphiti_episode_uuid", sa.String(length=255), nullable=True),
        sa.Column("bridge_written", sa.Boolean(), nullable=False, server_default="false"),
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
            "project_id",
            "source_type",
            "source_id",
            name="uq_context_ingestion_project_source_id",
        ),
    )
    op.create_index(
        op.f("ix_context_ingestion_log_project_id"),
        "context_ingestion_log",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_context_ingestion_project_source_id",
        "context_ingestion_log",
        ["project_id", "source_type", "source_id"],
        unique=False,
    )

    op.create_table(
        "raw_events",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("project_id", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "received_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "source_type",
            "source_id",
            name="uq_raw_events_source",
        ),
    )
    op.create_index(
        op.f("ix_raw_events_project_id"),
        "raw_events",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_raw_events_project_source_id",
        "raw_events",
        ["project_id", "source_type", "source_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_raw_events_project_source_id", table_name="raw_events")
    op.drop_index(op.f("ix_raw_events_project_id"), table_name="raw_events")
    op.drop_table("raw_events")

    op.drop_index(
        "ix_context_ingestion_project_source_id",
        table_name="context_ingestion_log",
    )
    op.drop_index(
        op.f("ix_context_ingestion_log_project_id"),
        table_name="context_ingestion_log",
    )
    op.drop_table("context_ingestion_log")

    op.drop_index(
        op.f("ix_context_sync_state_project_id"),
        table_name="context_sync_state",
    )
    op.drop_table("context_sync_state")
