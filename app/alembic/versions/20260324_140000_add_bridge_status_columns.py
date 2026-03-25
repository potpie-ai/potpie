"""Add bridge status tracking and entity_key to context_ingestion_log

Revision ID: 20260324_bridge_status_cols
Revises: 20260324_context_graph_tables
Create Date: 2026-03-24 14:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260324_bridge_status_cols"
down_revision: Union[str, None] = "20260324_context_graph_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "context_ingestion_log",
        sa.Column("entity_key", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "context_ingestion_log",
        sa.Column(
            "bridge_status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
    )
    op.add_column(
        "context_ingestion_log",
        sa.Column("bridge_error", sa.Text(), nullable=True),
    )
    op.add_column(
        "context_ingestion_log",
        sa.Column(
            "bridge_touched_by",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "context_ingestion_log",
        sa.Column(
            "bridge_modified_in",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "context_ingestion_log",
        sa.Column(
            "bridge_has_decision",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("context_ingestion_log", "bridge_has_decision")
    op.drop_column("context_ingestion_log", "bridge_modified_in")
    op.drop_column("context_ingestion_log", "bridge_touched_by")
    op.drop_column("context_ingestion_log", "bridge_error")
    op.drop_column("context_ingestion_log", "bridge_status")
    op.drop_column("context_ingestion_log", "entity_key")
