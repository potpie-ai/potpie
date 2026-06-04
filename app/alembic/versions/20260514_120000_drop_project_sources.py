"""Drop legacy ``project_sources`` table.

The pot/source model in ``context_graph_pot_sources`` replaced this table
when the context engine moved to pot-scoped tenancy. The legacy code
paths (``integrations.application.project_sources_service``, the
``/sources/projects/...`` HTTP routes, the ``backfill_linear_source``
querier) were migrated to ``context_graph_pot_sources``; nothing reads or
writes ``project_sources`` any more.

Downgrade re-creates the table shape (without backfilling rows — the
original backfill seeded it from ``projects.repo_name``, which is no
longer the source of truth).
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "drop_project_sources_20260514"
down_revision: Union[str, None] = "int_uniq_per_user_20260514"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_project_sources_integration_id", table_name="project_sources")
    op.drop_index("ix_project_sources_project_id", table_name="project_sources")
    op.drop_table("project_sources")


def downgrade() -> None:
    op.create_table(
        "project_sources",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.Text(), nullable=False),
        sa.Column("integration_id", sa.String(length=255), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("source_kind", sa.String(length=64), nullable=False),
        sa.Column("scope_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("scope_hash", sa.String(length=64), nullable=False),
        sa.Column("sync_enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "sync_mode", sa.String(length=32), nullable=False, server_default="hybrid"
        ),
        sa.Column(
            "webhook_status",
            sa.String(length=32),
            nullable=False,
            server_default="not_applicable",
        ),
        sa.Column("last_sync_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "health_score", sa.Integer(), nullable=False, server_default="100"
        ),
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
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["integration_id"],
            ["integrations.integration_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "scope_hash", name="uq_project_sources_project_scope_hash"
        ),
    )
    op.create_index(
        "ix_project_sources_project_id", "project_sources", ["project_id"]
    )
    op.create_index(
        "ix_project_sources_integration_id", "project_sources", ["integration_id"]
    )
