"""Add project_sources for multi-source context sync (GitHub + Linear).

Revision ID: 20260406_project_sources
Revises: ctx_graph_ledger_v1
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

revision: str = "20260406_project_sources"
down_revision: Union[str, None] = "ctx_graph_ledger_v1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _scope_hash(scope: dict) -> str:
    return hashlib.sha256(
        json.dumps(scope, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def upgrade() -> None:
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

    conn = op.get_bind()
    rows = conn.execute(
        text(
            """
            SELECT id, repo_name, status
            FROM projects
            WHERE repo_name IS NOT NULL
              AND trim(repo_name) <> ''
              AND lower(status) = 'ready'
            """
        )
    ).fetchall()
    for project_id, repo_name, _status in rows:
        scope = {"repo_name": repo_name}
        h = _scope_hash(scope)
        exists = conn.execute(
            text(
                "SELECT 1 FROM project_sources WHERE project_id = :pid AND scope_hash = :h"
            ),
            {"pid": project_id, "h": h},
        ).first()
        if exists:
            continue
        sid = str(uuid.uuid4())
        conn.execute(
            text(
                """
                INSERT INTO project_sources (
                    id, project_id, integration_id, provider, source_kind,
                    scope_json, scope_hash, sync_enabled, sync_mode, webhook_status,
                    health_score, created_at, updated_at
                ) VALUES (
                    :id, :project_id, NULL, 'github', 'repository',
                    CAST(:scope_json AS jsonb), :scope_hash, true, 'hybrid', 'not_applicable',
                    100, now(), now()
                )
                """
            ),
            {
                "id": sid,
                "project_id": project_id,
                "scope_json": json.dumps(scope),
                "scope_hash": h,
            },
        )


def downgrade() -> None:
    op.drop_index("ix_project_sources_integration_id", table_name="project_sources")
    op.drop_index("ix_project_sources_project_id", table_name="project_sources")
    op.drop_table("project_sources")
