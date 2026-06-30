"""Pot owner/user role normalization, invitations, and pot sources

Revision ID: pot_inv_src_20260420
Revises: pot_tenancy_20260407
Create Date: 2026-04-20

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "pot_inv_src_20260420"
down_revision = "pot_tenancy_20260407"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Collapse legacy roles (admin, read_only) to the canonical ``user`` role.
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "UPDATE context_graph_pot_members SET role = 'user' "
            "WHERE role IN ('admin', 'read_only')"
        )
    )

    # 2. Invitation persistence — owner invites by email, accept creates a member.
    op.create_table(
        "context_graph_pot_invitations",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False, server_default="user"),
        sa.Column("invited_by_user_id", sa.String(length=255), nullable=False),
        sa.Column("accepted_by_user_id", sa.String(length=255), nullable=True),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("accepted_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["pot_id"], ["context_graph_pots.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["invited_by_user_id"], ["users.uid"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["accepted_by_user_id"], ["users.uid"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_context_graph_pot_invitations_pot_id",
        "context_graph_pot_invitations",
        ["pot_id"],
    )
    op.create_index(
        "ix_context_graph_pot_invitations_email",
        "context_graph_pot_invitations",
        ["email"],
    )
    op.create_index(
        "ix_context_graph_pot_invitations_token_hash",
        "context_graph_pot_invitations",
        ["token_hash"],
        unique=True,
    )
    if conn.dialect.name == "postgresql":
        op.create_index(
            "ix_context_graph_pot_invitations_pending_email",
            "context_graph_pot_invitations",
            ["pot_id", "email"],
            unique=True,
            postgresql_where=sa.text("status = 'pending'"),
        )

    # 3. Pot sources — the ingestable data-scope layer (repo/team/channel/...).
    op.create_table(
        "context_graph_pot_sources",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("integration_id", sa.String(length=255), nullable=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column("scope_json", sa.Text(), nullable=True),
        sa.Column("scope_hash", sa.String(length=128), nullable=False),
        sa.Column("sync_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("sync_mode", sa.Text(), nullable=True),
        sa.Column("webhook_status", sa.Text(), nullable=True),
        sa.Column("last_sync_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("health_score", sa.Text(), nullable=True),
        sa.Column("added_by_user_id", sa.String(length=255), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["pot_id"], ["context_graph_pots.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["added_by_user_id"], ["users.uid"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "pot_id", "provider", "source_kind", "scope_hash",
            name="uq_context_graph_pot_source_scope",
        ),
    )
    op.create_index(
        "ix_context_graph_pot_sources_pot_id",
        "context_graph_pot_sources",
        ["pot_id"],
    )
    op.create_index(
        "ix_context_graph_pot_sources_provider",
        "context_graph_pot_sources",
        ["provider", "source_kind"],
    )

    # 4. Mirror every existing pot repository row into the new sources table.
    #    Scope hash format is deterministic so the UI dedupes across provider/owner/repo.
    conn.execute(
        sa.text(
            """
            INSERT INTO context_graph_pot_sources (
                id, pot_id, integration_id, provider, source_kind,
                scope_json, scope_hash, sync_enabled, added_by_user_id,
                created_at, updated_at
            )
            SELECT
                gen_random_uuid()::text,
                r.pot_id,
                NULL,
                r.provider,
                'repository',
                json_build_object(
                    'owner', r.owner,
                    'repo', r.repo,
                    'repo_name', r.owner || '/' || r.repo,
                    'provider_host', r.provider_host,
                    'external_repo_id', r.external_repo_id,
                    'remote_url', r.remote_url,
                    'default_branch', r.default_branch
                )::text,
                md5(r.provider || '|' || r.provider_host || '|' || lower(r.owner) || '|' || lower(r.repo)),
                true,
                r.added_by_user_id,
                r.created_at,
                r.created_at
            FROM context_graph_pot_repositories r
            WHERE NOT EXISTS (
                SELECT 1
                FROM context_graph_pot_sources s
                WHERE s.pot_id = r.pot_id
                  AND s.provider = r.provider
                  AND s.source_kind = 'repository'
                  AND s.scope_hash = md5(r.provider || '|' || r.provider_host || '|' || lower(r.owner) || '|' || lower(r.repo))
            )
            """
        )
    ) if conn.dialect.name == "postgresql" else None


def downgrade() -> None:
    op.drop_index(
        "ix_context_graph_pot_sources_provider",
        table_name="context_graph_pot_sources",
    )
    op.drop_index(
        "ix_context_graph_pot_sources_pot_id",
        table_name="context_graph_pot_sources",
    )
    op.drop_table("context_graph_pot_sources")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.drop_index(
            "ix_context_graph_pot_invitations_pending_email",
            table_name="context_graph_pot_invitations",
            postgresql_where=sa.text("status = 'pending'"),
        )
    op.drop_index(
        "ix_context_graph_pot_invitations_token_hash",
        table_name="context_graph_pot_invitations",
    )
    op.drop_index(
        "ix_context_graph_pot_invitations_email",
        table_name="context_graph_pot_invitations",
    )
    op.drop_index(
        "ix_context_graph_pot_invitations_pot_id",
        table_name="context_graph_pot_invitations",
    )
    op.drop_table("context_graph_pot_invitations")
