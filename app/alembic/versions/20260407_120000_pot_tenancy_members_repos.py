"""Pot tenancy: members, repositories, integrations, pot metadata

Revision ID: pot_tenancy_20260407
Revises: ctx_pots_20260406
Create Date: 2026-04-07

"""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from alembic import op

revision = "pot_tenancy_20260407"
down_revision = "ctx_pots_20260406"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "context_graph_pots",
        sa.Column("created_by_user_id", sa.String(length=255), nullable=True),
    )
    op.add_column("context_graph_pots", sa.Column("slug", sa.Text(), nullable=True))
    op.add_column(
        "context_graph_pots",
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.add_column(
        "context_graph_pots",
        sa.Column("archived_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_context_graph_pots_created_by_user_id",
        "context_graph_pots",
        "users",
        ["created_by_user_id"],
        ["uid"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_context_graph_pots_created_by_user_id",
        "context_graph_pots",
        ["created_by_user_id"],
        unique=False,
    )

    op.create_table(
        "context_graph_pot_members",
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("invited_by_user_id", sa.String(length=255), nullable=True),
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
        sa.ForeignKeyConstraint(["pot_id"], ["context_graph_pots.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.uid"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["invited_by_user_id"],
            ["users.uid"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("pot_id", "user_id"),
    )
    op.create_index(
        "ix_context_graph_pot_members_user_id",
        "context_graph_pot_members",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "context_graph_pot_repositories",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("provider_host", sa.Text(), nullable=False),
        sa.Column("owner", sa.Text(), nullable=False),
        sa.Column("repo", sa.Text(), nullable=False),
        sa.Column("external_repo_id", sa.Text(), nullable=True),
        sa.Column("remote_url", sa.Text(), nullable=True),
        sa.Column("default_branch", sa.Text(), nullable=True),
        sa.Column("added_by_user_id", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["pot_id"], ["context_graph_pots.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["added_by_user_id"],
            ["users.uid"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "pot_id",
            "provider",
            "provider_host",
            "owner",
            "repo",
            name="uq_context_graph_pot_repository",
        ),
    )
    op.create_index(
        "ix_context_graph_pot_repositories_pot_id",
        "context_graph_pot_repositories",
        ["pot_id"],
        unique=False,
    )

    op.create_table(
        "context_graph_pot_integrations",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("pot_id", sa.Text(), nullable=False),
        sa.Column("integration_type", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("provider_host", sa.Text(), nullable=True),
        sa.Column("external_account_id", sa.Text(), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("created_by_user_id", sa.String(length=255), nullable=False),
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
        sa.ForeignKeyConstraint(["pot_id"], ["context_graph_pots.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.uid"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_context_graph_pot_integrations_pot_id",
        "context_graph_pot_integrations",
        ["pot_id"],
        unique=False,
    )

    conn = op.get_bind()
    conn.execute(
        sa.text(
            "UPDATE context_graph_pots SET created_by_user_id = user_id "
            "WHERE created_by_user_id IS NULL"
        )
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO context_graph_pot_members (pot_id, user_id, role, created_at, updated_at)
            SELECT id, user_id, 'owner', created_at, created_at
            FROM context_graph_pots
            """
        )
    )

    rows = conn.execute(
        sa.text(
            "SELECT id, user_id, primary_repo_name, created_at FROM context_graph_pots "
            "WHERE primary_repo_name IS NOT NULL AND trim(primary_repo_name) <> ''"
        )
    ).fetchall()
    for pot_id, uid, primary_repo_name, created_at in rows:
        raw = (primary_repo_name or "").strip()
        if "/" not in raw:
            continue
        owner, name = raw.split("/", 1)
        owner = owner.strip()
        name = name.strip()
        if not owner or not name:
            continue
        exists = conn.execute(
            sa.text(
                "SELECT 1 FROM context_graph_pot_repositories WHERE pot_id = :pid "
                "AND provider = 'github' AND provider_host = 'github.com' "
                "AND owner = :owner AND repo = :repo"
            ),
            {"pid": pot_id, "owner": owner, "repo": name},
        ).scalar()
        if exists:
            continue
        rid = str(uuid.uuid4())
        conn.execute(
            sa.text(
                """
                INSERT INTO context_graph_pot_repositories (
                    id, pot_id, provider, provider_host, owner, repo,
                    added_by_user_id, created_at
                )
                VALUES (
                    :id, :pot_id, 'github', 'github.com', :owner, :repo,
                    :uid, :created_at
                )
                """
            ),
            {
                "id": rid,
                "pot_id": pot_id,
                "owner": owner,
                "repo": name,
                "uid": uid,
                "created_at": created_at,
            },
        )

    if conn.dialect.name == "postgresql":
        op.create_index(
            "ix_context_graph_pots_user_slug",
            "context_graph_pots",
            ["created_by_user_id", "slug"],
            unique=True,
            postgresql_where=sa.text("slug IS NOT NULL"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.drop_index(
            "ix_context_graph_pots_user_slug",
            table_name="context_graph_pots",
            postgresql_where=sa.text("slug IS NOT NULL"),
        )
    op.drop_table("context_graph_pot_integrations")
    op.drop_table("context_graph_pot_repositories")
    op.drop_table("context_graph_pot_members")
    op.drop_constraint("fk_context_graph_pots_created_by_user_id", "context_graph_pots", type_="foreignkey")
    op.drop_index("ix_context_graph_pots_created_by_user_id", table_name="context_graph_pots")
    op.drop_column("context_graph_pots", "archived_at")
    op.drop_column("context_graph_pots", "updated_at")
    op.drop_column("context_graph_pots", "slug")
    op.drop_column("context_graph_pots", "created_by_user_id")
