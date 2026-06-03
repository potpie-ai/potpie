"""Make context pot slugs globally unique

Revision ID: pot_slug_unique_20260420
Revises: pot_inv_src_20260420
Create Date: 2026-04-20

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "pot_slug_unique_20260420"
down_revision = "pot_inv_src_20260420"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        conn.execute(
            sa.text(
                """
                UPDATE context_graph_pots
                SET slug = NULL
                WHERE slug IS NOT NULL AND btrim(slug) = ''
                """
            )
        )
        conn.execute(
            sa.text(
                """
                UPDATE context_graph_pots
                SET slug = lower(btrim(slug))
                WHERE slug IS NOT NULL
                """
            )
        )
        conn.execute(
            sa.text(
                """
                WITH ranked AS (
                    SELECT
                        id,
                        slug,
                        row_number() OVER (
                            PARTITION BY slug
                            ORDER BY created_at ASC, id ASC
                        ) AS rn
                    FROM context_graph_pots
                    WHERE slug IS NOT NULL
                )
                UPDATE context_graph_pots p
                SET slug = ranked.slug || '-' || substring(p.id from 1 for 8)
                FROM ranked
                WHERE p.id = ranked.id AND ranked.rn > 1
                """
            )
        )
        op.create_index(
            "ix_context_graph_pots_slug_unique",
            "context_graph_pots",
            ["slug"],
            unique=True,
            postgresql_where=sa.text("slug IS NOT NULL"),
        )
        return

    op.create_index(
        "ix_context_graph_pots_slug_unique",
        "context_graph_pots",
        ["slug"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_context_graph_pots_slug_unique", table_name="context_graph_pots")
