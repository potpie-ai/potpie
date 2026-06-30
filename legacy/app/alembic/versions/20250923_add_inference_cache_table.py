"""Add inference cache table for content hash-based caching

Revision ID: 20250923_add_inference_cache
Revises: 20250911184844
Create Date: 2025-09-23 15:30:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "20250923_add_inference_cache"
down_revision = "20250911184844"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "inference_cache",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("project_id", sa.Text(), nullable=True),
        sa.Column("node_type", sa.String(50), nullable=True),
        sa.Column("content_length", sa.Integer(), nullable=True),
        sa.Column(
            "inference_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("embedding_vector", sa.ARRAY(sa.Float), nullable=True),
        sa.Column("tags", sa.ARRAY(sa.Text), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "last_accessed",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("access_count", sa.Integer(), server_default="1", nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_hash"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )

    # Create indexes for performance
    op.create_index(
        op.f("ix_inference_cache_content_hash"), "inference_cache", ["content_hash"]
    )
    op.create_index(
        op.f("ix_inference_cache_project_id"), "inference_cache", ["project_id"]
    )
    op.create_index(
        op.f("ix_inference_cache_created_at"), "inference_cache", ["created_at"]
    )
    op.create_index(
        op.f("ix_inference_cache_last_accessed"), "inference_cache", ["last_accessed"]
    )


def downgrade():
    op.drop_index(
        op.f("ix_inference_cache_last_accessed"), table_name="inference_cache"
    )
    op.drop_index(op.f("ix_inference_cache_created_at"), table_name="inference_cache")
    op.drop_index(op.f("ix_inference_cache_project_id"), table_name="inference_cache")
    op.drop_index(op.f("ix_inference_cache_content_hash"), table_name="inference_cache")
    op.drop_table("inference_cache")
