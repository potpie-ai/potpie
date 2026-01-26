"""add kg ingest run tables

Revision ID: 20251218121000
Revises: 20251217190000
Create Date: 2025-12-18 12:10:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20251218121000"
down_revision = "20251217190000"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "kg_ingest_runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("repo_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("commit_id", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.ForeignKeyConstraint(["repo_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.uid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index(op.f("ix_kg_ingest_runs_repo_id"), "kg_ingest_runs", ["repo_id"])
    op.create_index(op.f("ix_kg_ingest_runs_user_id"), "kg_ingest_runs", ["user_id"])
    op.create_index(
        op.f("ix_kg_ingest_runs_created_at"), "kg_ingest_runs", ["created_at"]
    )

    op.create_table(
        "kg_latest_successful_run",
        sa.Column("repo_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(["repo_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.uid"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["run_id"], ["kg_ingest_runs.run_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("repo_id", "user_id"),
    )
    op.create_index(
        op.f("ix_kg_latest_successful_run_run_id"),
        "kg_latest_successful_run",
        ["run_id"],
    )


def downgrade():
    op.drop_index(
        op.f("ix_kg_latest_successful_run_run_id"),
        table_name="kg_latest_successful_run",
    )
    op.drop_table("kg_latest_successful_run")
    op.drop_index(op.f("ix_kg_ingest_runs_created_at"), table_name="kg_ingest_runs")
    op.drop_index(op.f("ix_kg_ingest_runs_user_id"), table_name="kg_ingest_runs")
    op.drop_index(op.f("ix_kg_ingest_runs_repo_id"), table_name="kg_ingest_runs")
    op.drop_table("kg_ingest_runs")
