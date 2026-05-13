"""Add feedback_votes table for blind Potpie-vs-Copilot human eval.

Revision ID: 20260513_feedback_votes
Revises: 20260226_add_tool_calls_thinking
Create Date: 2026-05-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260513_feedback_votes"
down_revision: Union[str, None] = "20260226_add_tool_calls_thinking"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback_votes",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("comparison_id", sa.String(length=128), nullable=False),
        sa.Column("chosen_model", sa.String(length=16), nullable=False),
        sa.Column("chosen_position", sa.String(length=8), nullable=False),
        sa.Column("presentation_seed", sa.Integer(), nullable=False),
        sa.Column("voter_name", sa.String(length=255), nullable=False),
        sa.Column("voter_email", sa.String(length=255), nullable=False),
        sa.Column("voter_role", sa.String(length=64), nullable=True),
        sa.Column("confidence", sa.Integer(), nullable=True),
        sa.Column(
            "reason_tags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("time_on_page_ms", sa.Integer(), nullable=True),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("client_user_agent", sa.Text(), nullable=True),
        sa.Column(
            "submitted_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "voter_email",
            "comparison_id",
            name="uq_feedback_votes_email_comparison",
        ),
    )
    op.create_index(
        "ix_feedback_votes_comparison_id",
        "feedback_votes",
        ["comparison_id"],
    )
    op.create_index(
        "ix_feedback_votes_voter_email",
        "feedback_votes",
        ["voter_email"],
    )
    op.create_index(
        "ix_feedback_votes_session_id",
        "feedback_votes",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_votes_session_id", table_name="feedback_votes")
    op.drop_index("ix_feedback_votes_voter_email", table_name="feedback_votes")
    op.drop_index("ix_feedback_votes_comparison_id", table_name="feedback_votes")
    op.drop_table("feedback_votes")
