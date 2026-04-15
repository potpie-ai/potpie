"""Add tool_calls and thinking columns to messages table

Revision ID: 20260226_add_tool_calls_thinking
Revises: 20260205_ensure_inferring
Create Date: 2026-02-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260226_add_tool_calls_thinking"
down_revision: Union[str, None] = "20260205_ensure_inferring"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add tool_calls column (JSON type to store list of tool call objects)
    op.add_column(
        "messages",
        sa.Column("tool_calls", postgresql.JSON(astext_type=sa.Text()), nullable=True),
    )
    # Add thinking column (Text type to store reasoning/thinking content)
    op.add_column(
        "messages",
        sa.Column("thinking", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    # Remove the columns in reverse order
    op.drop_column("messages", "thinking")
    op.drop_column("messages", "tool_calls")
