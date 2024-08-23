"""Changes for implementation of conversations

Revision ID: 20240820182032_d3f532773223
Revises: 20240813145447_56e7763c7d20
Create Date: 2024-08-20 18:20:32.408674

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import ENUM

# revision identifiers, used by Alembic.
revision: str = "20240820182032_d3f532773223"
down_revision: Union[str, None] = "20240813145447_56e7763c7d20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

message_status_enum = ENUM(
    "ACTIVE", "ARCHIVED", "DELETED", name="message_status_enum", create_type=False
)


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Add the SYSTEM_GENERATED value to the message type enum
    op.execute("ALTER TYPE messagetype ADD VALUE 'SYSTEM_GENERATED'")

    # Commit to ensure the new enum value is recognized
    op.execute("COMMIT")

    # Drop the old foreign key constraint and create a new one with ON DELETE CASCADE
    op.drop_constraint(
        "conversations_user_id_fkey", "conversations", type_="foreignkey"
    )
    op.create_foreign_key(
        None, "conversations", "users", ["user_id"], ["uid"], ondelete="CASCADE"
    )

    # Drop the agent_ids column from conversations table
    op.drop_column("conversations", "agent_ids")

    # Drop the existing check constraint for sender_id
    op.drop_constraint("check_sender_id_for_type", "messages", type_="check")

    # Create a new check constraint with the correct logic
    op.create_check_constraint(
        "check_sender_id_for_type",
        "messages",
        "((type = 'HUMAN'::messagetype AND sender_id IS NOT NULL) OR "
        "(type IN ('AI_GENERATED'::messagetype, 'SYSTEM_GENERATED'::messagetype) AND sender_id IS NULL))",
    )

    op.alter_column(
        "projects",
        "updated_at",
        existing_type=postgresql.TIMESTAMP(),
        type_=sa.TIMESTAMP(timezone=True),
        existing_nullable=True,
    )
    op.alter_column(
        "users",
        "last_login_at",
        existing_type=postgresql.TIMESTAMP(),
        type_=sa.TIMESTAMP(timezone=True),
        existing_nullable=True,
    )

    # Create ENUM type in the database
    message_status_enum.create(op.get_bind(), checkfirst=True)

    # Add new column using the ENUM type
    op.add_column(
        "messages",
        sa.Column(
            "status", message_status_enum, nullable=False, server_default="ACTIVE"
        ),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Re-add the agent_ids column to conversations table
    op.add_column(
        "conversations",
        sa.Column(
            "agent_ids",
            postgresql.ARRAY(sa.VARCHAR()),
            autoincrement=False,
            nullable=False,
        ),
    )

    # Drop the new foreign key constraint and re-create the old one
    op.drop_constraint(None, "conversations", type_="foreignkey")
    op.create_foreign_key(
        "conversations_user_id_fkey", "conversations", "users", ["user_id"], ["uid"]
    )

    # Drop the new check constraint and re-create the old one
    op.drop_constraint("check_sender_id_for_type", "messages", type_="check")
    op.create_check_constraint(
        "check_sender_id_for_type",
        "messages",
        "((type = 'HUMAN'::messagetype AND sender_id IS NOT NULL) OR "
        "(type = 'AI_GENERATED'::messagetype AND sender_id IS NULL))",
    )
    op.alter_column(
        "users",
        "last_login_at",
        existing_type=sa.TIMESTAMP(timezone=True),
        type_=postgresql.TIMESTAMP(),
        existing_nullable=True,
    )
    op.alter_column(
        "projects",
        "updated_at",
        existing_type=sa.TIMESTAMP(timezone=True),
        type_=postgresql.TIMESTAMP(),
        existing_nullable=True,
    )

    # Drop the column
    op.drop_column("messages", "status")

    # Drop the ENUM type if it is no longer used
    message_status_enum.drop(op.get_bind(), checkfirst=False)
    # ### end Alembic commands ###
