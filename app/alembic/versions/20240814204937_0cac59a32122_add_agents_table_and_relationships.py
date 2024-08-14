"""Add agents table and relationships, and add SYSTEM_GENERATED to MessageType

Revision ID: 20240814204937_0cac59a32122
Revises: 20240813145447_56e7763c7d20
Create Date: 2024-08-14 20:49:37.840460

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20240814204937_0cac59a32122'
down_revision = '20240813145447_56e7763c7d20'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Add the new enum value to the MessageType enum
    op.execute("ALTER TYPE messagetype ADD VALUE 'SYSTEM_GENERATED'")

    # Create the agents table
    op.create_table('agents',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.String(length=1024), nullable=True),
        sa.Column('provider', sa.Enum('LANGCHAIN', 'CUSTOM', 'CREWAI', name='providertype'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('status', sa.Enum('ACTIVE', 'INACTIVE', 'DEPRECATED', name='agentstatus'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    # Create the conversation_agents association table
    op.create_table('conversation_agents',
        sa.Column('conversation_id', sa.String(length=255), nullable=True),
        sa.Column('agent_id', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE')
    )

    # Drop the agent_ids column from the conversations table
    op.drop_column('conversations', 'agent_ids')


def downgrade() -> None:
    # Downgrade steps
    op.add_column('conversations', sa.Column('agent_ids', postgresql.ARRAY(sa.String()), nullable=False))
    op.drop_table('conversation_agents')
    op.drop_table('agents')

    # Cannot remove an enum value in PostgreSQL easily, so we might need to recreate the enum if necessary
    # For simplicity, we'll leave the enum as-is during downgrade
