"""add_resumable_parsing_tables

Revision ID: da2f7e6526ff
Revises: 20250911184844
Create Date: 2025-11-18 20:26:17.860932

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'da2f7e6526ff'
down_revision: Union[str, None] = '20250911184844'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create parsing_sessions table
    op.create_table(
        'parsing_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', sa.Text(), nullable=False),
        sa.Column('commit_id', sa.String(length=255), nullable=False),
        sa.Column('session_number', sa.Integer(), nullable=False),
        sa.Column('coordinator_task_id', sa.String(length=255), nullable=False),
        sa.Column('total_work_units', sa.Integer(), nullable=False),
        sa.Column('completed_work_units', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('failed_work_units', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_files', sa.Integer(), nullable=False),
        sa.Column('processed_files', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('stage', sa.String(length=50), nullable=False),
        sa.Column('last_checkpoint_at', sa.DateTime(), nullable=True),
        sa.Column('checkpoint_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('nodes_created', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('edges_created', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id', 'commit_id', 'session_number', name='uq_project_commit_session')
    )
    op.create_index('idx_session_project_session', 'parsing_sessions', ['project_id', 'session_number'], unique=False)

    # Create parsing_work_units table
    op.create_table(
        'parsing_work_units',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', sa.Text(), nullable=False),
        sa.Column('commit_id', sa.String(length=255), nullable=False),
        sa.Column('work_unit_index', sa.Integer(), nullable=False),
        sa.Column('directory_path', sa.Text(), nullable=False),
        sa.Column('files', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('file_count', sa.Integer(), nullable=False),
        sa.Column('depth', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('attempt_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('nodes_created', sa.Integer(), nullable=True),
        sa.Column('edges_created', sa.Integer(), nullable=True),
        sa.Column('immediate_edges', sa.Integer(), nullable=True),
        sa.Column('defines', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('references', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('last_error_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id', 'commit_id', 'work_unit_index', name='uq_project_commit_unit')
    )
    op.create_index('idx_work_unit_project_status', 'parsing_work_units', ['project_id', 'status'], unique=False)
    op.create_index('idx_work_unit_project_commit', 'parsing_work_units', ['project_id', 'commit_id'], unique=False)

    # Create parsing_file_states table
    op.create_table(
        'parsing_file_states',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', sa.Text(), nullable=False),
        sa.Column('commit_id', sa.String(length=255), nullable=False),
        sa.Column('work_unit_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('nodes_created', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.ForeignKeyConstraint(['work_unit_id'], ['parsing_work_units.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id', 'commit_id', 'file_path', name='uq_project_commit_file')
    )
    op.create_index('idx_file_state_project_file', 'parsing_file_states', ['project_id', 'file_path'], unique=False)
    op.create_index('idx_file_state_work_unit', 'parsing_file_states', ['work_unit_id'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('idx_file_state_work_unit', table_name='parsing_file_states')
    op.drop_index('idx_file_state_project_file', table_name='parsing_file_states')
    op.drop_table('parsing_file_states')

    op.drop_index('idx_work_unit_project_commit', table_name='parsing_work_units')
    op.drop_index('idx_work_unit_project_status', table_name='parsing_work_units')
    op.drop_table('parsing_work_units')

    op.drop_index('idx_session_project_session', table_name='parsing_sessions')
    op.drop_table('parsing_sessions')
