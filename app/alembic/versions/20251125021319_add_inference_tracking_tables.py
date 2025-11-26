"""add_inference_tracking_tables

Revision ID: 20251125021319
Revises: da2f7e6526ff
Create Date: 2025-11-25 02:13:19.000000

Adds InferenceSession and InferenceWorkUnit tables for resumable inference.
Enables:
- Resume capability after failures
- Re-running inference with different models/prompts
- Tracking completion percentage for partial success
- Token usage tracking for cost analysis
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '20251125021319'
down_revision: Union[str, None] = 'da2f7e6526ff'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create inference_sessions table
    op.create_table(
        'inference_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', sa.Text(), nullable=False),
        sa.Column('commit_id', sa.String(length=255), nullable=False),
        sa.Column('session_number', sa.Integer(), nullable=False),
        sa.Column('coordinator_task_id', sa.String(length=255), nullable=True),

        # Progress tracking
        sa.Column('total_work_units', sa.Integer(), nullable=False),
        sa.Column('completed_work_units', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('failed_work_units', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_nodes', sa.Integer(), nullable=False),
        sa.Column('processed_nodes', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('skipped_nodes', sa.Integer(), nullable=True, server_default='0'),

        # Configuration
        sa.Column('use_inference_context', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('prompt_version', sa.String(length=50), nullable=True),

        # Status
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),

        # Checkpoints
        sa.Column('last_checkpoint_at', sa.DateTime(), nullable=True),
        sa.Column('checkpoint_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),

        # Results
        sa.Column('docstrings_generated', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('embeddings_generated', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('search_indices_created', sa.Integer(), nullable=True, server_default='0'),

        # Token tracking
        sa.Column('total_tokens_used', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('estimated_tokens_saved', sa.Integer(), nullable=True, server_default='0'),

        # Error tracking
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('last_error_at', sa.DateTime(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),

        # Constraints
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id', 'commit_id', 'session_number', name='uq_inference_project_commit_session')
    )
    op.create_index('idx_inference_session_project', 'inference_sessions', ['project_id'], unique=False)
    op.create_index('idx_inference_session_status', 'inference_sessions', ['status'], unique=False)
    op.create_index('idx_inference_session_project_status', 'inference_sessions', ['project_id', 'status'], unique=False)

    # Create inference_work_units table
    op.create_table(
        'inference_work_units',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', sa.Text(), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('commit_id', sa.String(length=255), nullable=False),

        # Work unit identification
        sa.Column('work_unit_index', sa.Integer(), nullable=False),
        sa.Column('directory_path', sa.Text(), nullable=False),
        sa.Column('is_root', sa.Boolean(), nullable=True, server_default='false'),

        # Work unit scope
        sa.Column('node_count', sa.Integer(), nullable=False),
        sa.Column('split_index', sa.Integer(), nullable=True),
        sa.Column('total_splits', sa.Integer(), nullable=True),

        # State tracking
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('attempt_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('max_attempts', sa.Integer(), nullable=True, server_default='3'),

        # Results
        sa.Column('nodes_processed', sa.Integer(), nullable=True),
        sa.Column('docstrings_generated', sa.Integer(), nullable=True),
        sa.Column('batches_processed', sa.Integer(), nullable=True),
        sa.Column('failed_batches', sa.Integer(), nullable=True),

        # Token tracking
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('context_used_count', sa.Integer(), nullable=True),
        sa.Column('fallback_count', sa.Integer(), nullable=True),

        # Error tracking
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_type', sa.String(length=100), nullable=True),
        sa.Column('last_error_at', sa.DateTime(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),

        # Constraints
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['inference_sessions.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id', 'work_unit_index', name='uq_inference_session_unit_index')
    )
    op.create_index('idx_inference_unit_session', 'inference_work_units', ['session_id'], unique=False)
    op.create_index('idx_inference_unit_status', 'inference_work_units', ['status'], unique=False)
    op.create_index('idx_inference_unit_session_status', 'inference_work_units', ['session_id', 'status'], unique=False)
    op.create_index('idx_inference_unit_project', 'inference_work_units', ['project_id'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order (work_units first due to FK)
    op.drop_index('idx_inference_unit_project', table_name='inference_work_units')
    op.drop_index('idx_inference_unit_session_status', table_name='inference_work_units')
    op.drop_index('idx_inference_unit_status', table_name='inference_work_units')
    op.drop_index('idx_inference_unit_session', table_name='inference_work_units')
    op.drop_table('inference_work_units')

    op.drop_index('idx_inference_session_project_status', table_name='inference_sessions')
    op.drop_index('idx_inference_session_status', table_name='inference_sessions')
    op.drop_index('idx_inference_session_project', table_name='inference_sessions')
    op.drop_table('inference_sessions')
