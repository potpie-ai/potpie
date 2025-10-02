"""Remove foreign key constraint for global cache independence

Revision ID: 20250928_simple_global_cache
Revises: 20250923_add_inference_cache
Create Date: 2025-09-28 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '20250928_simple_global_cache'
down_revision = '20250923_add_inference_cache'
branch_labels = None
depends_on = None

def upgrade():
    # Remove foreign key constraint that deletes cache when projects are deleted
    op.drop_constraint('inference_cache_project_id_fkey', 'inference_cache', type_='foreignkey')

    # project_id remains as nullable metadata field - no schema change needed
    # Existing indexes remain optimal for hash-only lookups

def downgrade():
    # Pre-cleanup: remove orphaned references so FK creation doesn't fail
    # If you prefer to retain rows, replace DELETE with UPDATE to set project_id=NULL
    op.execute(
        """
        DELETE FROM inference_cache ic
        WHERE ic.project_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM projects p WHERE p.id = ic.project_id
          )
        """
    )

    # Restore foreign key if needed (for rollback)
    op.create_foreign_key(
        'inference_cache_project_id_fkey',
        'inference_cache', 'projects',
        ['project_id'], ['id'],
        ondelete='CASCADE'
    )