"""Enable pgvector extension

Revision ID: 20251222_enable_pgvector
Revises: 20250928_simple_global_cache
Create Date: 2025-12-22 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '20251222_enable_pgvector'
down_revision = '20250928_simple_global_cache'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')


def downgrade() -> None:
    # Note: Dropping extension may fail if tables use vector type
    # This is a safety measure - manual cleanup may be required
    op.execute('DROP EXTENSION IF EXISTS vector CASCADE')
