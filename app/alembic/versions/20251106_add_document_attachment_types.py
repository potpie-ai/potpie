"""add_document_attachment_types

Revision ID: 20251106_add_document_types
Revises: 20250911184844
Create Date: 2025-11-06

"""

from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20251106_add_document_types"
down_revision: Union[str, None] = "20250911184844"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new attachment types to existing enum
    op.execute("ALTER TYPE attachmenttype ADD VALUE IF NOT EXISTS 'pdf'")
    op.execute("ALTER TYPE attachmenttype ADD VALUE IF NOT EXISTS 'spreadsheet'")
    op.execute("ALTER TYPE attachmenttype ADD VALUE IF NOT EXISTS 'code'")


def downgrade() -> None:
    # Note: PostgreSQL doesn't support removing enum values directly
    # Would need to recreate the enum type, which is complex
    # For downgrade, we'll leave the values (they won't hurt if unused)
    pass
