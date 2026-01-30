"""convert_attachment_types_to_lowercase

Revision ID: 1d75bb639a7e
Revises: 20251106_add_document_types
Create Date: 2025-11-07 14:25:57.453714

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "1d75bb639a7e"
down_revision: Union[str, None] = "20251106_add_document_types"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add lowercase enum values if they don't already exist.
    # PostgreSQL requires each ADD VALUE to be in its own transaction,
    # so we use autocommit mode via op.execute() with separate statements.
    for value in ("image", "video", "audio", "document"):
        op.execute(
            f"ALTER TYPE attachmenttype ADD VALUE IF NOT EXISTS '{value}'"
        )

    # Convert existing uppercase values to lowercase in message_attachments table
    op.execute(
        """
        UPDATE message_attachments
        SET attachment_type = LOWER(attachment_type::text)::attachmenttype
        WHERE attachment_type::text IN ('IMAGE', 'VIDEO', 'AUDIO', 'DOCUMENT')
    """
    )


def downgrade() -> None:
    # Convert back to uppercase (for rollback scenarios)
    op.execute(
        """
        UPDATE message_attachments
        SET attachment_type = UPPER(attachment_type::text)::attachmenttype
        WHERE attachment_type::text IN ('image', 'video', 'audio', 'document')
    """
    )
