"""add_media_attachments_support

Revision ID: 20250626135047_a7f9c1ec89e2
Revises: 20250310201406_97a740b07a50
Create Date: 2025-06-26 13:50:47.430149

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20250626135047_a7f9c1ec89e2"
down_revision: Union[str, None] = "20250310201406_97a740b07a50"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # Add column as nullable first
    op.add_column("messages", sa.Column("has_attachments", sa.Boolean(), nullable=True))
    # Update all existing rows to set has_attachments=False
    op.execute(
        "UPDATE messages SET has_attachments = FALSE WHERE has_attachments IS NULL"
    )
    # Now make the column NOT NULL
    op.alter_column("messages", "has_attachments", nullable=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    op.drop_column("messages", "has_attachments")
