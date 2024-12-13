"""Support for repo path

Revision ID: 20241127095409_625f792419e7
Revises: 20241028204107_684a330f9e9f
Create Date: 2024-11-27 09:54:09.683918

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20241127095409_625f792419e7"
down_revision: Union[str, None] = "20241028204107_684a330f9e9f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE projects ADD COLUMN repo_path TEXT DEFAULT NULL")


def downgrade() -> None:
    op.drop_column("projects", "repo_path")
