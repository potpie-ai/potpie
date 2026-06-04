"""merge heads

Revision ID: 82eb6e97aed3
Revises: 20241020111943_262d870e9686, 20241127095409_625f792419e7
Create Date: 2025-03-03 16:48:42.230151

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "82eb6e97aed3"
down_revision: Union[str, None] = (
    "20241020111943_262d870e9686",
    "20241127095409_625f792419e7",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
