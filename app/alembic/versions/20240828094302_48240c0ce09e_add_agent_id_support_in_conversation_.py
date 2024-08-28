"""Add agent id support in conversation table

Revision ID: 20240828094302_48240c0ce09e
Revises: 20240826215938_3c7be0985b17
Create Date: 2024-08-28 09:43:02.922148

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20240828094302_48240c0ce09e"
down_revision: Union[str, None] = "20240826215938_3c7be0985b17"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "conversations", sa.Column("agent_id", sa.String(length=255), nullable=False)
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("conversations", "agent_id")
    # ### end Alembic commands ###
