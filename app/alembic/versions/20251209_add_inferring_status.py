"""Add inferring status to projects

Revision ID: 20251209_add_inferring_status
Revises: 20251202164905_07bea433f543
Create Date: 2025-12-09 16:45:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20251209_add_inferring_status"
down_revision: Union[str, None] = "20251202164905_07bea433f543"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the existing check constraint
    op.drop_constraint("check_status", "projects", type_="check")

    # Create the new check constraint with 'inferring' status
    # Note: 'processing' is included to match ProjectStatusEnum.PROCESSING
    op.create_check_constraint(
        "check_status",
        "projects",
        "status IN ('submitted', 'cloned', 'parsed', 'processing', 'inferring', 'ready', 'error')",
    )


def downgrade() -> None:
    # Drop the constraint with 'inferring'
    op.drop_constraint("check_status", "projects", type_="check")

    # Restore the original constraint without 'inferring'
    # Note: 'processing' is included to match ProjectStatusEnum.PROCESSING
    op.create_check_constraint(
        "check_status",
        "projects",
        "status IN ('submitted', 'cloned', 'parsed', 'processing', 'ready', 'error')",
    )
