"""Ensure projects check_status constraint includes inferring

Revision ID: 20260205_ensure_inferring
Revises: 20251217190000
Create Date: 2026-02-05

Safe to run on any DB: drops and recreates check_status so that
'inferring' (and all other ProjectStatusEnum values) are allowed.
Fixes CheckViolation when inference_service sets status to inferring.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260205_ensure_inferring"
down_revision: Union[str, None] = "20251217190000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must match ProjectStatusEnum in app.modules.projects.projects_schema
_ALLOWED_STATUSES = (
    "submitted",
    "cloned",
    "parsed",
    "processing",
    "inferring",
    "ready",
    "error",
)
_CHECK_SQL = "status IN (" + ", ".join(f"'{s}'" for s in _ALLOWED_STATUSES) + ")"


def upgrade() -> None:
    op.drop_constraint("check_status", "projects", type_="check")
    op.create_check_constraint(
        "check_status",
        "projects",
        _CHECK_SQL,
    )


def downgrade() -> None:
    op.drop_constraint("check_status", "projects", type_="check")
    op.create_check_constraint(
        "check_status",
        "projects",
        "status IN ('submitted', 'cloned', 'parsed', 'processing', 'inferring', 'ready', 'error')",
    )
