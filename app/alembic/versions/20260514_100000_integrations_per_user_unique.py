"""Make ``integrations.unique_identifier`` unique per user, not globally.

Revision ID: int_uniq_per_user_20260514
Revises: ctx_narrow_open_batch_20260512
Create Date: 2026-05-14

The legacy ``ix_integrations_unique_identifier`` is a single-column unique
index on ``unique_identifier`` alone. For Linear, ``unique_identifier`` is
the workspace's ``organization.id`` — which is shared across everyone in
that workspace. The single-column constraint therefore prevents two users
of the same Linear org from installing the Potpie app independently:
the second user's INSERT trips the unique violation and the raw
``psycopg2.errors.UniqueViolation`` bubbles up through the OAuth
callback.

The intended invariant is "one integration per (workspace, user)" — the
per-user dedup is already implemented in
``IntegrationsService.check_existing_linear_integration``. This migration
brings the schema in line by replacing the single-column index with a
composite unique index on ``(unique_identifier, created_by)``. A plain
(non-unique) index on ``unique_identifier`` remains so webhook fan-out
(``process_linear_webhook``) can still look up all integrations matching
a given ``organizationId``.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "int_uniq_per_user_20260514"
down_revision: Union[str, None] = "ctx_narrow_open_batch_20260512"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_integrations_unique_identifier", table_name="integrations")
    op.create_index(
        "ix_integrations_unique_identifier",
        "integrations",
        ["unique_identifier"],
        unique=False,
    )
    op.create_index(
        "uq_integrations_unique_identifier_per_user",
        "integrations",
        ["unique_identifier", "created_by"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_integrations_unique_identifier_per_user", table_name="integrations"
    )
    op.drop_index("ix_integrations_unique_identifier", table_name="integrations")
    op.create_index(
        "ix_integrations_unique_identifier",
        "integrations",
        ["unique_identifier"],
        unique=True,
    )
