"""Persist raw invite token so pending invitations stay re-shareable.

Revision ID: ctx_invitation_token_20260515
Revises: ctx_agent_exec_log_20260515
Create Date: 2026-05-15

Adds ``context_graph_pot_invitations.token``. Until now only ``token_hash``
was stored and the raw token was surfaced exactly once at creation, so an
owner could not re-copy the link or re-send the email for an existing
pending invite.

This is safe to persist because the accept endpoint additionally requires
the signed-in user's email to match the invite — the token is an
*identifier*, not a bearer credential.

Backfill: existing *pending* rows predate any email being sent (the invite
email path did not exist before this change), so we mint a fresh
token+token_hash for them so they immediately become shareable. Non-pending
rows (accepted/revoked/expired) are left with ``token = NULL`` — there is
nothing left to share.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "ctx_invitation_token_20260515"
down_revision: Union[str, None] = "ctx_agent_exec_log_20260515"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "context_graph_pot_invitations",
        sa.Column("token", sa.Text(), nullable=True),
    )

    # Re-mint pending invites so the new copy-link / resend actions work for
    # rows created before this migration. token_hash is UNIQUE; token_urlsafe
    # is collision-safe at this size, so a straight per-row update is fine.
    bind = op.get_bind()
    pending = bind.execute(
        sa.text(
            "SELECT id FROM context_graph_pot_invitations "
            "WHERE status = 'pending'"
        )
    ).fetchall()
    for (invitation_id,) in pending:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        bind.execute(
            sa.text(
                "UPDATE context_graph_pot_invitations "
                "SET token = :token, token_hash = :token_hash "
                "WHERE id = :id"
            ),
            {"token": token, "token_hash": token_hash, "id": invitation_id},
        )


def downgrade() -> None:
    op.drop_column("context_graph_pot_invitations", "token")
