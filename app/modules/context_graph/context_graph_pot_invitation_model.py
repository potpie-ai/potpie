"""Pending email invitation to join a context-graph pot."""

from __future__ import annotations

from sqlalchemy import Column, ForeignKey, String, Text, TIMESTAMP, func
from sqlalchemy.orm import relationship

from app.core.base_model import Base


INVITATION_STATUS_PENDING = "pending"
INVITATION_STATUS_ACCEPTED = "accepted"
INVITATION_STATUS_REVOKED = "revoked"
INVITATION_STATUS_EXPIRED = "expired"
# Invitee was auto-added on invite, then opted out. Membership is removed and
# the pot disappears for them. No schema change: `status` is a free-form
# String(16) with no CHECK constraint, and the partial unique index that keeps
# one invite per (pot, email) only covers status = 'pending', so a declined
# row never blocks a fresh re-invite.
INVITATION_STATUS_DECLINED = "declined"


class ContextGraphPotInvitation(Base):
    __tablename__ = "context_graph_pot_invitations"

    id = Column(Text, primary_key=True)
    pot_id = Column(
        Text,
        ForeignKey("context_graph_pots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    email = Column(String(255), nullable=False, index=True)
    role = Column(String(32), nullable=False, default="user")
    invited_by_user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="RESTRICT"),
        nullable=False,
    )
    accepted_by_user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="SET NULL"),
        nullable=True,
    )
    token_hash = Column(String(128), nullable=False, unique=True, index=True)
    # Raw invite token, kept so a pending invite stays re-shareable (copy
    # link / resend email). Safe to persist: the accept endpoint additionally
    # requires the signed-in user's email to match, so the token is an
    # identifier, not a bearer credential. NULL for legacy/non-pending rows.
    token = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default=INVITATION_STATUS_PENDING)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    accepted_at = Column(TIMESTAMP(timezone=True), nullable=True)

    pot = relationship("ContextGraphPot")
