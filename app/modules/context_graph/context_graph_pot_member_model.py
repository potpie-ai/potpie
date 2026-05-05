"""Many-to-many user membership on a context graph pot with a role."""

from __future__ import annotations

from sqlalchemy import Column, ForeignKey, String, Text, TIMESTAMP, func
from sqlalchemy.orm import relationship

from app.core.base_model import Base


class ContextGraphPotMember(Base):
    __tablename__ = "context_graph_pot_members"

    pot_id = Column(
        Text,
        ForeignKey("context_graph_pots.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="CASCADE"),
        primary_key=True,
    )
    role = Column(String(32), nullable=False)
    invited_by_user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="SET NULL"),
        nullable=True,
    )
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    pot = relationship("ContextGraphPot", back_populates="members")
    user = relationship("User", foreign_keys=[user_id], back_populates="context_graph_pot_memberships")
