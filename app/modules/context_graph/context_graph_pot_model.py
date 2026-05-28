"""User-owned context graph pots (independent of ``projects``)."""

from sqlalchemy import Column, ForeignKey, String, Text, TIMESTAMP, func
from sqlalchemy.orm import relationship

from app.core.base_model import Base


class ContextGraphPot(Base):
    """A pot id (Graphiti ``group_id``); tenancy is via :class:`ContextGraphPotMember`."""

    __tablename__ = "context_graph_pots"

    id = Column(Text, primary_key=True)
    user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by_user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    display_name = Column(Text, nullable=True)
    slug = Column(Text, nullable=True)
    primary_repo_name = Column(Text, nullable=True)
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    archived_at = Column(TIMESTAMP(timezone=True), nullable=True)

    user = relationship("User", back_populates="context_graph_pots", foreign_keys=[user_id])
    members = relationship(
        "ContextGraphPotMember",
        back_populates="pot",
        cascade="all, delete-orphan",
    )
    repositories = relationship(
        "ContextGraphPotRepository",
        back_populates="pot",
        cascade="all, delete-orphan",
    )
    integrations = relationship(
        "ContextGraphPotIntegration",
        back_populates="pot",
        cascade="all, delete-orphan",
    )
