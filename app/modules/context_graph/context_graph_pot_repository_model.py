"""Repository attached to a context graph pot (source inside the pot, not the scope)."""

from __future__ import annotations

from sqlalchemy import Column, ForeignKey, String, Text, TIMESTAMP, UniqueConstraint, func
from sqlalchemy.orm import relationship

from app.core.base_model import Base


class ContextGraphPotRepository(Base):
    __tablename__ = "context_graph_pot_repositories"
    __table_args__ = (
        UniqueConstraint(
            "pot_id",
            "provider",
            "provider_host",
            "owner",
            "repo",
            name="uq_context_graph_pot_repository",
        ),
    )

    id = Column(Text, primary_key=True)
    pot_id = Column(
        Text,
        ForeignKey("context_graph_pots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider = Column(Text, nullable=False)
    provider_host = Column(Text, nullable=False)
    owner = Column(Text, nullable=False)
    repo = Column(Text, nullable=False)
    external_repo_id = Column(Text, nullable=True)
    remote_url = Column(Text, nullable=True)
    default_branch = Column(Text, nullable=True)
    added_by_user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="RESTRICT"),
        nullable=False,
    )
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    pot = relationship("ContextGraphPot", back_populates="repositories")
