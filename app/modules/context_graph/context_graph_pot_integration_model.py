"""Integration configuration attached to a context graph pot."""

from __future__ import annotations

from sqlalchemy import Column, ForeignKey, String, Text, TIMESTAMP, func
from sqlalchemy.orm import relationship

from app.core.base_model import Base


class ContextGraphPotIntegration(Base):
    __tablename__ = "context_graph_pot_integrations"

    id = Column(Text, primary_key=True)
    pot_id = Column(
        Text,
        ForeignKey("context_graph_pots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    integration_type = Column(Text, nullable=False)
    provider = Column(Text, nullable=False)
    provider_host = Column(Text, nullable=True)
    external_account_id = Column(Text, nullable=True)
    config_json = Column(Text, nullable=True)
    created_by_user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="RESTRICT"),
        nullable=False,
    )
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    pot = relationship("ContextGraphPot", back_populates="integrations")
