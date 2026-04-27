"""Pot source — the ingestable data scope (repo / team / channel / project).

Separate from :class:`ContextGraphPotIntegration`, which represents the
account-level connection/credential. A single integration can produce many
sources (e.g. one Linear connection → many teams).
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    String,
    Text,
    TIMESTAMP,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from app.core.base_model import Base


SOURCE_KIND_REPOSITORY = "repository"
SOURCE_KIND_ISSUE_TRACKER_TEAM = "issue_tracker_team"


class ContextGraphPotSource(Base):
    __tablename__ = "context_graph_pot_sources"
    __table_args__ = (
        UniqueConstraint(
            "pot_id",
            "provider",
            "source_kind",
            "scope_hash",
            name="uq_context_graph_pot_source_scope",
        ),
    )

    id = Column(Text, primary_key=True)
    pot_id = Column(
        Text,
        ForeignKey("context_graph_pots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    integration_id = Column(String(255), nullable=True)
    provider = Column(Text, nullable=False)
    source_kind = Column(Text, nullable=False)
    scope_json = Column(Text, nullable=True)
    scope_hash = Column(String(128), nullable=False)
    sync_enabled = Column(Boolean, nullable=False, default=True)
    sync_mode = Column(Text, nullable=True)
    webhook_status = Column(Text, nullable=True)
    last_sync_at = Column(TIMESTAMP(timezone=True), nullable=True)
    last_error = Column(Text, nullable=True)
    health_score = Column(Text, nullable=True)
    added_by_user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="RESTRICT"),
        nullable=False,
    )
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    pot = relationship("ContextGraphPot")
