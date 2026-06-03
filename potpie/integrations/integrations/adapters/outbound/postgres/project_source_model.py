"""Project-level attachment to an external provider (repo, Linear team, etc.)."""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TIMESTAMP

from app.core.base_model import Base


class ProjectSource(Base):
    __tablename__ = "project_sources"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(
        Text,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    integration_id = Column(
        String(255),
        ForeignKey("integrations.integration_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    provider = Column(String(32), nullable=False)
    source_kind = Column(String(64), nullable=False)
    scope_json = Column(JSONB, nullable=False)
    scope_hash = Column(String(64), nullable=False)
    sync_enabled = Column(Boolean, nullable=False, default=True)
    sync_mode = Column(String(32), nullable=False, default="hybrid")
    webhook_status = Column(String(32), nullable=False, default="not_applicable")
    last_sync_at = Column(TIMESTAMP(timezone=True), nullable=True)
    last_error = Column(Text, nullable=True)
    health_score = Column(Integer, nullable=False, default=100)
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
