from sqlalchemy import TIMESTAMP, Boolean, Column, String, func
from sqlalchemy.dialects.postgresql import JSONB

from app.core.base_model import Base


class Integration(Base):
    """Database model for integrations with JSON fields for extensibility"""

    __tablename__ = "integrations"

    integration_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    integration_type = Column(
        String(50), nullable=False
    )  # sentry, github, slack, jira, etc.
    status = Column(
        String(20), default="active", nullable=False
    )  # active, inactive, pending, error
    active = Column(Boolean, default=True, nullable=False)

    # JSON fields for extensible sub-structures
    auth_data = Column(JSONB)  # Authentication data (tokens, scopes, etc.)
    scope_data = Column(JSONB)  # Scope-specific data (org, workspace, project IDs)
    integration_metadata = Column(
        JSONB
    )  # Additional metadata (name, description, tags)

    # System fields
    unique_identifier = Column(String(255), nullable=False)
    created_by = Column(
        String(255), nullable=False
    )  # User ID who created the integration
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return f"<Integration(id={self.integration_id}, name={self.name}, type={self.integration_type})>"
