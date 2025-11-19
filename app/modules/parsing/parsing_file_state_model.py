import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.base_model import Base


class ParsingFileState(Base):
    """
    Tracks the parsing state of individual files within a work unit.

    Provides fine-grained tracking of which files have been successfully parsed,
    enabling idempotent operations and detailed progress reporting.
    """
    __tablename__ = "parsing_file_states"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign keys
    project_id = Column(Text, ForeignKey('projects.id'), nullable=False)
    commit_id = Column(String(255), nullable=False)
    work_unit_id = Column(UUID(as_uuid=True), ForeignKey('parsing_work_units.id'), nullable=False)

    # File identification
    file_path = Column(Text, nullable=False)  # Relative path

    # Processing state
    status = Column(String(50), nullable=False, default='pending')  # pending, completed, failed, skipped
    nodes_created = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Constraints
    __table_args__ = (
        UniqueConstraint('project_id', 'commit_id', 'file_path', name='uq_project_commit_file'),
        Index('idx_file_state_project_file', 'project_id', 'file_path'),
        Index('idx_file_state_work_unit', 'work_unit_id'),
    )
