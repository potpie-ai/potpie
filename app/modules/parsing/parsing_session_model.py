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
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.base_model import Base


class ParsingSession(Base):
    """
    Represents a parsing session for a specific commit of a project.

    A session tracks the entire lifecycle of parsing a repository commit,
    including work unit distribution, progress tracking, and completion status.
    """
    __tablename__ = "parsing_sessions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign keys
    project_id = Column(Text, ForeignKey('projects.id'), nullable=False)
    commit_id = Column(String(255), nullable=False)

    # Session metadata
    session_number = Column(Integer, nullable=False)  # 1, 2, 3... for retries
    coordinator_task_id = Column(String(255), nullable=False)

    # Progress tracking
    total_work_units = Column(Integer, nullable=False)
    completed_work_units = Column(Integer, default=0)
    failed_work_units = Column(Integer, default=0)

    total_files = Column(Integer, nullable=False)
    processed_files = Column(Integer, default=0)

    # Stage tracking
    stage = Column(String(50), nullable=False)  # scanning, parsing, aggregating, resolving, inferring, finalizing

    # Checkpoints
    last_checkpoint_at = Column(DateTime, nullable=True)
    checkpoint_data = Column(JSONB, nullable=True)  # Stage-specific checkpoint data

    # Results
    nodes_created = Column(Integer, default=0)
    edges_created = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Constraints
    __table_args__ = (
        UniqueConstraint('project_id', 'commit_id', 'session_number', name='uq_project_commit_session'),
        Index('idx_session_project_session', 'project_id', 'session_number'),
    )
