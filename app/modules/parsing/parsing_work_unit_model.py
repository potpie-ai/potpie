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


class ParsingWorkUnit(Base):
    """
    Represents a work unit (directory) to be parsed in the distributed parsing system.

    Work units are created during the scanning phase and track the parsing progress
    of a specific directory within a repository commit.
    """
    __tablename__ = "parsing_work_units"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign keys
    project_id = Column(Text, ForeignKey('projects.id'), nullable=False)
    commit_id = Column(String(255), nullable=False)

    # Work unit identification
    work_unit_index = Column(Integer, nullable=False)
    directory_path = Column(Text, nullable=False)

    # Work unit data
    files = Column(JSONB, nullable=False)  # List of file paths
    file_count = Column(Integer, nullable=False)
    depth = Column(Integer, nullable=False)

    # State tracking
    status = Column(String(50), nullable=False, default='pending')  # pending, processing, completed, failed
    celery_task_id = Column(String(255), nullable=True)  # Current task ID
    attempt_count = Column(Integer, default=0)

    # Results (populated on completion)
    nodes_created = Column(Integer, nullable=True)
    edges_created = Column(Integer, nullable=True)
    immediate_edges = Column(Integer, nullable=True)
    defines = Column(JSONB, nullable=True)  # Identifier -> node names mapping
    references = Column(JSONB, nullable=True)  # Unresolved references

    # Error tracking
    error_message = Column(Text, nullable=True)
    last_error_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Constraints
    __table_args__ = (
        UniqueConstraint('project_id', 'commit_id', 'work_unit_index', name='uq_project_commit_unit'),
        Index('idx_work_unit_project_status', 'project_id', 'status'),
        Index('idx_work_unit_project_commit', 'project_id', 'commit_id'),
    )
