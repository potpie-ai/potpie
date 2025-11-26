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
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.base_model import Base


class InferenceWorkUnit(Base):
    """
    Represents a work unit (directory) for inference in the distributed system.

    Work units are created when inference is spawned and track the progress
    of generating docstrings for a specific directory within a repository.

    This enables:
    - Per-directory retry on failure
    - Resume capability from last successful directory
    - Parallel processing across workers
    - Detailed progress tracking
    """
    __tablename__ = "inference_work_units"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign keys
    project_id = Column(Text, ForeignKey('projects.id'), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey('inference_sessions.id'), nullable=False)
    commit_id = Column(String(255), nullable=False)

    # Work unit identification
    work_unit_index = Column(Integer, nullable=False)  # 0, 1, 2, ...
    directory_path = Column(Text, nullable=False)  # e.g., "src/services" or "/" for root
    is_root = Column(Boolean, default=False)  # True for root-level files

    # Work unit scope
    node_count = Column(Integer, nullable=False)  # Nodes in this directory
    split_index = Column(Integer, nullable=True)  # For large directories split across workers
    total_splits = Column(Integer, nullable=True)  # Total splits for this directory

    # State tracking
    status = Column(String(50), nullable=False, default='pending')
    # Status values: pending, processing, completed, failed, skipped
    celery_task_id = Column(String(255), nullable=True)  # Current task ID
    attempt_count = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)

    # Results (populated on completion)
    nodes_processed = Column(Integer, nullable=True)
    docstrings_generated = Column(Integer, nullable=True)
    batches_processed = Column(Integer, nullable=True)
    failed_batches = Column(Integer, nullable=True)

    # Token tracking
    tokens_used = Column(Integer, nullable=True)
    context_used_count = Column(Integer, nullable=True)  # Nodes that used inference_context
    fallback_count = Column(Integer, nullable=True)  # Nodes that fell back to full text

    # Error tracking
    error_message = Column(Text, nullable=True)
    error_type = Column(String(100), nullable=True)  # Exception class name
    last_error_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            'session_id', 'work_unit_index',
            name='uq_inference_session_unit_index'
        ),
        Index('idx_inference_unit_session', 'session_id'),
        Index('idx_inference_unit_status', 'status'),
        Index('idx_inference_unit_session_status', 'session_id', 'status'),
        Index('idx_inference_unit_project', 'project_id'),
    )

    def is_retriable(self) -> bool:
        """Check if work unit can be retried."""
        return self.status == 'failed' and self.attempt_count < self.max_attempts

    def mark_processing(self, task_id: str):
        """Mark work unit as being processed."""
        self.status = 'processing'
        self.celery_task_id = task_id
        self.attempt_count += 1
        if not self.started_at:
            self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_completed(
        self,
        nodes_processed: int,
        docstrings_generated: int,
        batches_processed: int = 0,
        failed_batches: int = 0,
        tokens_used: int = 0,
        context_used_count: int = 0,
        fallback_count: int = 0,
    ):
        """Mark work unit as completed with results."""
        self.status = 'completed'
        self.nodes_processed = nodes_processed
        self.docstrings_generated = docstrings_generated
        self.batches_processed = batches_processed
        self.failed_batches = failed_batches
        self.tokens_used = tokens_used
        self.context_used_count = context_used_count
        self.fallback_count = fallback_count
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_failed(self, error_message: str, error_type: str = None):
        """Mark work unit as failed."""
        self.status = 'failed'
        self.error_message = error_message
        self.error_type = error_type
        self.last_error_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_skipped(self, reason: str = "No nodes to process"):
        """Mark work unit as skipped."""
        self.status = 'skipped'
        self.error_message = reason
        self.nodes_processed = 0
        self.docstrings_generated = 0
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
