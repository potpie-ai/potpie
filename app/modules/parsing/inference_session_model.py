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


class InferenceSession(Base):
    """
    Represents an inference session for a specific commit of a project.

    A session tracks the entire lifecycle of generating docstrings for a repository,
    including work unit distribution, progress tracking, and completion status.

    This enables:
    - Resume capability after failures
    - Re-running inference with different models/prompts
    - Tracking completion percentage for partial success
    """
    __tablename__ = "inference_sessions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign keys
    project_id = Column(Text, ForeignKey('projects.id'), nullable=False)
    commit_id = Column(String(255), nullable=False)

    # Session metadata
    session_number = Column(Integer, nullable=False)  # 1, 2, 3... for retries/re-runs
    coordinator_task_id = Column(String(255), nullable=True)

    # Progress tracking
    total_work_units = Column(Integer, nullable=False)
    completed_work_units = Column(Integer, default=0)
    failed_work_units = Column(Integer, default=0)

    total_nodes = Column(Integer, nullable=False)  # Total nodes to process
    processed_nodes = Column(Integer, default=0)  # Successfully processed
    skipped_nodes = Column(Integer, default=0)  # Already had docstrings

    # Configuration
    use_inference_context = Column(Boolean, default=True)  # Whether using optimized context
    model_name = Column(String(100), nullable=True)  # LLM model used (e.g., gpt-4)
    prompt_version = Column(String(50), nullable=True)  # For tracking prompt changes

    # Status
    status = Column(String(50), nullable=False, default='pending')
    # Status values: pending, running, paused, completed, failed, partial

    # Checkpoints for resume
    last_checkpoint_at = Column(DateTime, nullable=True)
    checkpoint_data = Column(JSONB, nullable=True)  # Work unit progress, last offset, etc.

    # Results (populated on completion)
    docstrings_generated = Column(Integer, default=0)
    embeddings_generated = Column(Integer, default=0)
    search_indices_created = Column(Integer, default=0)

    # Token tracking (for cost analysis)
    total_tokens_used = Column(Integer, default=0)
    estimated_tokens_saved = Column(Integer, default=0)  # Due to inference_context optimization

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
        UniqueConstraint('project_id', 'commit_id', 'session_number', name='uq_inference_project_commit_session'),
        Index('idx_inference_session_project', 'project_id'),
        Index('idx_inference_session_status', 'status'),
        Index('idx_inference_session_project_status', 'project_id', 'status'),
    )

    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_work_units == 0:
            return 0.0
        return (self.completed_work_units / self.total_work_units) * 100

    def is_resumable(self) -> bool:
        """Check if session can be resumed."""
        return self.status in ('paused', 'failed', 'partial')

    def mark_running(self):
        """Mark session as running."""
        self.status = 'running'
        if not self.started_at:
            self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_completed(self):
        """Mark session as completed."""
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_failed(self, error_message: str):
        """Mark session as failed."""
        self.status = 'failed'
        self.error_message = error_message
        self.last_error_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_partial(self):
        """Mark session as partially complete (75-95% success)."""
        self.status = 'partial'
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
