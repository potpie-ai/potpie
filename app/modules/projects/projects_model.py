from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import relationship, deferred

from app.core.base_model import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Text, primary_key=True)
    properties = Column(BYTEA)
    repo_name = Column(Text)
    branch_name = Column(Text)
    user_id = Column(
        String(255), ForeignKey("users.uid", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    commit_id = Column(String(255))
    is_deleted = Column(Boolean, default=False)
    updated_at = Column(
        TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now()
    )
    status = Column(String(255), default="created")

    __table_args__ = (
        ForeignKeyConstraint(["user_id"], ["users.uid"], ondelete="CASCADE"),
        CheckConstraint(
            "status IN ('submitted', 'cloned', 'parsed', 'ready', 'error')",
            name="check_status",
        ),
    )

    # Project relationships
    user = deferred(relationship("User", back_populates="projects"))
    search_indices = deferred(relationship("SearchIndex", back_populates="project"))
    tasks = deferred(relationship("Task", back_populates="project"))
