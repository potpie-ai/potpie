from enum import Enum
from sqlalchemy import TIMESTAMP, Boolean, CheckConstraint, Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, BYTEA
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.sql import func
from app.core.database import Base

class ProjectStatusEnum(str, Enum):
    CREATED = 'created'
    READY = 'ready' 
    ERROR = 'error'

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    directory = Column(Text, unique=True)
    is_default = Column(Boolean, default=False)
    project_name = Column(Text)
    properties = Column(BYTEA)
    repo_name = Column(Text)
    branch_name = Column(Text)
    user_id = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.utcnow(), nullable=False)  # Use UTC timestamp
    commit_id = Column(String(255))
    is_deleted = Column(Boolean, default=False)
    updated_at = Column(
        TIMESTAMP,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
    status = Column(String(255), default='created')
    __table_args__ = (
        ForeignKeyConstraint(["user_id"], ["users.uid"], ondelete="CASCADE"),
        CheckConstraint("status IN ('created', 'ready', 'error')", name='check_status'),
    )

    # Relationships
    user = relationship(
        "User", back_populates="projects"
    )  # Assumes a 'User' class exists