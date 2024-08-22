from sqlalchemy import Column, ForeignKey, String, TIMESTAMP, Boolean, CheckConstraint, func, Integer, Text
from sqlalchemy.dialects.postgresql import BYTEA
from app.core.database import Base
from sqlalchemy.orm import relationship

import enum

class ProjectStatusEnum(str, enum.Enum):
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
    user_id = Column(String(255), ForeignKey("users.uid", ondelete="CASCADE"), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    commit_id = Column(String(255))
    is_deleted = Column(Boolean, default=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now())
    status = Column(String(255), default='created')

    __table_args__ = (
        CheckConstraint("status IN ('created', 'ready', 'error')", name='check_status'),
    )
    
# Project relationships
Project.user = relationship("User", back_populates="projects")

