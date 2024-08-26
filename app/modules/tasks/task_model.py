from sqlalchemy import Column, Integer, String, ForeignKey, Enum, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import CheckConstraint
from datetime import datetime
import enum
from app.core.database import Base


class TaskType(enum.Enum):
    CODEBASE_PROCESSING = "CODE_INFERENCE"
    FILE_INFERENCE = "FILE_INFERENCE"
    FLOWS_PROCESSING = "FLOW_INFERENCE"
    FLOW_INFERENCE = "FLOW_INFERENCE"

class Task(Base):
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True)
    task_type = Column(Enum(TaskType), nullable=False)
    custom_status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    project_id = Column(String, ForeignKey('projects.id'), nullable=False)
    result = Column(String, nullable=True)
    
    # Use string-based reference for relationship
    # project = relationship("Project", back_populates="tasks")
Task.project = relationship("Project", back_populates="tasks")