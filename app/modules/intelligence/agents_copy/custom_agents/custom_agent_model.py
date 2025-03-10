from sqlalchemy import Column, DateTime, ForeignKeyConstraint, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.core.base_model import Base


class CustomAgent(Base):
    __tablename__ = "custom_agents"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    role = Column(String)
    goal = Column(String)
    backstory = Column(String)
    system_prompt = Column(String)
    tasks = Column(JSONB)
    deployment_url = Column(String, nullable=True)
    deployment_status = Column(String, default="STOPPED", nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    # Add foreign key constraint to users table
    __table_args__ = (ForeignKeyConstraint(["user_id"], ["users.uid"]),)

    # Add relationship to User model
    user = relationship("User", back_populates="custom_agents")
