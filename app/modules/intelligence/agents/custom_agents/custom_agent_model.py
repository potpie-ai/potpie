from sqlalchemy import Column, DateTime, ForeignKeyConstraint, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from typing import List

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
    visibility = Column(String, default="private", nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    # Add foreign key constraint to users table
    __table_args__ = (ForeignKeyConstraint(["user_id"], ["users.uid"]),)

    # Add relationships
    user = relationship("User", back_populates="custom_agents")
    shares = relationship(
        "CustomAgentShare", back_populates="agent", cascade="all, delete-orphan"
    )

    @property
    def shared_with_users(self) -> List["User"]:
        return [share.shared_with_user for share in self.shares]


class CustomAgentShare(Base):
    __tablename__ = "custom_agent_shares"

    id = Column(String, primary_key=True)
    agent_id = Column(String, nullable=False)
    shared_with_user_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # Add foreign key constraints
    __table_args__ = (
        ForeignKeyConstraint(["agent_id"], ["custom_agents.id"], ondelete="CASCADE"),
        ForeignKeyConstraint(
            ["shared_with_user_id"], ["users.uid"], ondelete="CASCADE"
        ),
    )

    # Add relationships
    agent = relationship("CustomAgent", back_populates="shares")
    shared_with_user = relationship("User")
