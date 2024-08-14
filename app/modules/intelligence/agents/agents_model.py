from sqlalchemy import Column, String, TIMESTAMP, func, Enum as SQLAEnum
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum
from uuid6 import uuid7

# Define the Provider Enum
class ProviderType(enum.Enum):
    LANGCHAIN = "langchain"
    CUSTOM = "custom"
    CREWAI = "crewAI"

class AgentStatus(enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEPRECATED = "DEPRECATED"

class Agent(Base):
    __tablename__ = "agents"

    id = Column(String(255), primary_key=True, default=lambda: str(uuid7()))
    name = Column(String(255), unique=True, nullable=False)
    description = Column(String(1024))  # Optional description of the agent
    provider = Column(SQLAEnum(ProviderType), nullable=False)  # Provider (e.g., Langchain, Custom, CrewAI)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False) #will store timestamp in utc
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    status = Column(SQLAEnum(AgentStatus), default=AgentStatus.ACTIVE, nullable=False)  # Active/Inactive status of the agent

    # Relationship with Conversation
    conversations = relationship("Conversation", secondary="conversation_agents", back_populates="agents")
