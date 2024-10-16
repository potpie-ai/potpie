from sqlalchemy import TIMESTAMP, Column, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.core.base_model import Base
from app.modules.users.user_model import User

# note - do not import this model for alembic migrations

class CustomAgent(Base):
    __tablename__ = "custom_agents"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.uid"))
    role = Column(String)
    goal = Column(String)
    backstory = Column(String)
    system_prompt = Column(String)
    tasks = Column(JSONB)
    deployment_url = Column(String)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship(User)
