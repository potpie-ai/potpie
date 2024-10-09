from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from app.core.base_model import Base
from app.modules.users.user_model import User

class CustomAgent(Base):
    __tablename__ = "custom_agents"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.uid"))
    role = Column(String)
    goal = Column(String)
    backstory = Column(String)
    tool_ids = Column(ARRAY(String))
    tasks = Column(JSONB)
    deployment_url = Column(String)

    user = relationship(User)