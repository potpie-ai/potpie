from sqlalchemy import JSON, Column, ForeignKey, Index, String
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.modules.users.user_model import User  # noqa


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    user_id = Column(String, ForeignKey("users.uid"), primary_key=True)
    preferences = Column(JSON, nullable=False, default={})

    user = relationship("User", back_populates="preferences")

    __table_args__ = (Index("idx_user_preferences_user_id", "user_id"),)
