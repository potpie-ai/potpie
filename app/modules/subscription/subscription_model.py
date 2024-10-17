from sqlalchemy import TIMESTAMP, Boolean, Column, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey
from app.core.base_model import Base



class Subscription(Base):
    __tablename__ = "subscriptions"

    subscription_id = Column(String(255), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    type = Column(String(255), nullable=False)
    user_id = Column(
        String(255), ForeignKey("users.uid", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    next_billing_date = Column(TIMESTAMP(timezone=True))
    subscription_info = Column(JSONB)

