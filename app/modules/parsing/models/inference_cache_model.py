from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey, ARRAY, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.core.base_model import Base

class InferenceCache(Base):
    __tablename__ = "inference_cache"

    id = Column(Integer, primary_key=True, index=True)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)
    project_id = Column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True)
    node_type = Column(String(50), nullable=True)
    content_length = Column(Integer, nullable=True)
    inference_data = Column(JSONB, nullable=False)
    embedding_vector = Column(ARRAY(Float), nullable=True)
    tags = Column(ARRAY(Text), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False, index=True)
    last_accessed = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False, index=True)
    access_count = Column(Integer, default=1, nullable=False)