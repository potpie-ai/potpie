from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.base_model import Base


class SearchIndex(Base):
    __tablename__ = "search_indices"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Text, ForeignKey("projects.id"), index=True)
    node_id = Column(String, index=True)
    name = Column(String, index=True)
    file_path = Column(String, index=True)
    content = Column(Text)

    project = relationship("Project", back_populates="search_indices")
