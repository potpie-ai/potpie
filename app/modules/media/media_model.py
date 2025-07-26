import enum

from sqlalchemy import TIMESTAMP, Column, ForeignKey, Integer, String, func
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy.dialects.postgresql import JSONB

from app.core.base_model import Base


class AttachmentType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"  # Future extension
    AUDIO = "audio"  # Future extension
    DOCUMENT = "document"  # Future extension


class StorageProvider(str, enum.Enum):
    LOCAL = "local"
    GCS = "gcs"
    S3 = "s3"  # Future extension
    AZURE = "azure"  # Future extension


class MessageAttachment(Base):
    __tablename__ = "message_attachments"

    id = Column(String(255), primary_key=True)
    message_id = Column(
        String(255),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    attachment_type = Column(SQLAEnum(AttachmentType), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    storage_path = Column(String(500), nullable=False)
    storage_provider = Column(
        SQLAEnum(StorageProvider), default=StorageProvider.GCS, nullable=False
    )
    file_metadata = Column(JSONB, nullable=True)  # dimensions, format, etc.
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)

    # Relationship removed to avoid import issues in standalone MediaService
    # The Message model still has the attachments relationship for navigation
