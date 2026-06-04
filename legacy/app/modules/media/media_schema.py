from datetime import datetime
from typing import Dict, Optional, Any

from pydantic import BaseModel, Field

from app.modules.media.media_model import AttachmentType, StorageProvider


class AttachmentInfo(BaseModel):
    id: str
    attachment_type: AttachmentType
    file_name: str
    mime_type: str
    file_size: int
    storage_path: Optional[str] = None  # Not exposed in API responses for security
    storage_provider: StorageProvider
    file_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    download_url: Optional[str] = None  # Signed URL for client access

    class Config:
        from_attributes = True


class AttachmentUploadResponse(BaseModel):
    id: str
    attachment_type: AttachmentType
    file_name: str
    mime_type: str
    file_size: int
    message: str = "Attachment uploaded successfully"


class AttachmentAccessResponse(BaseModel):
    download_url: str
    expires_in: int = Field(description="URL expiration time in seconds")


class MediaUploadError(BaseModel):
    error: str
    details: Optional[str] = None
