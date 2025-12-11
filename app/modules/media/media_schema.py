from datetime import datetime
from typing import Dict, List, Optional, Any

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
    token_count: Optional[int] = None  # Token count for documents (None for images)
    message: str = "Attachment uploaded successfully"


class AttachmentAccessResponse(BaseModel):
    download_url: str
    expires_in: int = Field(description="URL expiration time in seconds")


class MediaUploadError(BaseModel):
    error: str
    details: Optional[str] = None


class SupportedFormatInfo(BaseModel):
    mime_types: List[str]
    extensions: List[str]
    max_size_bytes: int
    description: str


class SupportedFormatsResponse(BaseModel):
    images: SupportedFormatInfo
    documents: SupportedFormatInfo
    spreadsheets: SupportedFormatInfo
    code_files: SupportedFormatInfo
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
