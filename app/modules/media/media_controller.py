from typing import List, Optional

from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io

from app.core.config_provider import config_provider
from app.modules.media.media_service import MediaService, MediaServiceError
from app.modules.media.media_schema import (
    AttachmentUploadResponse,
    AttachmentAccessResponse,
    AttachmentInfo,
)
from app.modules.conversations.access.access_service import ShareChatService
from app.modules.conversations.conversation.conversation_schema import (
    ConversationAccessType,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _safe_content_disposition_filename(name: str) -> str:
    """Sanitize filename for Content-Disposition to prevent header injection."""
    if not name or not name.strip():
        return "attachment"
    # Remove control chars, newlines, and double quotes; keep safe printable chars
    safe = "".join(
        c
        for c in name
        if c.isprintable() and c not in '"\\\r\n\t'
    ).strip()
    return safe[:255] if safe else "attachment"


class MediaController:
    def __init__(self, db: Session, user_id: str, user_email: str):
        self.db = db
        self.user_id = user_id
        self.user_email = user_email
        self.media_service = MediaService(db)
        self.share_chat_service = ShareChatService(db)

    def _check_multimodal_enabled(self):
        """Check if multimodal functionality is enabled"""
        if not config_provider.get_is_multimodal_enabled():
            raise HTTPException(
                status_code=501,
                detail={
                    "error": "Multimodal functionality is currently disabled",
                    "code": "MULTIMODAL_DISABLED",
                },
            )

    async def upload_image(
        self, file: UploadFile, message_id: Optional[str] = None
    ) -> AttachmentUploadResponse:
        """Upload image with feature flag check"""
        self._check_multimodal_enabled()
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")

            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")

            # Upload using media service
            result = await self.media_service.upload_image(
                file=file,
                file_name=file.filename,
                mime_type=file.content_type,
                message_id=message_id,
            )

            logger.info(f"User {self.user_id} uploaded image: {result.id}")
            return result

        except MediaServiceError as e:
            logger.error(f"Media service error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading image: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to upload image")

    async def upload_file_any(
        self, file: UploadFile, message_id: Optional[str] = None
    ) -> AttachmentUploadResponse:
        """Upload any file: images are validated and processed; other files stored as document."""
        self._check_multimodal_enabled()
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        file_name = file.filename
        mime_type = file.content_type or "application/octet-stream"
        is_image = mime_type.startswith("image/") and mime_type in getattr(
            self.media_service, "ALLOWED_IMAGE_TYPES", {}
        )
        try:
            if is_image:
                return await self.upload_image(file, message_id)
            return await self.media_service.upload_file(
                file=file,
                file_name=file_name,
                mime_type=mime_type,
                message_id=message_id,
            )
        except MediaServiceError as e:
            logger.error(f"Media service error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to upload file")

    async def get_attachment_access_url(
        self, attachment_id: str, expiration_minutes: int = 60
    ) -> AttachmentAccessResponse:
        """Generate a signed URL for accessing an attachment"""
        try:
            # Get attachment to verify it exists and check permissions
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Orphan attachments (not linked to a message) are not accessible
            if not attachment.message_id:
                raise HTTPException(
                    status_code=403,
                    detail="Attachment is not linked to a message and cannot be accessed",
                )
            access_granted = await self._check_attachment_access(
                attachment.message_id
            )
            if not access_granted:
                raise HTTPException(status_code=403, detail="Access denied")

            # Generate signed URL
            signed_url = await self.media_service.generate_signed_url(
                attachment_id, expiration_minutes
            )

            return AttachmentAccessResponse(
                download_url=signed_url,
                expires_in=expiration_minutes * 60,  # Convert to seconds
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating attachment access URL: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate access URL")

    async def download_attachment(self, attachment_id: str) -> StreamingResponse:
        """Download attachment file directly"""
        try:
            # Get attachment to verify it exists and check permissions
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Orphan attachments (not linked to a message) are not accessible
            if not attachment.message_id:
                raise HTTPException(
                    status_code=403,
                    detail="Attachment is not linked to a message and cannot be accessed",
                )
            access_granted = await self._check_attachment_access(
                attachment.message_id
            )
            if not access_granted:
                raise HTTPException(status_code=403, detail="Access denied")

            # Get the file data from storage
            file_data = await self.media_service.get_attachment_data(attachment_id)

            # Create streaming response
            io.BytesIO(file_data)

            # Set appropriate headers (sanitize filename to prevent header injection)
            safe_filename = _safe_content_disposition_filename(
                attachment.file_name
            )
            headers = {
                "Content-Disposition": f'inline; filename="{safe_filename}"',
                "Content-Type": attachment.mime_type,
                "Content-Length": str(len(file_data)),
            }

            return StreamingResponse(
                io.BytesIO(file_data), media_type=attachment.mime_type, headers=headers
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error downloading attachment: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to download attachment")

    async def get_attachment_info(self, attachment_id: str) -> AttachmentInfo:
        """Get attachment information"""
        try:
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Orphan attachments (not linked to a message) are not accessible
            if not attachment.message_id:
                raise HTTPException(
                    status_code=403,
                    detail="Attachment is not linked to a message and cannot be accessed",
                )
            access_granted = await self._check_attachment_access(
                attachment.message_id
            )
            if not access_granted:
                raise HTTPException(status_code=403, detail="Access denied")

            return AttachmentInfo(
                id=attachment.id,
                attachment_type=attachment.attachment_type,
                file_name=attachment.file_name,
                mime_type=attachment.mime_type,
                file_size=attachment.file_size,
                storage_provider=attachment.storage_provider,
                file_metadata=attachment.file_metadata,
                created_at=attachment.created_at,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting attachment info: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get attachment info")

    async def delete_attachment(self, attachment_id: str) -> dict:
        """Delete an attachment"""
        try:
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Orphan attachments (not linked to a message) cannot be deleted via API
            if not attachment.message_id:
                raise HTTPException(
                    status_code=403,
                    detail="Attachment is not linked to a message and cannot be accessed",
                )
            access_granted = await self._check_attachment_access(
                attachment.message_id, require_write=True
            )
            if not access_granted:
                raise HTTPException(status_code=403, detail="Access denied")

            success = await self.media_service.delete_attachment(attachment_id)
            if success:
                return {"message": "Attachment deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Attachment not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting attachment: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to delete attachment")

    async def get_message_attachments(self, message_id: str) -> List[AttachmentInfo]:
        """Get all attachments for a message"""
        try:
            # Check access to the message
            access_granted = await self._check_attachment_access(message_id)
            if not access_granted:
                raise HTTPException(status_code=403, detail="Access denied")

            attachments = await self.media_service.get_message_attachments(message_id)
            return attachments

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting message attachments: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to get message attachments"
            )

    async def _check_attachment_access(
        self, message_id: str, require_write: bool = False
    ) -> bool:
        """Check if user has access to the message containing the attachment"""
        try:
            # Get the message and its conversation
            from app.modules.conversations.message.message_model import Message

            message = self.db.query(Message).filter(Message.id == message_id).first()

            if not message:
                return False

            # Check conversation access
            access_level = await self.share_chat_service.check_conversation_access(
                message.conversation_id, self.user_email
            )

            if access_level == ConversationAccessType.NOT_FOUND:
                return False

            if require_write and access_level == ConversationAccessType.READ:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking attachment access: {str(e)}")
            return False

    async def test_multimodal_functionality(self, attachment_id: str) -> dict:
        """Test multimodal with feature flag check"""
        self._check_multimodal_enabled()
        try:
            attachment = await self.media_service.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            # Orphan attachments (not linked to a message) are not accessible
            if not attachment.message_id:
                raise HTTPException(
                    status_code=403,
                    detail="Attachment is not linked to a message and cannot be accessed",
                )
            access_granted = await self._check_attachment_access(
                attachment.message_id
            )
            if not access_granted:
                raise HTTPException(status_code=403, detail="Access denied")

            # Test multimodal functionality
            result = await self.media_service.test_multimodal_functionality(
                attachment_id
            )
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error testing multimodal functionality: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to test multimodal functionality"
            )
