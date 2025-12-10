from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.core.database import get_db, get_async_db
from app.modules.auth.auth_service import AuthService
from app.modules.media.media_controller import MediaController
from app.modules.media.media_schema import (
    AttachmentUploadResponse,
    AttachmentAccessResponse,
    AttachmentInfo,
    SupportedFormatsResponse,
)
from app.modules.media.media_service import MediaService


router = APIRouter()


class MediaAPI:
    @staticmethod
    @router.post("/media/upload", response_model=AttachmentUploadResponse)
    async def upload_attachment(
        file: UploadFile = File(..., description="File to upload (image or document)"),
        message_id: Optional[str] = Query(
            None, description="Optional message ID to link attachment"
        ),
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        """Upload an image or document file with multimodal feature flag check"""

        # Check if multimodal functionality is enabled
        if not config_provider.get_is_multimodal_enabled():
            raise HTTPException(
                status_code=501,  # Not Implemented
                detail={
                    "error": "Multimodal functionality is currently disabled",
                    "code": "MULTIMODAL_DISABLED",
                    "message": "File upload is not available in this deployment configuration",
                },
            )

        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)

        # Determine if image or document based on MIME type
        mime_type = file.content_type or "application/octet-stream"

        if mime_type.startswith("image/"):
            return await controller.upload_image(file, message_id)
        else:
            return await controller.upload_document(file, message_id)

    @staticmethod
    @router.post("/media/validate-document")
    async def validate_document_upload(
        conversation_id: str = Form(...),
        file_size: int = Form(...),
        file_name: str = Form(...),
        mime_type: str = Form(...),
        db: Session = Depends(get_db),
        async_db=Depends(get_async_db),
        user=Depends(AuthService.check_auth),
    ):
        """
        Validate if a document can be uploaded without exceeding context limits.

        This endpoint should be called BEFORE uploading the actual file.
        """
        # Check if multimodal functionality is enabled
        if not config_provider.get_is_multimodal_enabled():
            raise HTTPException(
                status_code=501,
                detail={
                    "error": "Multimodal functionality is currently disabled",
                    "code": "MULTIMODAL_DISABLED",
                    "message": "Document validation is not available in this deployment configuration",
                },
            )

        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)
        return await controller.validate_document_upload(
            conversation_id=conversation_id,
            file_size=file_size,
            file_name=file_name,
            mime_type=mime_type,
            async_db=async_db,
        )

    @staticmethod
    @router.get(
        "/media/{attachment_id}/access", response_model=AttachmentAccessResponse
    )
    async def get_attachment_access_url(
        attachment_id: str,
        expiration_minutes: int = Query(
            60, ge=1, le=1440, description="URL expiration time in minutes"
        ),
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        """
        Get a signed URL for accessing an attachment.

        - **attachment_id**: Unique attachment identifier
        - **expiration_minutes**: URL expiration time (1-1440 minutes, default 60)
        - Returns a signed URL that can be used to download the attachment

        Access is granted only if user has permission to view the conversation
        containing the attachment.
        """
        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)
        return await controller.get_attachment_access_url(
            attachment_id, expiration_minutes
        )

    @staticmethod
    @router.get("/media/{attachment_id}/download")
    async def download_attachment(
        attachment_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        """
        Direct download of an attachment file.

        - **attachment_id**: Unique attachment identifier
        - Returns the file content directly with appropriate headers

        This endpoint serves as a fallback when signed URLs are not available.
        Access is granted only if user has permission to view the conversation
        containing the attachment.
        """
        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)
        return await controller.download_attachment(attachment_id)

    @staticmethod
    @router.get("/media/{attachment_id}/info", response_model=AttachmentInfo)
    async def get_attachment_info(
        attachment_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        """
        Get attachment metadata and information.

        - **attachment_id**: Unique attachment identifier
        - Returns attachment metadata (filename, size, type, etc.)

        Access is granted only if user has permission to view the conversation
        containing the attachment.
        """
        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)
        return await controller.get_attachment_info(attachment_id)

    @staticmethod
    @router.delete("/media/{attachment_id}")
    async def delete_attachment(
        attachment_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        """
        Delete an attachment.

        - **attachment_id**: Unique attachment identifier
        - Removes attachment from storage and database

        Delete access is granted only if user has write permission to the conversation
        containing the attachment.
        """
        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)
        return await controller.delete_attachment(attachment_id)

    @staticmethod
    @router.get(
        "/messages/{message_id}/attachments", response_model=List[AttachmentInfo]
    )
    async def get_message_attachments(
        message_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        """
        Get all attachments for a specific message.

        - **message_id**: Unique message identifier
        - Returns list of attachment information

        Access is granted only if user has permission to view the conversation
        containing the message.
        """
        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)
        return await controller.get_message_attachments(message_id)

    @staticmethod
    @router.get("/media/{attachment_id}/test-multimodal")
    async def test_multimodal_functionality(
        attachment_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        """Test multimodal functionality for an attachment with feature flag check"""

        # Check if multimodal functionality is enabled
        if not config_provider.get_is_multimodal_enabled():
            raise HTTPException(
                status_code=501,  # Not Implemented
                detail={
                    "error": "Multimodal functionality is currently disabled",
                    "code": "MULTIMODAL_DISABLED",
                    "message": "Multimodal testing is not available in this deployment configuration",
                },
            )

        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)
        return await controller.test_multimodal_functionality(attachment_id)

    @staticmethod
    @router.get("/media/supported-formats", response_model=SupportedFormatsResponse)
    async def get_supported_formats(
        db: Session = Depends(get_db),
    ):
        """
        Get supported file formats and constraints.

        Returns information about:
        - Supported image types (JPEG, PNG, WebP, GIF)
        - Supported document types (PDF, DOCX)
        - Supported spreadsheet types (CSV, XLSX)
        - Supported code file extensions
        - Maximum file size limits

        This endpoint does not require authentication.
        """
        service = MediaService(db)
        return service.get_supported_formats()

    @staticmethod
    @router.post("/media/admin/cleanup-orphans")
    async def trigger_orphan_cleanup(
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        """
        Manually trigger orphan attachment cleanup.

        This is an admin endpoint for testing/monitoring.
        In production, cleanup runs automatically via Celery beat.
        """
        # TODO: Add admin role check
        from app.modules.media.attachment_cleanup_service import AttachmentCleanupService

        cleanup_service = AttachmentCleanupService(db)

        # First get stats
        stats = cleanup_service.get_orphan_stats()

        # Then run cleanup
        result = cleanup_service.cleanup_orphaned_attachments()

        return {
            "before": stats,
            "result": result,
        }
