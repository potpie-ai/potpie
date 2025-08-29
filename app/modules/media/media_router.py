from typing import List, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.media.media_controller import MediaController
from app.modules.media.media_schema import (
    AttachmentUploadResponse,
    AttachmentAccessResponse,
    AttachmentInfo,
)


router = APIRouter()


class MediaAPI:

    @staticmethod
    @router.post("/media/upload", response_model=AttachmentUploadResponse)
    async def upload_image(
        file: UploadFile = File(..., description="Image file to upload"),
        message_id: Optional[str] = Query(
            None, description="Optional message ID to link attachment"
        ),
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        """
        Upload an image file.

        - **file**: Image file (JPEG, PNG, WebP, GIF)
        - **message_id**: Optional message ID to immediately link the attachment
        - Returns attachment info with unique ID

        The uploaded file will be:
        1. Validated for type and size
        2. Processed (resized if needed)
        3. Stored in Google Cloud Storage
        4. Linked to message if message_id provided
        """
        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)
        return await controller.upload_image(file, message_id)

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
        """
        Test multimodal functionality for an attachment.

        - **attachment_id**: Unique attachment identifier
        - Returns test results showing if the attachment can be used for multimodal AI

        This endpoint tests:
        1. Image retrieval from storage
        2. Base64 conversion for LLM processing
        3. Multimodal readiness status
        """
        user_id = user["user_id"]
        user_email = user["email"]

        controller = MediaController(db, user_id, user_email)
        return await controller.test_multimodal_functionality(attachment_id)
