import os
import logging
from typing import Optional, Dict, Any, List, Union
from io import BytesIO
import base64

from PIL import Image
from google.cloud import storage
from google.cloud.exceptions import NotFound
from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from uuid6 import uuid7

from app.modules.media.media_model import (
    MessageAttachment,
    AttachmentType,
    StorageProvider,
)
from app.modules.media.media_schema import AttachmentInfo, AttachmentUploadResponse

logger = logging.getLogger(__name__)


class MediaServiceError(Exception):
    """Base exception for MediaService errors"""

    pass


class MediaService:
    # Configuration constants
    ALLOWED_IMAGE_TYPES = {
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
        "image/gif": "GIF",
    }

    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_DIMENSION = 2048  # Reduce from 4096 to preserve more detail in base64
    JPEG_QUALITY = 98  # Increase from 95 to preserve text readability
    # Add minimum dimension threshold to avoid over-compression of small images
    MIN_DIMENSION_FOR_RESIZE = 1024  # Only resize if larger than this

    def __init__(self, db: Session):
        self.db = db
        self.bucket_name = os.getenv("GCS_BUCKET_NAME", "potpie-media-attachments")
        self.gcs_project_id = os.getenv("GCS_PROJECT_ID")

        # Initialize GCS client with service account credentials for signed URLs
        try:
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

            if service_account_path and os.path.exists(service_account_path):
                # Use service account credentials for signed URL support
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path
                )
                self.gcs_client = storage.Client(
                    credentials=credentials, project=self.gcs_project_id
                )
                logger.info(
                    "Initialized GCS client with service account credentials (signed URLs supported)"
                )
            else:
                # Fallback to default credentials (signed URLs will fail)
                self.gcs_client = storage.Client(project=self.gcs_project_id)
                logger.warning(
                    "GCS client initialized with default credentials (signed URLs not supported)"
                )

            self.bucket = self.gcs_client.bucket(self.bucket_name)
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {str(e)}")
            raise MediaServiceError(f"Failed to initialize cloud storage: {str(e)}")

    async def upload_image(
        self,
        file: Union[UploadFile, bytes],
        file_name: str,
        mime_type: str,
        message_id: Optional[str] = None,
    ) -> AttachmentUploadResponse:
        """Upload and process an image file"""
        try:
            # Read file data
            if isinstance(file, UploadFile):
                file_data = await file.read()
                if not file_name:
                    file_name = file.filename or "unknown"
                if not mime_type:
                    mime_type = file.content_type or "application/octet-stream"
                # Validate that we actually got bytes
                if not isinstance(file_data, bytes):
                    raise MediaServiceError(
                        f"Expected bytes from file.read(), got {type(file_data)}"
                    )
            else:
                file_data = file
                # Validate that file_data is bytes
                if not isinstance(file_data, bytes):
                    raise MediaServiceError(f"Expected bytes, got {type(file_data)}")

            # Validate the image
            logger.debug(
                f"Validating image data of type {type(file_data)} with size {len(file_data) if isinstance(file_data, bytes) else 'unknown'}"
            )
            self._validate_image(file_data, mime_type)

            # Generate unique attachment ID
            attachment_id = str(uuid7())

            # Process the image
            logger.debug(f"Processing image data of type {type(file_data)}")
            processed_image_data, file_metadata = await self._process_image(
                file_data, mime_type
            )

            # Generate storage path
            storage_path = self._generate_storage_path(attachment_id, file_name)

            # Upload to GCS
            await self._upload_to_gcs(storage_path, processed_image_data, mime_type)

            # Create database record
            attachment = MessageAttachment(
                id=attachment_id,
                message_id=message_id,
                attachment_type=AttachmentType.IMAGE,
                file_name=file_name,
                file_size=len(processed_image_data),
                mime_type=mime_type,
                storage_path=storage_path,
                storage_provider=StorageProvider.GCS,
                file_metadata=file_metadata,
            )

            self.db.add(attachment)
            self.db.commit()

            logger.info(
                f"Successfully uploaded image {attachment_id} to {storage_path}"
            )

            return AttachmentUploadResponse(
                id=attachment_id,
                attachment_type=AttachmentType.IMAGE,
                file_name=file_name,
                mime_type=mime_type,
                file_size=len(processed_image_data),
            )

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error uploading image: {str(e)}")
            if isinstance(e, (MediaServiceError, HTTPException)):
                raise
            raise MediaServiceError(f"Failed to upload image: {str(e)}")

    def _validate_image(self, file_data: bytes, mime_type: str) -> None:
        """Validate image file"""
        # Check MIME type
        if mime_type not in self.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type: {mime_type}. Allowed types: {', '.join(self.ALLOWED_IMAGE_TYPES.keys())}",
            )

        # Check file size
        if len(file_data) > self.MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image size ({len(file_data)} bytes) exceeds maximum allowed ({self.MAX_IMAGE_SIZE} bytes)",
            )

        # Verify it's actually a valid image
        try:
            img = Image.open(BytesIO(file_data))
            img.verify()
        except Exception as e:
            logger.warning(f"Image validation failed: {str(e)}")
            raise HTTPException(
                status_code=400, detail="Invalid or corrupted image file"
            )

    async def _process_image(
        self, file_data: bytes, mime_type: str
    ) -> tuple[bytes, Dict[str, Any]]:
        """Process and potentially resize image"""
        try:
            img = Image.open(BytesIO(file_data))

            # Store original file_metadata
            file_metadata = {
                "original_width": img.width,
                "original_height": img.height,
                "original_format": img.format,
                "original_mode": img.mode,
                "original_size": len(file_data),
            }

            # Check if resizing is needed
            needs_resize = (
                img.width > self.MAX_DIMENSION or img.height > self.MAX_DIMENSION
            )

            if needs_resize:
                # Only resize if the image is significantly larger than minimum threshold
                if max(img.width, img.height) > self.MIN_DIMENSION_FOR_RESIZE:
                    # Calculate new dimensions maintaining aspect ratio
                    img.thumbnail(
                        (self.MAX_DIMENSION, self.MAX_DIMENSION),
                        Image.Resampling.LANCZOS,
                    )
                    file_metadata["resized"] = True
                    file_metadata["new_width"] = img.width
                    file_metadata["new_height"] = img.height
                    logger.info(
                        f"Resized image from {file_metadata['original_width']}x{file_metadata['original_height']} to {img.width}x{img.height}"
                    )
                else:
                    logger.info(
                        f"Skipping resize for small image: {img.width}x{img.height}"
                    )
                    file_metadata["resized"] = False

            # Convert to RGB if necessary (for JPEG output)
            if img.mode not in ("RGB", "L"):
                if img.mode == "RGBA":
                    # Create a white background for transparent images
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(
                        img, mask=img.split()[-1]
                    )  # Use alpha channel as mask
                    img = background
                else:
                    img = img.convert("RGB")
                file_metadata["mode_converted"] = True

            # Save processed image
            output = BytesIO()
            save_format = self.ALLOWED_IMAGE_TYPES.get(mime_type, "JPEG")

            if save_format == "JPEG":
                img.save(
                    output, format=save_format, quality=self.JPEG_QUALITY, optimize=True
                )
            else:
                img.save(output, format=save_format, optimize=True)

            processed_data = output.getvalue()
            file_metadata["processed_size"] = len(processed_data)

            return processed_data, file_metadata

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise MediaServiceError(f"Failed to process image: {str(e)}")

    def _generate_storage_path(self, attachment_id: str, file_name: str) -> str:
        """Generate a unique storage path for the file"""
        # Extract file extension
        file_extension = ""
        if "." in file_name:
            file_extension = file_name.split(".")[-1].lower()

        # Create path: attachments/year/month/attachment_id.extension
        from datetime import datetime

        now = datetime.utcnow()
        path = f"attachments/{now.year:04d}/{now.month:02d}/{attachment_id}"

        if file_extension:
            path += f".{file_extension}"

        return path

    async def _upload_to_gcs(
        self, storage_path: str, file_data: bytes, mime_type: str
    ) -> None:
        """Upload file to Google Cloud Storage"""
        try:
            blob = self.bucket.blob(storage_path)
            blob.upload_from_string(file_data, content_type=mime_type)
            logger.info(f"Successfully uploaded to GCS: {storage_path}")
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {str(e)}")
            raise MediaServiceError(f"Failed to upload to cloud storage: {str(e)}")

    async def get_attachment(self, attachment_id: str) -> Optional[MessageAttachment]:
        """Get attachment record by ID"""
        try:
            attachment = (
                self.db.query(MessageAttachment)
                .filter(MessageAttachment.id == attachment_id)
                .first()
            )
            return attachment
        except Exception as e:
            logger.error(f"Error retrieving attachment {attachment_id}: {str(e)}")
            raise MediaServiceError(f"Failed to retrieve attachment: {str(e)}")

    async def get_attachment_data(self, attachment_id: str) -> bytes:
        """Download attachment data from storage"""
        try:
            attachment = await self.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            if attachment.storage_provider == StorageProvider.GCS:
                blob = self.bucket.blob(attachment.storage_path)
                return blob.download_as_bytes()
            else:
                raise MediaServiceError(
                    f"Unsupported storage provider: {attachment.storage_provider}"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error downloading attachment {attachment_id}: {str(e)}")
            raise MediaServiceError(f"Failed to download attachment: {str(e)}")

    async def generate_signed_url(
        self, attachment_id: str, expiration_minutes: int = 60
    ) -> str:
        """Generate a signed URL for secure access to attachment"""
        try:
            attachment = await self.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            if attachment.storage_provider == StorageProvider.GCS:
                blob = self.bucket.blob(attachment.storage_path)

                # Generate signed URL
                from datetime import timedelta

                url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(minutes=expiration_minutes),
                    method="GET",
                    response_disposition=f'inline; filename="{attachment.file_name}"',
                )
                return url
            else:
                raise MediaServiceError(
                    f"Unsupported storage provider: {attachment.storage_provider}"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Error generating signed URL for attachment {attachment_id}: {str(e)}"
            )
            raise MediaServiceError(f"Failed to generate access URL: {str(e)}")

    async def delete_attachment(self, attachment_id: str) -> bool:
        """Delete attachment from storage and database"""
        try:
            attachment = await self.get_attachment(attachment_id)
            if not attachment:
                return False

            # Delete from storage
            if attachment.storage_provider == StorageProvider.GCS:
                try:
                    blob = self.bucket.blob(attachment.storage_path)
                    blob.delete()
                    logger.info(f"Deleted from GCS: {attachment.storage_path}")
                except NotFound:
                    logger.warning(f"File not found in GCS: {attachment.storage_path}")

            # Delete from database
            self.db.delete(attachment)
            self.db.commit()

            logger.info(f"Successfully deleted attachment {attachment_id}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting attachment {attachment_id}: {str(e)}")
            raise MediaServiceError(f"Failed to delete attachment: {str(e)}")

    async def update_message_attachments(
        self, message_id: str, attachment_ids: List[str]
    ) -> None:
        """Update message to link with uploaded attachments"""
        try:
            # Update attachments with message_id
            if attachment_ids:
                # Update attachments with message_id
                updated_count = (
                    self.db.query(MessageAttachment)
                    .filter(
                        MessageAttachment.id.in_(attachment_ids),
                        MessageAttachment.message_id.is_(None),
                    )
                    .update(
                        {MessageAttachment.message_id: message_id},
                        synchronize_session=False,
                    )
                )

                # Update message has_attachments flag
                from app.modules.conversations.message.message_model import Message

                message = (
                    self.db.query(Message).filter(Message.id == message_id).first()
                )
                if message:
                    message.has_attachments = True

                self.db.commit()
                logger.info(
                    f"Updated message {message_id} with {len(attachment_ids)} attachments"
                )

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating message attachments: {str(e)}")
            raise MediaServiceError(f"Failed to update message attachments: {str(e)}")

    async def get_message_attachments(
        self, message_id: str, include_download_urls: bool = True
    ) -> List[AttachmentInfo]:
        """Get all attachments for a specific message with optional signed URLs"""
        try:
            attachments = (
                self.db.query(MessageAttachment).filter_by(message_id=message_id).all()
            )

            result = []
            for attachment in attachments:
                download_url = None

                # Generate signed URL if requested
                if include_download_urls:
                    try:
                        download_url = await self.generate_signed_url(
                            attachment.id, expiration_minutes=60
                        )
                        logger.info(
                            f"Generated signed URL for attachment {attachment.id}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate signed URL for attachment {attachment.id}: {str(e)}"
                        )
                        # Fallback to direct download endpoint
                        download_url = f"/api/media/{attachment.id}/download"
                        logger.info(
                            f"Using fallback download URL for attachment {attachment.id}: {download_url}"
                        )

                result.append(
                    AttachmentInfo(
                        id=attachment.id,
                        attachment_type=attachment.attachment_type,
                        file_name=attachment.file_name,
                        file_size=attachment.file_size,
                        mime_type=attachment.mime_type,
                        storage_provider=attachment.storage_provider,
                        created_at=attachment.created_at,
                        file_metadata=attachment.file_metadata or {},
                        download_url=download_url,
                    )
                )

            return result
        except Exception as e:
            logger.error(
                f"Error getting message attachments for message {message_id}: {str(e)}"
            )
            raise MediaServiceError(f"Failed to get message attachments: {str(e)}")

    async def get_image_as_base64(self, attachment_id: str) -> str:
        """Get image as base64 string for LLM processing"""
        try:
            attachment = await self.get_attachment(attachment_id)
            if not attachment:
                raise MediaServiceError(f"Attachment {attachment_id} not found")

            if attachment.attachment_type != AttachmentType.IMAGE:
                raise MediaServiceError(f"Attachment {attachment_id} is not an image")

            # Get image data from storage
            image_data = await self.get_attachment_data(attachment_id)

            # Convert to base64
            base64_string = base64.b64encode(image_data).decode("utf-8")

            logger.info(
                f"Converted image {attachment_id} to base64 (size: {len(base64_string)} chars)"
            )
            return base64_string

        except Exception as e:
            logger.error(f"Error converting image {attachment_id} to base64: {str(e)}")
            raise MediaServiceError(f"Failed to convert image to base64: {str(e)}")

    async def get_message_images_as_base64(
        self, message_id: str
    ) -> Dict[str, Dict[str, Union[str, int]]]:
        """Get all images for a message as base64 dict with metadata"""
        try:
            attachments = await self.get_message_attachments(
                message_id, include_download_urls=False
            )
            image_attachments = [
                att
                for att in attachments
                if att.attachment_type == AttachmentType.IMAGE
            ]

            if not image_attachments:
                return {}

            result = {}
            for attachment in image_attachments:
                try:
                    base64_data = await self.get_image_as_base64(attachment.id)
                    result[attachment.id] = {
                        "base64": base64_data,
                        "mime_type": attachment.mime_type,
                        "file_name": attachment.file_name,
                        "file_size": attachment.file_size,
                    }
                except Exception as e:
                    logger.error(
                        f"Failed to convert attachment {attachment.id} to base64: {str(e)}"
                    )
                    # Continue with other images even if one fails
                    continue

            logger.info(
                f"Converted {len(result)} images to base64 for message {message_id}"
            )
            return result

        except Exception as e:
            logger.error(
                f"Error getting message images as base64 for message {message_id}: {str(e)}"
            )
            raise MediaServiceError(f"Failed to get message images as base64: {str(e)}")

    async def get_conversation_recent_images(
        self, conversation_id: str, limit: int = 10
    ) -> Dict[str, Dict[str, Union[str, int]]]:
        """Get recent images from conversation history for multimodal context"""
        try:
            from app.modules.conversations.message.message_model import (
                Message,
                MessageStatus,
                MessageType,
            )

            # Get recent messages with attachments
            recent_messages = (
                self.db.query(Message)
                .filter_by(conversation_id=conversation_id, status=MessageStatus.ACTIVE)
                .filter(Message.has_attachments == True)
                .filter(Message.type == MessageType.HUMAN)  # Only user messages
                .order_by(Message.created_at.desc())
                .limit(limit)
                .all()
            )

            all_images = {}
            for message in recent_messages:
                try:
                    message_images = await self.get_message_images_as_base64(message.id)
                    all_images.update(message_images)
                except Exception as e:
                    logger.error(
                        f"Failed to get images for message {message.id}: {str(e)}"
                    )
                    continue

            logger.info(
                f"Retrieved {len(all_images)} recent images from conversation {conversation_id}"
            )
            return all_images

        except Exception as e:
            logger.error(
                f"Error getting recent images for conversation {conversation_id}: {str(e)}"
            )
            raise MediaServiceError(
                f"Failed to get recent conversation images: {str(e)}"
            )

    async def test_multimodal_functionality(self, attachment_id: str) -> Dict[str, Any]:
        """Test method to verify multimodal functionality"""
        try:
            # Get attachment info
            attachment = await self.get_attachment(attachment_id)
            if not attachment:
                return {"error": "Attachment not found"}

            # Test base64 conversion
            base64_data = await self.get_image_as_base64(attachment_id)

            return {
                "status": "success",
                "attachment_id": attachment_id,
                "file_name": attachment.file_name,
                "mime_type": attachment.mime_type,
                "file_size": attachment.file_size,
                "base64_length": len(base64_data),
                "base64_preview": (
                    base64_data[:50] + "..." if len(base64_data) > 50 else base64_data
                ),
                "multimodal_ready": True,
            }

        except Exception as e:
            logger.error(f"Error testing multimodal functionality: {str(e)}")
            return {"status": "error", "error": str(e), "multimodal_ready": False}
