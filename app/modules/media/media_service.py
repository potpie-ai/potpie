import logging
import hashlib
from typing import Optional, Dict, Any, List, Union
from io import BytesIO
import base64

from PIL import Image
import boto3
from botocore.exceptions import ClientError
from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from uuid6 import uuid7

from app.core.config_provider import config_provider
from app.modules.media.media_model import (
    MessageAttachment,
    AttachmentType,
    StorageProvider,
)
from app.modules.media.media_schema import AttachmentInfo, AttachmentUploadResponse
from app.modules.media.text_extraction_service import (
    TextExtractionService,
    TextExtractionError,
)
from app.modules.intelligence.provider.token_counter import get_token_counter

logger = logging.getLogger(__name__)


class MediaServiceError(Exception):
    """Base exception for MediaService errors"""

    pass


class MediaError:
    """Structured error codes for media operations"""
    MULTIMODAL_DISABLED = "MULTIMODAL_DISABLED"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INVALID_FILE = "INVALID_FILE"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    STORAGE_ERROR = "STORAGE_ERROR"
    NOT_FOUND = "NOT_FOUND"
    ACCESS_DENIED = "ACCESS_DENIED"
    PROCESSING_ERROR = "PROCESSING_ERROR"


def create_media_error(
    status_code: int,
    code: str,
    message: str,
    details: Optional[str] = None,
) -> HTTPException:
    """Create a structured HTTPException for media errors"""
    detail = {
        "error": message,
        "code": code,
    }
    if details:
        detail["details"] = details
    return HTTPException(status_code=status_code, detail=detail)


class MediaService:
    # Configuration constants
    ALLOWED_IMAGE_TYPES = {
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
        "image/gif": "GIF",
    }

    SUPPORTED_DOCUMENT_MIME_TYPES = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ]

    SUPPORTED_SPREADSHEET_MIME_TYPES = [
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ]

    CODE_FILE_EXTENSIONS = [
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".h",
        ".cs", ".rb", ".go", ".rs", ".php", ".swift", ".kt", ".scala",
        ".sh", ".bash", ".sql", ".r", ".m", ".mm", ".md", ".json", ".xml",
        ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg"
    ]

    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_DIMENSION = 2048  # Reduce from 4096 to preserve more detail in base64
    JPEG_QUALITY = 98  # Increase from 95 to preserve text readability
    # Add minimum dimension threshold to avoid over-compression of small images
    MIN_DIMENSION_FOR_RESIZE = 1024  # Only resize if larger than this

    def __init__(self, db: Session):
        self.db = db
        self.is_multimodal_enabled = config_provider.get_is_multimodal_enabled()

        self.storage_provider = StorageProvider.LOCAL
        self.bucket_name = None
        self.object_storage_descriptor: dict[str, Any] | None = None
        self.s3_client = None
        self.text_extraction_service = TextExtractionService()
        self.token_counter = get_token_counter()

        if self.is_multimodal_enabled:
            descriptor = config_provider.get_object_storage_descriptor()
            provider = descriptor["provider"]
            try:
                # Convert to uppercase to match enum values (GCS, S3, LOCAL, AZURE)
                self.storage_provider = StorageProvider(provider.upper())
            except ValueError as exc:  # fall back to safe default
                logger.error("Unsupported storage provider '%s': %s", provider, exc)
                raise MediaServiceError(
                    f"Unsupported storage provider configured: {provider}"
                ) from exc

            self.bucket_name = descriptor["bucket_name"]
            self.object_storage_descriptor = descriptor

            if self.storage_provider != StorageProvider.LOCAL:
                self._initialize_storage_client()
        else:
            logger.info(
                "Multimodal functionality disabled - cloud storage not initialized"
            )

    def _initialize_storage_client(self):
        if not self.object_storage_descriptor:
            raise MediaServiceError("Object storage descriptor not configured")

        try:
            client_kwargs = dict(self.object_storage_descriptor["client_kwargs"])
            self.s3_client = boto3.client("s3", **client_kwargs)
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(
                "Initialized boto3 client for provider %s (bucket=%s)",
                self.storage_provider.value,
                self.bucket_name,
            )
        except Exception as e:
            logger.error("Failed to initialize object storage client: %s", e)
            raise MediaServiceError(f"Failed to initialize object storage: {e}")

    def _check_multimodal_enabled(self):
        """Raise appropriate error if multimodal functionality is disabled"""
        if not self.is_multimodal_enabled:
            raise MediaServiceError("Multimodal functionality is disabled")

    def is_multimodal_available(self) -> bool:
        """Check if multimodal functionality is available"""
        return self.is_multimodal_enabled

    async def upload_image(
        self,
        file: Union[UploadFile, bytes],
        file_name: str,
        mime_type: str,
        message_id: Optional[str] = None,
    ) -> AttachmentUploadResponse:
        """Upload and process an image file"""
        self._check_multimodal_enabled()
        try:
            # Read file data
            # Use duck typing to check if it's an UploadFile-like object
            if hasattr(file, "read") and hasattr(file, "filename"):
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
            # await self._upload_to_gcs(storage_path, processed_image_data, mime_type)
            await self._upload_to_cloud(storage_path, processed_image_data, mime_type)

            # Create database record
            attachment = MessageAttachment(
                id=attachment_id,
                message_id=message_id,
                attachment_type=AttachmentType.IMAGE,
                file_name=file_name,
                file_size=len(processed_image_data),
                mime_type=mime_type,
                storage_path=storage_path,
                storage_provider=self.storage_provider,
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

    async def upload_document(
        self,
        file: Union[UploadFile, bytes],
        file_name: str,
        mime_type: str,
        message_id: Optional[str] = None,
        model: str = "openai/gpt-4o",  # For token counting
    ) -> AttachmentUploadResponse:
        """Upload and process a document file (PDF, DOCX, CSV, etc)."""
        self._check_multimodal_enabled()  # Reuse same feature flag

        try:
            # Read file data
            # Use duck typing to check if it's an UploadFile-like object
            if hasattr(file, "read") and hasattr(file, "filename"):
                file_data = await file.read()
                if not file_name:
                    file_name = file.filename or "unknown"
                if not mime_type:
                    mime_type = file.content_type or "application/octet-stream"
                if not isinstance(file_data, bytes):
                    raise MediaServiceError(
                        f"Expected bytes from file.read(), got {type(file_data)}"
                    )
            else:
                file_data = file
                if not isinstance(file_data, bytes):
                    raise MediaServiceError(f"Expected bytes, got {type(file_data)}")

            # Validate file size (use same limit as images for now: 10MB)
            if len(file_data) > self.MAX_IMAGE_SIZE:
                raise create_media_error(
                    400,
                    MediaError.FILE_TOO_LARGE,
                    "File size exceeds maximum allowed",
                    f"File: {len(file_data)} bytes, Limit: {self.MAX_IMAGE_SIZE} bytes",
                )

            # Determine attachment type
            attachment_type = self._determine_attachment_type(mime_type, file_name)

            # Extract text
            try:
                extracted_text, extraction_metadata = (
                    self.text_extraction_service.extract_text(
                        file_data, mime_type, file_name
                    )
                )
            except TextExtractionError as e:
                logger.error(f"Text extraction failed for {file_name}: {e}")
                raise create_media_error(
                    400,
                    MediaError.EXTRACTION_FAILED,
                    "Failed to extract text from document",
                    str(e),
                )

            # Count tokens
            token_count = self.token_counter.count_tokens(extracted_text, model)

            # Validate against context window (if conversation context provided)
            # Note: This validation happens AFTER extraction, so file is already processed
            # Frontend should call /validate-document first to prevent wasted processing
            logger.info(f"Document {file_name} extracted with {token_count} tokens")

            # Generate unique attachment ID
            attachment_id = str(uuid7())

            # Generate storage path for original file
            storage_path = self._generate_storage_path(attachment_id, file_name)

            # Upload original file to cloud storage
            await self._upload_to_cloud(storage_path, file_data, mime_type)

            # Prepare file metadata
            file_metadata = {
                "original_size": len(file_data),
                "extracted_text_length": len(extracted_text),
                "token_count": token_count,
                **extraction_metadata,
            }

            # Store extracted text based on size
            if self.text_extraction_service.should_store_inline(extracted_text):
                # Store in JSONB
                file_metadata["extracted_text"] = extracted_text
                file_metadata["text_storage"] = "inline"
                logger.info(
                    f"Storing extracted text inline for {attachment_id} ({len(extracted_text)} chars)"
                )
            else:
                # Store in S3 as separate file
                extracted_text_path = f"{storage_path}.extracted.txt"
                await self._upload_to_cloud(
                    extracted_text_path, extracted_text.encode("utf-8"), "text/plain"
                )
                file_metadata["extracted_text_path"] = extracted_text_path
                file_metadata["text_storage"] = "s3"
                logger.info(
                    f"Storing extracted text in S3 for {attachment_id} ({len(extracted_text)} chars)"
                )

            # Create database record
            attachment = MessageAttachment(
                id=attachment_id,
                message_id=message_id,
                attachment_type=attachment_type,
                file_name=file_name,
                file_size=len(file_data),
                mime_type=mime_type,
                storage_path=storage_path,
                storage_provider=self.storage_provider,
                file_metadata=file_metadata,
            )

            self.db.add(attachment)
            self.db.commit()

            logger.info(
                f"Successfully uploaded document {attachment_id} with {token_count} tokens"
            )

            return AttachmentUploadResponse(
                id=attachment_id,
                attachment_type=attachment_type,
                file_name=file_name,
                mime_type=mime_type,
                file_size=len(file_data),
                token_count=token_count,
            )

        except HTTPException:
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error uploading document: {str(e)}", exc_info=True)
            if isinstance(e, MediaServiceError):
                raise
            raise MediaServiceError(f"Failed to upload document: {str(e)}")

    def _determine_attachment_type(
        self, mime_type: str, file_name: str
    ) -> AttachmentType:
        """Determine attachment type from MIME type and filename."""
        if "pdf" in mime_type:
            return AttachmentType.PDF
        elif (
            "word" in mime_type
            or "vnd.openxmlformats-officedocument.wordprocessingml" in mime_type
        ):
            return AttachmentType.DOCUMENT
        elif (
            "csv" in mime_type
            or "spreadsheet" in mime_type
            or "vnd.openxmlformats-officedocument.spreadsheetml" in mime_type
        ):
            return AttachmentType.SPREADSHEET
        elif mime_type.startswith("text/") or self._is_code_file(file_name):
            return AttachmentType.CODE
        else:
            return AttachmentType.DOCUMENT  # Fallback

    def _is_code_file(self, file_name: str) -> bool:
        """Check if filename suggests code file."""
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".cs",
            ".rb",
            ".go",
            ".rs",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".sh",
            ".bash",
            ".sql",
            ".r",
            ".m",
            ".mm",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".conf",
            ".cfg",
        }
        extension = "." + file_name.split(".")[-1].lower() if "." in file_name else ""
        return extension in code_extensions

    async def get_extracted_text(self, attachment_id: str) -> str:
        """Retrieve extracted text from attachment (JSONB or S3)."""
        try:
            attachment = await self.get_attachment(attachment_id)
            if not attachment:
                raise HTTPException(status_code=404, detail="Attachment not found")

            metadata = attachment.file_metadata or {}

            # Check inline storage first
            if metadata.get("text_storage") == "inline":
                return metadata.get("extracted_text", "")

            # Fetch from S3
            elif metadata.get("text_storage") == "s3":
                extracted_text_path = metadata.get("extracted_text_path")
                if not extracted_text_path:
                    raise MediaServiceError("Extracted text path not found in metadata")

                try:
                    response = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=extracted_text_path,
                    )
                    text_bytes = response["Body"].read()
                    return text_bytes.decode("utf-8")
                except Exception as e:
                    logger.error(f"Failed to fetch extracted text from S3: {e}")
                    raise MediaServiceError(
                        f"Failed to retrieve extracted text: {str(e)}"
                    )

            else:
                # No extracted text available
                return ""

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting extracted text for {attachment_id}: {str(e)}")
            raise MediaServiceError(f"Failed to get extracted text: {str(e)}")

    def _validate_image(self, file_data: bytes, mime_type: str) -> None:
        """Validate image file"""
        # Check MIME type
        if mime_type not in self.ALLOWED_IMAGE_TYPES:
            raise create_media_error(
                400,
                MediaError.UNSUPPORTED_FORMAT,
                f"Unsupported image type: {mime_type}",
                f"Allowed types: {', '.join(self.ALLOWED_IMAGE_TYPES.keys())}"
            )

        # Check file size
        if len(file_data) > self.MAX_IMAGE_SIZE:
            raise create_media_error(
                400,
                MediaError.FILE_TOO_LARGE,
                "Image size exceeds maximum allowed",
                f"File: {len(file_data)} bytes, Limit: {self.MAX_IMAGE_SIZE} bytes"
            )

        # Verify it's actually a valid image
        try:
            img = Image.open(BytesIO(file_data))
            img.verify()
        except Exception as e:
            logger.warning(f"Image validation failed: {str(e)}")
            raise create_media_error(
                400,
                MediaError.INVALID_FILE,
                "Invalid or corrupted image file",
                str(e)
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

    async def _upload_to_cloud(
        self, storage_path: str, file_data: bytes, mime_type: str
    ) -> None:
        """Upload file to the configured object storage bucket via boto3."""
        if not self.s3_client or not self.bucket_name:
            raise MediaServiceError("Object storage client not initialized for upload")

        try:
            logger.info(
                "Uploading object -> provider=%s bucket=%s key=%s content_type=%s",
                self.storage_provider.value,
                self.bucket_name,
                storage_path,
                mime_type,
            )
            md5_b64 = base64.b64encode(hashlib.md5(file_data).digest()).decode("utf-8")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=storage_path,
                Body=file_data,
                ContentType=mime_type,
                ContentMD5=md5_b64,
            )
            logger.info("Uploaded object to %s:%s", self.bucket_name, storage_path)
        except Exception as e:
            logger.error("Failed to upload to object storage: %s", e)
            raise MediaServiceError(f"Failed to upload to object storage: {e}")

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

            if not self.s3_client or not self.bucket_name:
                raise MediaServiceError(
                    "Object storage client not initialized for download"
                )

            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=attachment.storage_path,
                )
                return response["Body"].read()
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code == "NoSuchKey":
                    raise HTTPException(
                        status_code=404, detail="Attachment not found in storage"
                    )
                raise

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

            if not self.s3_client or not self.bucket_name:
                raise MediaServiceError(
                    "Object storage client not initialized for signed URL"
                )

            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": attachment.storage_path,
                },
                ExpiresIn=expiration_minutes * 60,
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

            try:
                if not self.s3_client or not self.bucket_name:
                    raise MediaServiceError(
                        "Object storage client not initialized for delete"
                    )

                try:
                    self.s3_client.delete_object(
                        Bucket=self.bucket_name,
                        Key=attachment.storage_path,
                    )
                    logger.info(
                        "Deleted object from %s:%s",
                        self.bucket_name,
                        attachment.storage_path,
                    )
                except ClientError as e:
                    if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                        logger.warning(
                            "File not found in bucket=%s key=%s; continuing with DB delete",
                            self.bucket_name,
                            attachment.storage_path,
                        )
                    else:
                        raise
            except Exception as e:
                logger.error(f"Failed to delete from cloud storage: {str(e)}")
                raise MediaServiceError(
                    f"Failed to delete from cloud storage: {str(e)}"
                )

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
        self._check_multimodal_enabled()
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

    def get_supported_formats(self) -> dict:
        """Return supported file formats and constraints"""
        return {
            "images": {
                "mime_types": list(self.ALLOWED_IMAGE_TYPES.keys()),
                "extensions": [".jpg", ".jpeg", ".png", ".webp", ".gif"],
                "max_size_bytes": self.MAX_IMAGE_SIZE,
                "description": "Image files for visual analysis",
            },
            "documents": {
                "mime_types": self.SUPPORTED_DOCUMENT_MIME_TYPES,
                "extensions": [".pdf", ".docx", ".doc"],
                "max_size_bytes": self.MAX_IMAGE_SIZE,
                "description": "Document files with text extraction",
            },
            "spreadsheets": {
                "mime_types": self.SUPPORTED_SPREADSHEET_MIME_TYPES,
                "extensions": [".csv", ".xlsx", ".xls"],
                "max_size_bytes": self.MAX_IMAGE_SIZE,
                "description": "Spreadsheet files with data extraction",
            },
            "code_files": {
                "mime_types": ["text/plain", "application/json"],
                "extensions": self.CODE_FILE_EXTENSIONS,
                "max_size_bytes": self.MAX_IMAGE_SIZE,
                "description": "Source code and configuration files",
            },
            "max_file_size_bytes": self.MAX_IMAGE_SIZE,
        }
