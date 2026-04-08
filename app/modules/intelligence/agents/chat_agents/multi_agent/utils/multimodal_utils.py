"""Multimodal utility functions for multi-agent system"""

import base64
import mimetypes
from typing import List, Optional, Sequence, Union, Dict

from pydantic_ai.messages import DocumentUrl, ImageUrl, UserContent

from app.modules.intelligence.agents.chat_agent import ChatContext
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Keys must stay aligned with pydantic_ai.messages._document_format_lookup (file input for OpenAI).
_SUPPORTED_DOCUMENT_MIMES = frozenset(
    {
        "application/pdf",
        "text/plain",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/html",
        "text/markdown",
        "application/msword",
        "application/vnd.ms-excel",
    }
)

_EXT_TO_MIME = {
    "pdf": "application/pdf",
    "txt": "text/plain",
    "csv": "text/csv",
    "md": "text/markdown",
    "html": "text/html",
    "htm": "text/html",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "doc": "application/msword",
    "xls": "application/vnd.ms-excel",
}


def coerce_document_mime(mime_type: Optional[str], file_name: str) -> Optional[str]:
    """Return a MIME type supported by pydantic-ai DocumentUrl / OpenAI file input, or None."""
    m = (mime_type or "").strip().lower()
    if m in _SUPPORTED_DOCUMENT_MIMES:
        return m
    guessed, _ = mimetypes.guess_type(file_name or "")
    if guessed and guessed in _SUPPORTED_DOCUMENT_MIMES:
        return guessed
    ext = ""
    if file_name and "." in file_name:
        ext = file_name.rsplit(".", 1)[-1].lower()
    if ext in _EXT_TO_MIME:
        return _EXT_TO_MIME[ext]
    if m == "application/octet-stream" and ext in _EXT_TO_MIME:
        return _EXT_TO_MIME[ext]
    return None


def validate_and_build_data_url(image_data: dict, attachment_id: str) -> Optional[str]:
    """
    Validate image data and build a data URL if valid.

    Args:
        image_data: Dictionary containing image data with 'base64' and optionally 'mime_type'
        attachment_id: ID of the attachment for logging purposes

    Returns:
        Data URL string if valid, None otherwise
    """
    try:
        # Validate image data structure
        if not isinstance(image_data, dict):
            logger.error(
                f"Invalid image data structure for {attachment_id}: {type(image_data)}"
            )
            return None

        # Validate required fields
        if "base64" not in image_data:
            logger.error(f"Missing base64 data for image {attachment_id}")
            return None

        base64_data = image_data["base64"]
        if not isinstance(base64_data, str) or not base64_data:
            logger.error(f"Invalid base64 data for image {attachment_id}")
            return None

        # Validate base64 format
        try:
            base64.b64decode(base64_data)
        except Exception as e:
            logger.error(f"Invalid base64 format for image {attachment_id}: {str(e)}")
            return None

        # Get mime type with better fallback
        mime_type = image_data.get("mime_type", "image/jpeg")
        if (
            not mime_type
            or not isinstance(mime_type, str)
            or not mime_type.startswith("image/")
        ):
            logger.warning(
                f"Invalid mime type for image {attachment_id}: {mime_type}, defaulting to image/jpeg"
            )
            mime_type = "image/jpeg"

        # Create data URL
        return f"data:{mime_type};base64,{base64_data}"

    except Exception as e:
        logger.error(
            f"Failed to validate image {attachment_id}: {str(e)}",
            exc_info=True,
        )
        return None


def validate_and_build_document_data_url(doc_data: dict, attachment_id: str) -> Optional[str]:
    """
    Validate document data and build a data URL if valid.

    Args:
        doc_data: Dictionary containing document data with 'base64', 'mime_type', 'file_name'
        attachment_id: ID of the attachment for logging purposes

    Returns:
        Data URL string if valid, None otherwise
    """
    try:
        # Validate document data structure
        if not isinstance(doc_data, dict):
            logger.error(
                f"Invalid document data structure for {attachment_id}: {type(doc_data)}"
            )
            return None

        # Validate required fields
        if "base64" not in doc_data:
            logger.error(f"Missing base64 data for document {attachment_id}")
            return None

        base64_data = doc_data["base64"]
        if not isinstance(base64_data, str) or not base64_data:
            logger.error(f"Invalid base64 data for document {attachment_id}")
            return None

        # Validate base64 format
        try:
            base64.b64decode(base64_data)
        except Exception as e:
            logger.error(f"Invalid base64 format for document {attachment_id}: {str(e)}")
            return None

        # Get mime type - required for documents
        mime_type = doc_data.get("mime_type")
        if not mime_type or not isinstance(mime_type, str):
            logger.warning(
                f"Missing or invalid mime type for document {attachment_id}"
            )
            mime_type = "application/octet-stream"

        # Create data URL
        return f"data:{mime_type};base64,{base64_data}"

    except Exception as e:
        logger.error(
            f"Failed to validate document {attachment_id}: {str(e)}",
            exc_info=True,
        )
        return None


def document_to_user_content(doc_data: dict, attachment_id: str) -> Optional[UserContent]:
    """Build a DocumentUrl for LLM file input, or a text note if the type is unsupported.

    Do not use ImageUrl for documents: OpenAI rejects non-image data URLs in image_url parts.
    """
    data_url = validate_and_build_document_data_url(doc_data, attachment_id)
    if not data_url:
        return None
    mime = coerce_document_mime(doc_data.get("mime_type"), doc_data.get("file_name", ""))
    if mime:
        return DocumentUrl(
            url=data_url,
            media_type=mime,
            identifier=attachment_id,
        )
    file_name = doc_data.get("file_name", "document")
    declared = doc_data.get("mime_type") or "unknown"
    logger.warning(
        "Document %s (%s) is not a supported file-input MIME; sending text placeholder only",
        attachment_id,
        declared,
    )
    return (
        f'\n[Attached file "{file_name}" ({declared}) cannot be passed as a model file input.]\n'
    )


def create_multimodal_user_content(ctx: ChatContext) -> Sequence[UserContent]:
    """Create multimodal user content with images and documents using PydanticAI's ImageUrl"""
    content: List[UserContent] = [ctx.query]

    # Add current images to the content
    current_images = ctx.get_current_images_only()

    for attachment_id, image_data in current_images.items():
        data_url = validate_and_build_data_url(image_data, attachment_id)
        if data_url:
            content.append(ImageUrl(url=data_url))

    # If no current images, add context images as fallback
    if not current_images:
        context_images = ctx.get_context_images_only()

        for attachment_id, image_data in context_images.items():
            data_url = validate_and_build_data_url(image_data, attachment_id)
            if data_url:
                content.append(ImageUrl(url=data_url))

    # Add document attachments to the content
    current_documents = ctx.get_current_documents_only()

    for attachment_id, doc_data in current_documents.items():
        part = document_to_user_content(doc_data, attachment_id)
        if part:
            content.append(part)

    return content
