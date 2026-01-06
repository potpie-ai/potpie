"""Multimodal utility functions for multi-agent system"""

import base64
from typing import List, Optional, Sequence
from pydantic_ai.messages import UserContent, ImageUrl

from app.modules.intelligence.agents.chat_agent import ChatContext
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


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


def create_multimodal_user_content(ctx: ChatContext) -> Sequence[UserContent]:
    """Create multimodal user content with images using PydanticAI's ImageUrl"""
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

    return content
