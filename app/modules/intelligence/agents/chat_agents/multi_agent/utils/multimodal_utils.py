"""Multimodal utility functions for multi-agent system"""

from typing import List, Sequence
from pydantic_ai.messages import UserContent, ImageUrl

from app.modules.intelligence.agents.chat_agent import ChatContext
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_multimodal_user_content(ctx: ChatContext) -> Sequence[UserContent]:
    """Create multimodal user content with images using PydanticAI's ImageUrl"""
    content: List[UserContent] = [ctx.query]

    # Add current images to the content
    current_images = ctx.get_current_images_only()

    for attachment_id, image_data in current_images.items():
        try:
            # Validate image data structure
            if not isinstance(image_data, dict):
                logger.error(
                    f"Invalid image data structure for {attachment_id}: {type(image_data)}"
                )
                continue

            # Validate required fields
            if "base64" not in image_data:
                logger.error(f"Missing base64 data for image {attachment_id}")
                continue

            base64_data = image_data["base64"]
            if not isinstance(base64_data, str) or not base64_data:
                logger.error(f"Invalid base64 data for image {attachment_id}")
                continue

            # Validate base64 format
            try:
                import base64

                base64.b64decode(base64_data)
            except Exception as e:
                logger.error(
                    f"Invalid base64 format for image {attachment_id}: {str(e)}"
                )
                continue

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
            data_url = f"data:{mime_type};base64,{base64_data}"
            content.append(ImageUrl(url=data_url))

        except Exception as e:
            logger.error(
                f"Failed to add image {attachment_id} to content: {str(e)}",
                exc_info=True,
            )
            continue

    # If no current images, add context images as fallback
    if not current_images:
        context_images = ctx.get_context_images_only()

        for attachment_id, image_data in context_images.items():
            try:
                # Apply same validation as above
                if not isinstance(image_data, dict) or "base64" not in image_data:
                    logger.error(f"Invalid context image data for {attachment_id}")
                    continue

                base64_data = image_data["base64"]
                if not isinstance(base64_data, str) or not base64_data:
                    logger.error(
                        f"Invalid base64 data for context image {attachment_id}"
                    )
                    continue

                # Validate base64 format
                try:
                    import base64

                    base64.b64decode(base64_data)
                except Exception as e:
                    logger.error(
                        f"Invalid base64 format for context image {attachment_id}: {str(e)}"
                    )
                    continue

                mime_type = image_data.get("mime_type", "image/jpeg")
                if (
                    not mime_type
                    or not isinstance(mime_type, str)
                    or not mime_type.startswith("image/")
                ):
                    mime_type = "image/jpeg"

                data_url = f"data:{mime_type};base64,{base64_data}"
                content.append(ImageUrl(url=data_url))
            except Exception as e:
                logger.error(
                    f"Failed to add context image {attachment_id} to content: {str(e)}",
                    exc_info=True,
                )
                continue

    return content
