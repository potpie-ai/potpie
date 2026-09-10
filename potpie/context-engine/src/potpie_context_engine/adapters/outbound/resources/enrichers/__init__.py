"""Optional enrichers for resource parsing ([documents] extra)."""

from potpie_context_engine.adapters.outbound.resources.enrichers.ocr_rapidocr import (
    ocr_image_text,
)
from potpie_context_engine.adapters.outbound.resources.enrichers.vision_httpx_ollama import (
    caption_image_local,
)
from potpie_context_engine.adapters.outbound.resources.enrichers.vision_openai import (
    caption_image_openai,
)

__all__ = [
    "caption_image_local",
    "caption_image_openai",
    "ocr_image_text",
]
