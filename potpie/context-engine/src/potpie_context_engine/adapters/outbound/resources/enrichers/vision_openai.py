"""Vision captions via httpx → OpenAI API (opt-in cloud)."""

from __future__ import annotations

import base64
import os
from pathlib import Path

DEFAULT_OPENAI_VISION_MODEL = "gpt-4o-mini"


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"


def caption_image_openai(
    path: Path,
    *,
    model: str | None = None,
    prompt: str = "Describe this image in one or two sentences for a search index.",
    api_key: str | None = None,
    timeout: float = 120.0,
) -> str:
    try:
        import httpx
    except ImportError:
        return ""

    key = api_key or os.getenv("OPENAI_API_KEY") or ""
    if not key:
        return ""

    model_name = model or os.getenv("POTPIE_OPENAI_VISION_MODEL") or DEFAULT_OPENAI_VISION_MODEL
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = _mime_type(path)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except Exception:
        return ""

    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


__all__ = ["caption_image_openai", "DEFAULT_OPENAI_VISION_MODEL"]
