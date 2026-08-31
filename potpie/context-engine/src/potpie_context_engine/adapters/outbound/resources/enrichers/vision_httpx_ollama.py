"""Vision captions via httpx → Ollama (no ollama pip package)."""

from __future__ import annotations

import base64
import os
from pathlib import Path

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_VISION_MODEL = "moondream"


def _image_bytes(path: Path) -> bytes:
    return path.read_bytes()


def caption_image_local(
    path: Path,
    *,
    model: str | None = None,
    prompt: str = "Describe this image in one or two sentences for a search index.",
    base_url: str | None = None,
    timeout: float = 120.0,
) -> str:
    try:
        import httpx
    except ImportError:
        return ""

    host = (base_url or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    model_name = model or os.getenv("POTPIE_VISION_MODEL") or DEFAULT_VISION_MODEL
    encoded = base64.b64encode(_image_bytes(path)).decode("ascii")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [encoded],
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{host}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return ""

    text = str(data.get("response") or "").strip()
    return text


def ollama_reachable(base_url: str | None = None, timeout: float = 3.0) -> bool:
    try:
        import httpx
    except ImportError:
        return False
    host = (base_url or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{host}/api/tags")
            return response.status_code == 200
    except Exception:
        return False


__all__ = [
    "caption_image_local",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_VISION_MODEL",
    "ollama_reachable",
]
