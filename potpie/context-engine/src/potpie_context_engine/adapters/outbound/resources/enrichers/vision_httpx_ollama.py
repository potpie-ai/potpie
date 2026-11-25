"""Vision captions via httpx → Ollama (no ollama pip package)."""

from __future__ import annotations

import base64
import json
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
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, list) and len(parsed) == 4 and all(
            isinstance(value, (int, float)) for value in parsed
        ):
            return ""
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


def ollama_model_available(
    model: str, base_url: str | None = None, timeout: float = 3.0
) -> bool:
    """Check whether the local Ollama server has a requested model tag."""
    try:
        import httpx
    except ImportError:
        return False
    host = (base_url or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{host}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
    except Exception:
        return False
    return any(
        isinstance(entry, dict)
        and (
            str(entry.get("name", "")) == model
            if ":" in model
            else str(entry.get("name", "")).split(":", 1)[0] == model
        )
        for entry in models
    )


def ollama_model_supports_vision(
    model: str, base_url: str | None = None, timeout: float = 3.0
) -> bool:
    """Check Ollama's declared capabilities before selecting a vision model."""
    try:
        import httpx
    except ImportError:
        return False
    host = (base_url or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{host}/api/show", json={"model": model})
            response.raise_for_status()
            capabilities = response.json().get("capabilities", [])
    except Exception:
        return False
    return isinstance(capabilities, list) and "vision" in capabilities


__all__ = [
    "caption_image_local",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_VISION_MODEL",
    "ollama_reachable",
    "ollama_model_available",
    "ollama_model_supports_vision",
]
