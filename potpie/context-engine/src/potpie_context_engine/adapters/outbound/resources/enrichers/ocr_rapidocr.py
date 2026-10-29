"""RapidOCR text extraction for images ([documents] extra)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def ocr_image_text(path: Path) -> str:
    """Return OCR text for an image path, or empty string if OCR is unavailable.

    The `[documents]` extra ships the modern `rapidocr` package (via
    docling[rapidocr]); the legacy `rapidocr_onnxruntime` API is kept as a
    fallback for environments that installed it directly. Engines are cached
    per process — construction loads three models and PDF ingest calls this
    once per figure.
    """
    engine = _modern_engine()
    if engine is None:
        return _ocr_image_text_legacy(path)
    try:
        result = engine(str(path))
        txts = getattr(result, "txts", None) or ()
        return "\n".join(str(t).strip() for t in txts if t).strip()
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _modern_engine() -> Any | None:
    """Resolve a modern `rapidocr` engine once, or None if unusable."""
    try:
        from rapidocr import RapidOCR
    except ImportError:
        return None

    try:
        return RapidOCR()
    except Exception:
        # The default config wants onnxruntime; docling[rapidocr] ships the
        # torch engine instead, so retry with it before giving up.
        pass
    try:
        from rapidocr.utils.typings import EngineType

        return RapidOCR(
            params={
                "Det.engine_type": EngineType.TORCH,
                "Cls.engine_type": EngineType.TORCH,
                "Rec.engine_type": EngineType.TORCH,
            }
        )
    except Exception:
        return None


@lru_cache(maxsize=1)
def _legacy_engine() -> Any | None:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    try:
        return RapidOCR()
    except Exception:
        return None


def _ocr_image_text_legacy(path: Path) -> str:
    engine = _legacy_engine()
    if engine is None:
        return ""
    try:
        result, _ = engine(str(path))
    except Exception:
        return ""
    if not result:
        return ""
    lines: list[str] = []
    for row in result:
        if len(row) >= 2 and row[1]:
            lines.append(str(row[1]).strip())
    return "\n".join(lines).strip()


__all__ = ["ocr_image_text"]
