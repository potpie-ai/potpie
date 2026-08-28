"""RapidOCR text extraction for images ([documents] extra)."""

from __future__ import annotations

from pathlib import Path


def ocr_image_text(path: Path) -> str:
    """Return OCR text for an image path, or empty string if OCR is unavailable."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return ""

    engine = RapidOCR()
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
