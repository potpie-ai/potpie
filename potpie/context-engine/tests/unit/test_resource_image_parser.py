"""Unit tests for image parser (mocked OCR/vision)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from potpie_context_engine.adapters.outbound.resources.parsers.image import (
    parse_image_to_staging,
)

pytestmark = pytest.mark.unit


def test_parse_image_to_staging_writes_ocr_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "slide.png"
    source.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    out = tmp_path / "staging"

    with (
        patch(
            "potpie_context_engine.adapters.outbound.resources.parsers.image.ocr_image_text",
            return_value="Hello from OCR",
        ),
        patch(
            "potpie_context_engine.adapters.outbound.resources.parsers.image.caption_image_local",
            return_value="A diagram of the system.",
        ),
    ):
        manifest = parse_image_to_staging(source, out, vision_provider="local")

    assert manifest.source_kind == "image"
    assert manifest.sections[0].slug == "visual"
    chunk_txt = out / "visual" / "0000.txt"
    ocr_txt = out / "visual" / "0000.ocr.txt"
    assert chunk_txt.is_file()
    assert ocr_txt.is_file()
    assert "diagram" in chunk_txt.read_text(encoding="utf-8").lower()
    assert ocr_txt.read_text(encoding="utf-8") == "Hello from OCR"
    assert (out / "artifacts" / "source.png").is_file()
