"""Unit tests for document-capability detection."""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.resources import capabilities


pytestmark = pytest.mark.unit


def test_rapidocr_capability_detects_modern_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """docling[rapidocr] installs `rapidocr`, not `rapidocr_onnxruntime` — the
    probe must accept either, or doctor reports OCR missing on working hosts."""

    def fake_has_module(name: str) -> bool:
        return name == "rapidocr"

    monkeypatch.setattr(capabilities, "_has_module", fake_has_module)
    assert capabilities.documents_capabilities()["rapidocr"] is True
