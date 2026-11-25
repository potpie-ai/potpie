"""Unit tests for the RapidOCR enricher import paths (fake engine modules)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.enrichers.ocr_rapidocr import (
    ocr_image_text,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_engine_caches() -> None:
    from potpie_context_engine.adapters.outbound.resources.enrichers import (
        ocr_rapidocr,
    )

    ocr_rapidocr._modern_engine.cache_clear()
    ocr_rapidocr._legacy_engine.cache_clear()
    yield
    ocr_rapidocr._modern_engine.cache_clear()
    ocr_rapidocr._legacy_engine.cache_clear()


def test_ocr_image_text_uses_modern_rapidocr_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`[documents]` installs `rapidocr` (v3, RapidOCROutput.txts), not the
    legacy `rapidocr_onnxruntime` — OCR must work through the modern API."""

    class FakeOutput:
        txts = ("Hello", "World")

    class FakeOCR:
        def __call__(self, path: str) -> FakeOutput:
            return FakeOutput()

    fake = types.ModuleType("rapidocr")
    fake.RapidOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)

    image = tmp_path / "slide.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert ocr_image_text(image) == "Hello\nWorld"


def test_ocr_image_text_falls_back_to_legacy_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeLegacyOCR:
        def __call__(self, path: str) -> tuple[list[list[object]], object]:
            return [[[0, 0, 1, 1], "Legacy line", 0.9]], None

    fake = types.ModuleType("rapidocr_onnxruntime")
    fake.RapidOCR = FakeLegacyOCR
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake)

    image = tmp_path / "slide.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert ocr_image_text(image) == "Legacy line"


def test_ocr_image_text_returns_empty_when_no_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)

    image = tmp_path / "slide.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert ocr_image_text(image) == ""


def test_ocr_image_text_falls_back_to_torch_engine_when_onnxruntime_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """docling[rapidocr] installs the torch engine, not onnxruntime — the
    default RapidOCR() init raises ImportError and must not crash ingest."""

    class EngineType:
        TORCH = "torch"

    class FakeOutput:
        txts = ("Torch line",)

    class FakeOCR:
        def __init__(self, params: dict | None = None) -> None:
            if not params:
                raise ImportError("onnxruntime is not installed.")
            assert params.get("Det.engine_type") == EngineType.TORCH

        def __call__(self, path: str) -> FakeOutput:
            return FakeOutput()

    fake = types.ModuleType("rapidocr")
    fake.RapidOCR = FakeOCR
    fake_utils = types.ModuleType("rapidocr.utils")
    fake_typings = types.ModuleType("rapidocr.utils.typings")
    fake_typings.EngineType = EngineType
    monkeypatch.setitem(sys.modules, "rapidocr", fake)
    monkeypatch.setitem(sys.modules, "rapidocr.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", fake_typings)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)

    image = tmp_path / "slide.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert ocr_image_text(image) == "Torch line"


def test_ocr_engine_is_constructed_once_across_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PDF ingest OCRs one figure at a time; the engine (a multi-second model
    load) must be reused across calls, not rebuilt per image."""
    constructions = []

    class FakeOutput:
        txts = ("line",)

    class FakeOCR:
        def __init__(self, params: dict | None = None) -> None:
            constructions.append(1)

        def __call__(self, path: str) -> FakeOutput:
            return FakeOutput()

    fake = types.ModuleType("rapidocr")
    fake.RapidOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)

    image = tmp_path / "slide.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert ocr_image_text(image) == "line"
    assert ocr_image_text(image) == "line"
    assert sum(constructions) == 1
