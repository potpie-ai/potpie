from __future__ import annotations

import logging
from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.parsers import docling_convert


pytestmark = pytest.mark.unit


def test_structured_office_fallback_is_marked_and_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "guide.docx"
    source.write_bytes(b"fake-docx")

    class _Document:
        def export_to_markdown(self) -> str:
            return "## Setup\n\nInstall the package."

    class _Converter:
        def convert(self, path: str) -> object:
            del path
            return type("Result", (), {"document": _Document()})()

    monkeypatch.setattr(
        docling_convert, "_require_docling_converter", lambda: _Converter()
    )
    monkeypatch.setattr(
        docling_convert,
        "build_sections_from_docling",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("structured ledger unavailable")
        ),
    )

    with caplog.at_level(logging.WARNING):
        manifest, _, _, _, _ = docling_convert.parse_docling_file(source)

    assert manifest.parser_tier == "docling-markdown"
    assert manifest.provenance_version == 0
    assert "structured ledger unavailable" in caplog.text
    assert "falling back to Markdown" in caplog.text
