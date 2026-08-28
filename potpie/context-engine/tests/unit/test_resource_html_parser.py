"""Unit tests for HTML parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.parsers.html import (
    html_to_markdownish,
    parse_html_to_staging,
)

pytestmark = pytest.mark.unit


def test_html_to_markdownish_headings() -> None:
    raw = "<html><body><h2>Setup</h2><p>Install dependencies.</p></body></html>"
    text = html_to_markdownish(raw)
    assert "Setup" in text
    assert "Install dependencies" in text


def test_parse_html_to_staging(tmp_path: Path) -> None:
    source = tmp_path / "page.html"
    source.write_text(
        "<html><body><h2>Intro</h2><p>First paragraph.</p></body></html>",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    manifest = parse_html_to_staging(source, out)
    assert manifest.source_kind == "html"
    assert (out / "meta.json").is_file()
    assert manifest.sections
