"""Unit tests for provenance models and element chunker."""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.resources.parsers.pdf_docling_provenance import (
    _chunk_elements,
    _LedgerElement,
)
from potpie_context_engine.domain.resource_models import (
    ChunkProvenanceRecord,
    DocumentElementRecord,
)

pytestmark = pytest.mark.unit


def test_document_element_bbox_validation() -> None:
    with pytest.raises(ValueError):
        DocumentElementRecord(
            element_id="elem-1",
            element_type="text",
            bbox=[0.0, 0.0, 1.0],
        )


def test_chunk_elements_multi_element_provenance() -> None:
    first = DocumentElementRecord(
        element_id="elem-00001",
        element_type="paragraph",
        text="alpha",
        page_number=1,
        bbox=[0.0, 0.0, 1.0, 1.0],
        text_hash="sha256:abc",
    )
    second = DocumentElementRecord(
        element_id="elem-00002",
        element_type="paragraph",
        text="beta",
        page_number=2,
        bbox=[0.1, 0.1, 0.9, 0.9],
        text_hash="sha256:def",
    )
    staged = _chunk_elements(
        [
            _LedgerElement(record=first, searchable_text="alpha"),
            _LedgerElement(record=second, searchable_text="beta"),
        ],
        chunk_target=4000,
    )
    assert len(staged) == 1
    assert "alpha" in staged[0].text and "beta" in staged[0].text
    assert len(staged[0].provenance) == 2
    element_ids = {row.element_id for row in staged[0].provenance}
    assert element_ids == {"elem-00001", "elem-00002"}


def test_chunk_elements_splits_across_pages() -> None:
    long_a = "word " * 500
    long_b = "term " * 500
    first = DocumentElementRecord(element_id="elem-a", element_type="text", text=long_a)
    second = DocumentElementRecord(
        element_id="elem-b",
        element_type="text",
        text=long_b,
        page_number=8,
    )
    staged = _chunk_elements(
        [
            _LedgerElement(record=first, searchable_text=long_a.strip()),
            _LedgerElement(record=second, searchable_text=long_b.strip()),
        ],
        chunk_target=400,
    )
    assert len(staged) >= 2
    pages = {row.page_number for chunk in staged for row in chunk.provenance}
    assert 8 in pages


def test_chunk_provenance_record_charspan() -> None:
    row = ChunkProvenanceRecord(
        element_id="elem-1",
        char_start=0,
        char_end=12,
    )
    assert row.char_end == 12
