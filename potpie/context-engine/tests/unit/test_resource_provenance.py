"""Unit tests for provenance models and element chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.parsers.pdf_docling_provenance import (
    build_sections_from_docling,
    _chunk_elements,
    _LedgerElement,
    parse_pdf_docling_provenance,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_HARD_CAP,
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


class _FakeTextItem:
    def __init__(
        self,
        label: str,
        text: str,
        *,
        page_number: int | None = None,
    ) -> None:
        self.label = label
        self.text = text
        self.prov = (
            [
                type(
                    "Prov",
                    (),
                    {
                        "page_no": page_number,
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "charspan": [0, len(text)],
                    },
                )()
            ]
            if page_number is not None
            else []
        )


class _FakeTableItem(_FakeTextItem):
    def __init__(self, csv_text: str, *, page_number: int = 1) -> None:
        super().__init__("table", "", page_number=page_number)
        self._csv_text = csv_text

    def export_to_dataframe(self, doc: object) -> object:
        csv_text = self._csv_text

        class _Frame:
            def to_csv(self, path=None, index: bool = False) -> str | None:
                del index
                if path is not None:
                    Path(path).write_text(csv_text, encoding="utf-8")
                    return None
                return csv_text

        return _Frame()

    def export_to_markdown(self, doc: object) -> str:
        del doc
        return self._csv_text


class _FakePictureItem(_FakeTextItem):
    def get_image(self, document: object) -> object:
        del document

        class _Image:
            def save(self, file_object: object, format: str) -> None:
                del format
                file_object.write(b"fake-png")

        return _Image()


class _FakeDocument:
    def __init__(self, items: list[object]) -> None:
        self._items = items

    def iterate_items(self):
        return [(item, 0) for item in self._items]


def _patch_docling_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "potpie_context_engine.adapters.outbound.resources.parsers."
        "pdf_docling_provenance._require_docling_types",
        lambda: (
            object,
            object,
            object,
            object,
            object,
            object,
            object,
            _FakeTextItem,
            _FakeTableItem,
            _FakePictureItem,
        ),
    )


def test_docling_preserves_footnotes_and_only_drops_repeated_page_furniture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_docling_types(monkeypatch)
    document = _FakeDocument(
        [
            _FakeTextItem("page_header", "Quarterly report", page_number=1),
            _FakeTextItem(
                "text", "Revenue\u00a0 rose   12%\r\nthis quarter.", page_number=1
            ),
            _FakeTextItem(
                "footnote", "1. Includes audited adjustments.", page_number=1
            ),
            _FakeTextItem("page_footer", "Acme confidential", page_number=1),
            _FakeTextItem("page_header", "Quarterly report", page_number=2),
            _FakeTextItem("text", "Operating income improved.", page_number=2),
            _FakeTextItem("page_footer", "Acme confidential", page_number=2),
            _FakeTextItem("page_header", "Appendix A", page_number=3),
            _FakeTextItem("text", "Definitions follow.", page_number=3),
        ]
    )

    sections, elements, _, _ = build_sections_from_docling(document, chunk_target=4000)
    body = "\n".join(chunk for section in sections for chunk in section.chunks)
    element_texts = {element.text for element in elements}

    assert "Quarterly report" not in body
    assert "Acme confidential" not in body
    assert "Appendix A" in body
    assert "1. Includes audited adjustments." in body
    assert "Revenue rose 12%\nthis quarter." in body
    assert "1. Includes audited adjustments." in element_texts


def test_repeated_docling_headers_keep_section_and_provenance_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_docling_types(monkeypatch)
    document = _FakeDocument(
        [
            _FakeTextItem("section_header", "Context", page_number=1),
            _FakeTextItem("text", "first context passage", page_number=1),
            _FakeTextItem("section_header", "Context", page_number=2),
            _FakeTextItem("text", "second context passage", page_number=2),
        ]
    )

    sections, _, provenance, _ = build_sections_from_docling(
        document, chunk_target=4000
    )

    assert [section.slug for section in sections] == ["context", "context-2"]
    assert sections[0].chunks == ["first context passage"]
    assert sections[1].chunks == ["second context passage"]
    assert [
        row.page_number for row in provenance[("context", 0)]
    ] == [1]
    assert [
        row.page_number for row in provenance[("context-2", 0)]
    ] == [2]


def test_docling_picture_search_text_includes_ocr_and_visual_caption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_docling_types(monkeypatch)
    monkeypatch.setattr(
        "potpie_context_engine.adapters.outbound.resources.parsers."
        "pdf_docling_provenance._try_ocr_picture",
        lambda path: "1542M 63.24",
    )
    monkeypatch.setattr(
        "potpie_context_engine.adapters.outbound.resources.parsers."
        "pdf_docling_provenance._try_vision_caption",
        lambda path, provider: "The larger model line is above the smaller model line.",
    )

    picture = _FakePictureItem("picture", "Figure 4. Model capacity", page_number=9)
    sections, elements, _, _ = build_sections_from_docling(
        _FakeDocument([picture]),
        chunk_target=4000,
        artifacts_dir=tmp_path / "artifacts",
        vision_provider="local",
    )

    body = "\n".join(chunk for section in sections for chunk in section.chunks)
    element = next(row for row in elements if row.element_type == "picture")
    assert "Figure 1" in body
    assert "1542M 63.24" in body
    assert "larger model line" in body
    assert element.figure_number == 1
    assert element.visual_caption.startswith("The larger model")
    assert element.ocr_text == "1542M 63.24"


def test_docling_manifest_reports_picture_enrichment_observations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_docling_types(monkeypatch)
    monkeypatch.setattr(
        "potpie_context_engine.adapters.outbound.resources.parsers."
        "pdf_docling_provenance._build_converter",
        lambda: type(
            "Converter",
            (),
            {"convert": lambda self, source: type(
                "Result", (), {"document": _FakeDocument([_FakePictureItem(
                    "picture", "Figure 4", page_number=9
                )])}
            )()},
        )(),
    )
    monkeypatch.setattr(
        "potpie_context_engine.adapters.outbound.resources.parsers."
        "pdf_docling_provenance._try_ocr_picture",
        lambda path: "1542M 63.24",
    )
    monkeypatch.setattr(
        "potpie_context_engine.adapters.outbound.resources.parsers."
        "pdf_docling_provenance._try_vision_caption",
        lambda path, provider: "A larger model is shown.",
    )
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"pdf fixture")

    manifest, *_ = parse_pdf_docling_provenance(
        source,
        staging_root=tmp_path / "staging",
        vision_provider="local",
    )

    observations = manifest.ingestion_diagnostics["picture_enrichment"]
    assert observations["pictures_total"] == 1
    assert observations["pictures_with_ocr"] == 1
    assert observations["pictures_with_visual_caption"] == 1
    assert observations["vision_provider"] == "local"
    assert observations["warnings"] == []


def test_pdf_vision_caption_skips_unreachable_ollama(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from potpie_context_engine.adapters.outbound.resources.parsers import (
        pdf_docling_provenance,
    )

    called = False

    def fail_if_called(path: Path) -> str:
        nonlocal called
        del path
        called = True
        return "unexpected caption"

    monkeypatch.setattr(
        "potpie_context_engine.adapters.outbound.resources.enrichers."
        "vision_httpx_ollama.ollama_reachable",
        lambda: False,
    )
    monkeypatch.setattr(
        "potpie_context_engine.adapters.outbound.resources.enrichers."
        "vision_httpx_ollama.caption_image_local",
        fail_if_called,
    )

    assert pdf_docling_provenance._try_vision_caption(tmp_path / "figure.png", "local") == ""
    assert called is False


def test_docling_cleanup_preserves_code_block_horizontal_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_docling_types(monkeypatch)
    document = _FakeDocument(
        [
            _FakeTextItem(
                "code",
                "total =  12\r\n\treturn total",
                page_number=1,
            )
        ]
    )

    sections, _, _, _ = build_sections_from_docling(document, chunk_target=4000)

    assert sections[0].chunks == ["total =  12\n\treturn total"]


def test_table_rows_repeat_column_meaning_across_target_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_docling_types(monkeypatch)
    rows = [f"region-{idx},value-{idx}" for idx in range(12)]
    table = _FakeTableItem("Region,Revenue\n" + "\n".join(rows) + "\n", page_number=4)

    sections, elements, provenance, _ = build_sections_from_docling(
        _FakeDocument([table]),
        chunk_target=120,
        artifacts_dir=tmp_path / "artifacts",
    )
    chunks = [chunk for section in sections for chunk in section.chunks]

    assert len(chunks) > 1
    assert all("Region" in chunk and "Revenue" in chunk for chunk in chunks)
    assert all(len(chunk) <= CHUNK_HARD_CAP for chunk in chunks)
    for idx in range(12):
        assert f"region-{idx}" in "\n".join(chunks)
        assert f"value-{idx}" in "\n".join(chunks)
    table_element = next(
        element for element in elements if element.element_type == "table"
    )
    assert table_element.artifact_ref == "artifacts/table-0001.csv"
    assert any(
        row.element_id == table_element.element_id and row.page_number == 4
        for rows in provenance.values()
        for row in rows
    )


def test_oversized_table_row_stays_under_hard_cap_with_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_docling_types(monkeypatch)
    terminal_marker = "END-OF-OVERSIZED-CELL"
    table = _FakeTableItem(
        "Key,Payload\nalpha," + ("x" * (CHUNK_HARD_CAP + 500)) + terminal_marker + "\n"
    )

    sections, _, _, _ = build_sections_from_docling(
        _FakeDocument([table]), chunk_target=CHUNK_HARD_CAP * 2
    )
    chunks = [chunk for section in sections for chunk in section.chunks]

    assert len(chunks) > 1
    assert all(len(chunk) <= CHUNK_HARD_CAP for chunk in chunks)
    assert all("Key" in chunk and "Payload" in chunk for chunk in chunks)
    assert all("Payload=" in chunk for chunk in chunks[1:])
    assert terminal_marker in chunks[-1]


def test_docling_preserves_all_provenance_entries_for_one_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_docling_types(monkeypatch)
    item = _FakeTextItem("text", "spans two pages", page_number=3)
    item.prov.append(
        type(
            "Prov",
            (),
            {
                "page_no": 4,
                "bbox": [0.0, 0.0, 2.0, 2.0],
                "charspan": [5, len(item.text)],
            },
        )()
    )

    _, _, provenance, _ = build_sections_from_docling(
        _FakeDocument([item]), chunk_target=4000
    )

    rows = next(iter(provenance.values()))
    assert [row.page_number for row in rows] == [3, 4]
    assert [row.bbox for row in rows] == [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0, 2.0],
    ]


def test_chunk_refit_does_not_invent_neighbor_page_citations() -> None:
    ledgers = []
    for page in range(1, 7):
        text = f"PAGE-{page}-" + (str(page) * 3890)
        record = DocumentElementRecord(
            element_id=f"elem-{page}",
            element_type="text",
            text=text,
            page_number=page,
        )
        ledgers.append(_LedgerElement(record=record, searchable_text=text))

    chunks = _chunk_elements(ledgers, chunk_target=4000)

    assert len(chunks) == 6
    for index, chunk in enumerate(chunks):
        expected_pages = {index + 1}
        assert {row.page_number for row in chunk.provenance} == expected_pages
        assert all(
            (f"PAGE-{page}-" in chunk.text) == (page in expected_pages)
            for page in range(1, 7)
        )


def test_csv_table_cells_with_pipes_keep_csv_column_association(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_docling_types(monkeypatch)
    table = _FakeTableItem("Metric|unit,Value\nCost|USD,99\n")

    sections, _, _, _ = build_sections_from_docling(
        _FakeDocument([table]), chunk_target=4000, artifacts_dir=tmp_path
    )
    body = "\n".join(chunk for section in sections for chunk in section.chunks)

    assert "Metric|unit=Cost|USD" in body
    assert "Value=99" in body


def test_markdown_table_escaped_pipes_keep_header_association(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_docling_types(monkeypatch)

    class _MarkdownTable(_FakeTableItem):
        def export_to_dataframe(self, doc: object) -> object:
            del doc
            raise RuntimeError("dataframe unavailable")

        def export_to_markdown(self, doc: object) -> str:
            del doc
            return "| Metric\\|unit | Value |\n| --- | --- |\n| Cost\\|USD | 99 |"

    sections, _, _, _ = build_sections_from_docling(
        _FakeDocument([_MarkdownTable("")]), chunk_target=4000
    )
    body = "\n".join(chunk for section in sections for chunk in section.chunks)

    assert "Metric|unit=Cost|USD" in body
    assert "Value=99" in body


def test_oversized_table_header_uses_bounded_column_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_docling_types(monkeypatch)
    long_header = "H" * (CHUNK_HARD_CAP + 50)
    table = _FakeTableItem(f"{long_header},Value\nrow-key,terminal-value\n")

    sections, _, _, _ = build_sections_from_docling(
        _FakeDocument([table]), chunk_target=CHUNK_HARD_CAP * 2, artifacts_dir=tmp_path
    )
    chunks = [chunk for section in sections for chunk in section.chunks]

    assert all(len(chunk) <= CHUNK_HARD_CAP for chunk in chunks)
    data_chunk = next(chunk for chunk in chunks if "terminal-value" in chunk)
    assert "column-1=row-key" in data_chunk
    assert "Value=terminal-value" in data_chunk
