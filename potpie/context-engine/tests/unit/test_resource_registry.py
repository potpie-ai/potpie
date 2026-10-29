"""Unit tests for resource import + FTS registry."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    LocalResourceStore,
)
from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    parse_file_to_staging,
)
from potpie_context_engine.application.services.resource_service import (
    ResourceService,
)
from potpie_context_engine.domain.resource_models import (
    ChunkProvenanceRecord,
    ChunkProvenanceSidecar,
    DocumentElementRecord,
    ResourceManifest,
    SectionManifest,
)
from potpie_context_engine.core.ports.resource_store import Chunk

pytestmark = pytest.mark.unit


def test_import_and_search_chunks(tmp_path: Path) -> None:
    home = tmp_path / "potpie-home"
    store = LocalResourceStore(home=home)
    service = ResourceService(store=store)
    pot_id = "pot_test"
    source = tmp_path / "guide.md"
    source.write_text(
        "## Payments rollback\n\nRollback steps for payments deploy failures.",
        encoding="utf-8",
    )
    staging = tmp_path / "staging"
    manifest = parse_file_to_staging(source, staging)
    report = service.import_staging(
        pot_id=pot_id,
        doc_slug="payments-guide",
        staging_dir=staging,
    )
    assert not report.errors
    hits = service.search_chunks(pot_id=pot_id, query="rollback", limit=5)
    assert hits
    chunk = store.get_chunk_text(
        pot_id=pot_id,
        doc_slug="payments-guide",
        section_slug=manifest.sections[0].slug,
        seq=0,
    )
    assert "rollback" in chunk["content"].lower()


def test_chunk_payload_exposes_additive_evidence_text() -> None:
    chunk = Chunk(
        uri="potpie://res/paper/title/0000",
        pot_id="pot-a",
        doc_slug="paper",
        section_slug="title",
        seq=0,
        content="Alec Radford and colleagues present a language model.",
        section_title="Language Models are Unsupervised Multitask Learners",
    )

    payload = chunk.to_payload()

    assert payload["content"] == "Alec Radford and colleagues present a language model."
    assert payload["evidence_text"].startswith(
        "Language Models are Unsupervised Multitask Learners"
    )
    assert "Alec Radford" in payload["evidence_text"]

    payload_without_pages = Chunk(
        uri="potpie://res/paper/body/0000",
        pot_id="pot-a",
        doc_slug="paper",
        section_slug="body",
        seq=0,
        content="Evidence without a page mapping.",
    ).to_payload()
    assert payload_without_pages["citation"]["pages"] == []
    assert payload_without_pages["citation"]["pages_available"] is False


def test_failed_replacement_preserves_previous_payload_catalog_and_search(
    tmp_path: Path,
) -> None:
    home = tmp_path / "potpie-home"
    store = LocalResourceStore(home=home)
    service = ResourceService(store=store)
    pot_id = "pot_test"
    old_source = tmp_path / "old.md"
    old_source.write_text(
        "## Stable guide\n\nThe previous revision remains searchable.",
        encoding="utf-8",
    )
    old_stage = tmp_path / "old-stage"
    parse_file_to_staging(old_source, old_stage)
    old_report = service.import_staging(
        pot_id=pot_id,
        doc_slug="stable-guide",
        staging_dir=old_stage,
        source_ref="file:///old.md",
    )
    assert not old_report.errors

    broken_source = tmp_path / "broken.md"
    broken_source.write_text(
        "## Stable guide\n\nThe replacement is malformed.",
        encoding="utf-8",
    )
    broken_stage = tmp_path / "broken-stage"
    parse_file_to_staging(broken_source, broken_stage)
    section_slug = ResourceManifest.model_validate(
        json.loads((broken_stage / "meta.json").read_text(encoding="utf-8"))
    ).sections[0].slug
    (broken_stage / section_slug / "0000.txt").unlink()

    failed = service.import_staging(
        pot_id=pot_id,
        doc_slug="stable-guide",
        staging_dir=broken_stage,
        source_ref="file:///broken.md",
        force=True,
    )

    assert failed.errors
    old_chunk = service.get_chunk(
        pot_id=pot_id,
        uri="potpie://res/stable-guide/stable-guide/0000",
    )
    assert "previous revision remains searchable" in old_chunk["content"]
    assert old_chunk["source_ref"] == "file:///old.md"
    assert service.search_chunks(
        pot_id=pot_id, query="previous revision", limit=5
    )[0]["uri"] == old_chunk["uri"]
    assert service.search_chunks(pot_id=pot_id, query="replacement malformed") == []
    assert service.list_documents(pot_id=pot_id)[0]["source_ref"] == "file:///old.md"


def _import_doc(
    service: ResourceService, tmp_path: Path, pot_id: str, slug: str, body: str
) -> None:
    source = tmp_path / f"{slug}.md"
    source.write_text(body, encoding="utf-8")
    staging = tmp_path / f"staging-{slug}"
    parse_file_to_staging(source, staging)
    report = service.import_staging(pot_id=pot_id, doc_slug=slug, staging_dir=staging)
    assert not report.errors


def test_search_chunks_natural_language_query_matches_on_some_terms(
    tmp_path: Path,
) -> None:
    """A multi-word question must not require every token to appear in the chunk."""
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    pot_id = "pot_test"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "http-spec",
        "## Conditional requests\n\nThe If-Match header field makes a request "
        "conditional: the server evaluates the entity tag so a stale writer "
        "cannot clobber newer state.",
    )
    hits = service.search_chunks(
        pot_id=pot_id, query="how does If-Match prevent lost updates", limit=5
    )
    assert hits
    assert hits[0]["doc_slug"] == "http-spec"


def test_search_chunks_all_terms_hit_ranks_above_partial_hit(tmp_path: Path) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    pot_id = "pot_test"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "full-hit",
        "## Rollback\n\nRollback steps for payments deploy failures.",
    )
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "partial-hit",
        "## Deploys\n\nDeploy checklist for the payments service.",
    )
    hits = service.search_chunks(pot_id=pot_id, query="payments rollback", limit=5)
    assert [h["doc_slug"] for h in hits][0] == "full-hit"


def test_search_chunks_unions_strict_and_broad_query_forms(tmp_path: Path) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    pot_id = "pot_test"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "strict-hit",
        "## Exact evidence\n\nalpha beta gamma",
    )
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "broad-hit",
        "## Partial evidence\n\nalpha beta only",
    )

    hits = service.search_chunks(pot_id=pot_id, query="alpha beta gamma", limit=5)

    assert {hit["doc_slug"] for hit in hits} == {"strict-hit", "broad-hit"}
    assert hits[0]["doc_slug"] == "strict-hit"


def test_search_chunks_expands_common_wording_and_numeric_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_EMBEDDER", "none")
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    pot_id = "pot_test"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "paper",
        "## Results\n\nWe removed the encyclopedia. The larger model has 1542M parameters.",
    )

    wording_hits = service.search_chunks(
        pot_id=pot_id, query="omitted Wikipedia", limit=5
    )
    number_hits = service.search_chunks(
        pot_id=pot_id, query="1542 M", limit=5
    )

    assert wording_hits and wording_hits[0]["doc_slug"] == "paper"
    assert number_hits and number_hits[0]["doc_slug"] == "paper"


class _MarkerEmbedder:
    name = "test-marker"
    dimensions = 2

    def embed(self, text: str) -> tuple[float, float]:
        if "stale writes" in text.lower() or "if-match" in text.lower():
            return (1.0, 0.0)
        return (0.0, 1.0)

    def embed_many(self, texts: list[str]) -> list[tuple[float, float]]:
        return [self.embed(text) for text in texts]


def test_sqlite_index_returns_semantic_chunk_hits(tmp_path: Path) -> None:
    from potpie_context_engine.adapters.outbound.resources.fts_index import (
        SqliteFtsResourceIndex,
    )

    index = SqliteFtsResourceIndex(
        home=tmp_path / "index-home", embedder=_MarkerEmbedder()
    )
    chunks = [
        Chunk(
            uri="potpie://res/guide/body/0000",
            pot_id="pot-test",
            doc_slug="guide",
            section_slug="body",
            seq=0,
            content="If-Match blocks lost updates.",
        ),
        Chunk(
            uri="potpie://res/guide/body/0001",
            pot_id="pot-test",
            doc_slug="guide",
            section_slug="body",
            seq=1,
            content="This passage discusses weather.",
        ),
    ]
    assert index.index_document(
        pot_id="pot-test", doc_slug="guide", chunks=chunks
    ) == 2

    hits = index.search(
        pot_id="pot-test", query="How can a system prevent stale writes?", limit=2
    )

    assert hits[0].seq == 0
    assert hits[0].semantic_score == 1.0
    assert hits[0].retrieval_source in {"semantic", "hybrid"}


def test_sqlite_index_reports_active_semantic_provider(tmp_path: Path) -> None:
    from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
        HashingEmbedder,
    )
    from potpie_context_engine.adapters.outbound.resources.fts_index import (
        SqliteFtsResourceIndex,
    )

    index = SqliteFtsResourceIndex(
        home=tmp_path / "index-home", embedder=HashingEmbedder()
    )

    capabilities = index.capabilities()
    status = index.status()

    assert capabilities.semantic is True
    assert capabilities.semantic_provider == "hashing"
    assert capabilities.semantic_model is None
    assert status.detail == "semantic=hashing"

    lexical_index = SqliteFtsResourceIndex(home=tmp_path / "lexical-home")
    lexical_capabilities = lexical_index.capabilities()
    assert lexical_capabilities.semantic is False
    assert lexical_capabilities.semantic_provider is None
    assert lexical_index.status().detail == "semantic=disabled"


class _AdversarialEmbedder:
    name = "adversarial"
    dimensions = 2

    def embed(self, text: str) -> tuple[float, float]:
        if "target metric" in text.lower() or "semantic decoy" in text.lower():
            return (1.0, 0.0)
        return (0.0, 1.0)

    def embed_many(self, texts: list[str]) -> list[tuple[float, float]]:
        return [self.embed(text) for text in texts]


def test_exact_lexical_match_beats_semantic_only_tie(tmp_path: Path) -> None:
    from potpie_context_engine.adapters.outbound.resources.fts_index import (
        SqliteFtsResourceIndex,
    )

    index = SqliteFtsResourceIndex(
        home=tmp_path / "index-home", embedder=_AdversarialEmbedder()
    )
    chunks = [
        Chunk(
            uri="potpie://res/target/body/0000",
            pot_id="pot-test",
            doc_slug="target",
            section_slug="body",
            seq=0,
            content="The target metric is 6.31 percent.",
        ),
        Chunk(
            uri="potpie://res/decoy/body/0000",
            pot_id="pot-test",
            doc_slug="decoy",
            section_slug="body",
            seq=0,
            content="A semantic decoy discusses unrelated measurements.",
        ),
    ]
    index.index_document(pot_id="pot-test", doc_slug="target", chunks=chunks[:1])
    index.index_document(pot_id="pot-test", doc_slug="decoy", chunks=chunks[1:])

    hits = index.search(pot_id="pot-test", query="target metric", limit=2)

    assert hits[0].doc_slug == "target"
    assert hits[0].retrieval_source == "hybrid"


def test_multi_facet_query_unions_independent_evidence(tmp_path: Path) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    pot_id = "pot_test"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "size-evidence",
        "## Model size\n\nThe larger model has 1542M parameters.",
    )
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "dataset-evidence",
        "## Dataset\n\nThe training data omitted Wikipedia.",
    )

    hits = service.search_chunks(
        pot_id=pot_id,
        query="What is the model size and which dataset was omitted?",
        limit=5,
    )

    assert {hit["doc_slug"] for hit in hits} == {
        "size-evidence",
        "dataset-evidence",
    }


def test_query_forms_include_independent_multi_facet_forms() -> None:
    from potpie_context_engine.adapters.outbound.resources.sqlite_registry import (
        _fts_query_forms,
    )

    forms = _fts_query_forms(
        "What is the model size and which dataset was omitted?"
    )

    assert any('"model" "size"' in form for form in forms)
    assert any('"dataset" "omitted"' in form for form in forms)


def test_search_chunks_survives_fts_operator_characters(tmp_path: Path) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    pot_id = "pot_test"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "guide",
        "## Quoting\n\nUnbalanced quotes and parens should still find this chunk.",
    )
    hits = service.search_chunks(
        pot_id=pot_id, query='unbalanced "quotes (parens', limit=5
    )
    assert hits


def test_search_chunks_applies_synonyms_inside_each_query_facet(
    tmp_path: Path,
) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    _import_doc(
        service,
        tmp_path,
        "pot_test",
        "decoy",
        "## Archive\n\nAn encyclopedia was excluded from an unrelated archive.",
    )
    _import_doc(
        service,
        tmp_path,
        "pot_test",
        "paper",
        "## Training Dataset\n\nWe removed all Wikipedia documents from WebText because it is a common data source for other datasets.",
    )

    hits = service.search_chunks(
        pot_id="pot_test",
        query="Which encyclopedia was excluded from WebText, and why?",
        limit=5,
    )

    assert hits
    assert hits[0]["doc_slug"] == "paper"
    assert hits[0]["section_slug"] == "training-dataset"
    assert "Wikipedia" in hits[0]["content"]


def test_search_chunks_matches_non_ascii_query(tmp_path: Path) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    pot_id = "pot_test"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "cafe-guide",
        "## Menu\n\nThe café serves espresso; naïve pricing applies in Zürich.",
    )
    assert service.search_chunks(pot_id=pot_id, query="café", limit=5)
    assert service.search_chunks(pot_id=pot_id, query="Zürich espresso", limit=5)


def test_search_chunks_passes_raw_fts_prefix_syntax_through(tmp_path: Path) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    pot_id = "pot_test"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "payments-guide",
        "## Rollback\n\nRollback steps for payments deploy failures.",
    )
    hits = service.search_chunks(pot_id=pot_id, query="rollb*", limit=5)
    assert hits
    assert hits[0]["doc_slug"] == "payments-guide"


def test_search_chunks_or_fallback_ignores_stopword_only_matches(
    tmp_path: Path,
) -> None:
    """A chunk sharing only function words with the query must not come back."""
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "potpie-home"))
    pot_id = "pot_test"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "weather-note",
        "## Weather\n\nThe weather is nice today and how it does change.",
    )
    hits = service.search_chunks(
        pot_id=pot_id, query="how does the deploy work", limit=5
    )
    assert hits == []


def test_search_chunks_empty_query_does_not_create_registry_db(tmp_path: Path) -> None:
    from potpie_context_engine.adapters.outbound.resources.sqlite_registry import (
        SqliteResourceRegistry,
    )

    db_path = tmp_path / "resources" / "registry.db"
    registry = SqliteResourceRegistry(db_path=db_path)
    assert registry.search_chunks(pot_id="pot_test", query="   ", limit=5) == []
    assert not db_path.exists()


def _pdf_stage(
    tmp_path: Path,
    *,
    name: str,
    title: str,
    content: str,
    pages: tuple[int, ...] = (),
    parser_tier: str = "docling",
    artifact_ref: str | None = None,
    element_metadata: dict[str, object] | None = None,
) -> Path:
    stage = tmp_path / name
    section_slug = "details"
    section = SectionManifest(
        slug=section_slug,
        title=title,
        summary="",
        ordinal=0,
        content_hash="",
        chunks=[{"seq": 0, "label": "details"}],
    )
    manifest = ResourceManifest(
        source_ref="file:///reports/quarterly.pdf",
        source_kind="pdf",
        provenance_version=2 if pages else 0,
        parser_tier=parser_tier,
        sections=[section],
    )
    section_dir = stage / section_slug
    section_dir.mkdir(parents=True)
    (stage / "meta.json").write_text(
        json.dumps(manifest.model_dump()), encoding="utf-8"
    )
    (section_dir / "0000.txt").write_text(content, encoding="utf-8")
    if pages:
        element_rows = []
        provenance_rows = []
        for index, page in enumerate(pages, start=1):
            element_id = f"elem-{index:05d}"
            element_rows.append(
                DocumentElementRecord(
                    element_id=element_id,
                    element_type="table" if artifact_ref else "text",
                    text=content,
                    artifact_ref=artifact_ref,
                    page_number=page,
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    **(element_metadata or {}),
                )
            )
            provenance_rows.append(
                ChunkProvenanceRecord(
                    element_id=element_id,
                    page_number=page,
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    char_start=0,
                    char_end=len(content),
                )
            )
        (stage / "elements.jsonl").write_text(
            "".join(json.dumps(row.model_dump()) + "\n" for row in element_rows),
            encoding="utf-8",
        )
        (section_dir / "0000.prov.json").write_text(
            json.dumps(ChunkProvenanceSidecar(provenance=provenance_rows).model_dump()),
            encoding="utf-8",
        )
    return stage


def test_pdf_title_search_returns_truthful_citation_and_artifact_details(
    tmp_path: Path,
) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "home"))
    stage = _pdf_stage(
        tmp_path,
        name="pdf-stage",
        title="Obscure Heading Needle",
        content="Region=west | Revenue=Q4-77881",
        pages=(2, 3),
        artifact_ref="artifacts/table-0001.csv",
    )
    service.import_staging(
        pot_id="pot-a", doc_slug="quarterly-report", staging_dir=stage
    )

    title_hits = service.search_chunks(
        pot_id="pot-a", query="Obscure Heading Needle", limit=5
    )
    assert len(title_hits) == 1
    hit = title_hits[0]
    assert "Obscure Heading Needle" not in hit["content"]
    assert hit["section_title"] == "Obscure Heading Needle"
    assert hit["source_ref"] == "file:///reports/quarterly.pdf"
    assert hit["source_kind"] == "pdf"
    assert hit["page_numbers"] == [2, 3]
    assert hit["citation"] == {
        "chunk_uri": "potpie://res/quarterly-report/details/0000",
        "source_ref": "file:///reports/quarterly.pdf",
        "source_kind": "pdf",
        "pages": [2, 3],
        "pages_available": True,
        "provenance_available": True,
        "parser_tier": "docling",
    }
    assert hit["provenance"][0]["artifact_ref"] == "artifacts/table-0001.csv"
    assert hit["provenance"][0]["element_type"] == "table"

    row_hits = service.search_chunks(
        pot_id="pot-a", query="Q4-77881", limit=5, page_number=3
    )
    assert [row["uri"] for row in row_hits] == [hit["uri"]]


def test_pdf_element_metadata_survives_search_provenance(tmp_path: Path) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "home"))
    stage = _pdf_stage(
        tmp_path,
        name="pdf-metadata",
        title="Table evidence",
        content="Table 3 | Model | Accuracy\nlarge | 6.31%",
        pages=(8,),
        artifact_ref="artifacts/table-0003.csv",
        element_metadata={
            "caption": "Table 3: Model accuracy",
            "table_number": 3,
        },
    )
    service.import_staging(
        pot_id="pot-a", doc_slug="metadata-report", staging_dir=stage
    )

    hit = service.search_chunks(
        pot_id="pot-a", query="Table 3 accuracy 6.31%", limit=5
    )[0]

    provenance = hit["provenance"][0]
    assert provenance["table_number"] == 3
    assert provenance["caption"] == "Table 3: Model accuracy"
    assert "Table 3: Model accuracy" in hit["evidence_text"]
    assert "6.31%" in hit["evidence_text"]
    assert "Page Number: 8" in hit["evidence_text"]


def test_title_intent_searches_section_title_field(tmp_path: Path) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "home"))
    stage = _pdf_stage(
        tmp_path,
        name="pdf-title",
        title="Language Models are Unsupervised Multitask Learners",
        content="Alec Radford and colleagues present a language model.",
    )
    service.import_staging(pot_id="pot-a", doc_slug="paper", staging_dir=stage)

    hits = service.search_chunks(
        pot_id="pot-a", query="What is the paper's title?", limit=5
    )

    assert hits
    assert hits[0]["section_title"] == (
        "Language Models are Unsupervised Multitask Learners"
    )


def test_title_intent_prefers_document_title_over_ordinary_headings(
    tmp_path: Path,
) -> None:
    from potpie_context_engine.adapters.outbound.resources.sqlite_registry import (
        SqliteResourceRegistry,
    )

    registry = SqliteResourceRegistry(tmp_path / "registry.db")
    registry.upsert_document(
        pot_id="pot-a",
        doc_slug="paper",
        source_ref="file:///paper.pdf",
        source_kind="pdf",
    )
    registry.replace_sections(
        pot_id="pot-a",
        doc_slug="paper",
        sections=[
            {
                "slug": "z-document-title",
                "title": "Language Models are Unsupervised Multitask Learners",
                "ordinal": 0,
            },
            {"slug": "introduction", "title": "1. Introduction", "ordinal": 1},
        ],
    )
    registry.replace_chunk_text_document(
        pot_id="pot-a",
        doc_slug="paper",
        rows=[
            {
                "section_slug": "z-document-title",
                "seq": 0,
                "section_title": "Language Models are Unsupervised Multitask Learners",
                "content": "Alec Radford and colleagues.",
            },
            {
                "section_slug": "introduction",
                "seq": 0,
                "section_title": "1. Introduction",
                "content": "This paper studies language models.",
            },
        ],
    )

    hits = registry.search_chunks(
        pot_id="pot-a", query="What is the paper title?", limit=5
    )

    assert hits[0]["section_slug"] == "z-document-title"


def test_search_chunks_abstains_when_exact_named_metric_is_unsupported(
    tmp_path: Path,
) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "home"))
    pot_id = "pot-a"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "paper",
        "## Results\n\nThe model achieves strong zero-shot results on language tasks.",
    )

    hits = service.search_chunks(
        pot_id=pot_id,
        query="What exact zero-shot ImageNet accuracy did GPT-2 achieve?",
        limit=5,
    )

    assert hits == []


def test_multi_facet_search_reports_complete_evidence_bundle(tmp_path: Path) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "home"))
    pot_id = "pot-a"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "size-evidence",
        "## Model size\n\nThe larger model has 1542M parameters.",
    )
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "dataset-evidence",
        "## Dataset\n\nThe training data omitted Wikipedia.",
    )

    hits = service.search_chunks(
        pot_id=pot_id,
        query="What is the model size and which dataset was omitted?",
        limit=5,
    )

    assert {tuple(hit["facet_coverage"]) for hit in hits} >= {(0,), (1,)}
    assert all(hit["answer_bundle_complete"] is True for hit in hits)


def test_table_identifier_and_numeric_value_are_structural_matches(
    tmp_path: Path,
) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "home"))
    pot_id = "pot-a"
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "table-10",
        "## Results\n\nTable 10 columns: Model | Accuracy\nRow 1: GPT-2 Accuracy=35.0%",
    )
    _import_doc(
        service,
        tmp_path,
        pot_id,
        "table-3",
        "## Results\n\nTable 3 columns: Model | Accuracy\nRow 1: GPT-2 Accuracy=30.0%",
    )

    hits = service.search_chunks(
        pot_id=pot_id, query="Table 10 GPT-2 accuracy 35.0%", limit=5
    )

    assert hits[0]["doc_slug"] == "table-10"
    assert hits[0]["structural_match"]["identifier_matches"] == 1
    assert hits[0]["structural_match"]["numeric_matches"] >= 1


def test_resource_search_filters_unknown_metadata_and_isolates_pots(
    tmp_path: Path,
) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "home"))
    pdf_stage = _pdf_stage(
        tmp_path,
        name="pdf-filter",
        title="Metrics",
        content="isolation-token",
        pages=(5,),
    )
    degraded_stage = _pdf_stage(
        tmp_path,
        name="pdf-degraded",
        title="Metrics",
        content="isolation-token",
        parser_tier="pypdf-degraded",
    )
    service.import_staging(pot_id="pot-a", doc_slug="known-pdf", staging_dir=pdf_stage)
    service.import_staging(
        pot_id="pot-b", doc_slug="unknown-pages", staging_dir=degraded_stage
    )

    assert service.search_chunks(
        pot_id="pot-a",
        query="isolation-token",
        limit=5,
        doc_slug="known-pdf",
        source_kind="pdf",
        page_number=5,
    )
    assert (
        service.search_chunks(
            pot_id="pot-a", query="isolation-token", limit=5, page_number=99
        )
        == []
    )
    assert (
        service.search_chunks(
            pot_id="pot-a", query="isolation-token", limit=5, source_kind="markdown"
        )
        == []
    )
    assert (
        service.search_chunks(
            pot_id="pot-a", query="isolation-token", limit=5, doc_slug="missing"
        )
        == []
    )
    assert (
        service.search_chunks(
            pot_id="pot-b", query="isolation-token", limit=5, page_number=5
        )
        == []
    )

    unknown_hit = service.search_chunks(
        pot_id="pot-b", query="isolation-token", limit=5
    )[0]
    assert unknown_hit["page_numbers"] == []
    assert unknown_hit["citation"]["provenance_available"] is False
    assert unknown_hit["citation"]["parser_tier"] == "pypdf-degraded"


def test_registry_migrates_existing_fts_and_backfills_title_and_page(
    tmp_path: Path,
) -> None:
    from potpie_context_engine.adapters.outbound.resources.sqlite_registry import (
        SqliteResourceRegistry,
    )
    from potpie_context_engine.core.ports.resource_index import ResourceSearchFilters

    db_path = tmp_path / "registry.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (
                pot_id TEXT NOT NULL, doc_slug TEXT NOT NULL,
                source_ref TEXT, source_kind TEXT, revision INTEGER,
                updated_at TEXT, PRIMARY KEY (pot_id, doc_slug)
            );
            CREATE TABLE sections (
                pot_id TEXT NOT NULL, doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL, title TEXT, summary TEXT,
                content_hash TEXT, ordinal INTEGER,
                PRIMARY KEY (pot_id, doc_slug, section_slug)
            );
            CREATE TABLE chunk_provenance (
                pot_id TEXT NOT NULL, doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL, seq INTEGER NOT NULL,
                element_id TEXT NOT NULL, page_number INTEGER,
                bbox_json TEXT, char_start INTEGER, char_end INTEGER,
                PRIMARY KEY (pot_id, doc_slug, section_slug, seq, element_id)
            );
            CREATE TABLE chunk_text (
                pot_id TEXT NOT NULL, doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL, seq INTEGER NOT NULL,
                content TEXT NOT NULL, ocr_text TEXT,
                PRIMARY KEY (pot_id, doc_slug, section_slug, seq)
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                pot_id UNINDEXED, doc_slug UNINDEXED, section_slug UNINDEXED,
                seq UNINDEXED, content, ocr_text
            );
            INSERT INTO documents VALUES (
                'pot-old', 'legacy', 'file:///legacy.pdf', 'pdf', 1, '2026-01-01'
            );
            INSERT INTO sections VALUES (
                'pot-old', 'legacy', 'body', 'Migrated Heading', '', '', 0
            );
            INSERT INTO chunk_provenance VALUES (
                'pot-old', 'legacy', 'body', 0, 'elem-1', 9, NULL, 0, 12
            );
            INSERT INTO chunk_text VALUES (
                'pot-old', 'legacy', 'body', 0, 'legacy body', ''
            );
            INSERT INTO chunks_fts VALUES (
                'pot-old', 'legacy', 'body', 0, 'legacy body', ''
            );
            """
        )

    registry = SqliteResourceRegistry(db_path=db_path)
    rows = registry.search_chunks(
        "pot-old",
        "Migrated Heading",
        filters=ResourceSearchFilters(source_kind="pdf", page_number=9),
    )
    assert [(row["doc_slug"], row["section_slug"], row["seq"]) for row in rows] == [
        ("legacy", "body", 0)
    ]


def test_registry_migration_recovers_after_interruption(tmp_path: Path) -> None:
    from potpie_context_engine.adapters.outbound.resources.sqlite_registry import (
        SqliteResourceRegistry,
    )

    db_path = tmp_path / "interrupted.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (
                pot_id TEXT NOT NULL, doc_slug TEXT NOT NULL,
                source_ref TEXT, source_kind TEXT, revision INTEGER,
                updated_at TEXT, PRIMARY KEY (pot_id, doc_slug)
            );
            CREATE TABLE sections (
                pot_id TEXT NOT NULL, doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL, title TEXT, summary TEXT,
                content_hash TEXT, ordinal INTEGER,
                PRIMARY KEY (pot_id, doc_slug, section_slug)
            );
            CREATE TABLE chunk_provenance (
                pot_id TEXT NOT NULL, doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL, seq INTEGER NOT NULL,
                element_id TEXT NOT NULL, page_number INTEGER,
                bbox_json TEXT, char_start INTEGER, char_end INTEGER,
                PRIMARY KEY (pot_id, doc_slug, section_slug, seq, element_id)
            );
            CREATE TABLE chunk_text (
                pot_id TEXT NOT NULL, doc_slug TEXT NOT NULL,
                section_slug TEXT NOT NULL, seq INTEGER NOT NULL,
                content TEXT NOT NULL, ocr_text TEXT,
                PRIMARY KEY (pot_id, doc_slug, section_slug, seq)
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                pot_id UNINDEXED, doc_slug UNINDEXED, section_slug UNINDEXED,
                seq UNINDEXED, content, ocr_text
            );
            INSERT INTO documents VALUES (
                'pot-old', 'legacy', 'file:///legacy.pdf', 'pdf', 1, '2026-01-01'
            );
            INSERT INTO sections VALUES (
                'pot-old', 'legacy', 'body', 'Retryable Heading', '', '', 0
            );
            INSERT INTO chunk_text VALUES (
                'pot-old', 'legacy', 'body', 0, 'legacy body', ''
            );
            INSERT INTO chunks_fts VALUES (
                'pot-old', 'legacy', 'body', 0, 'legacy body', ''
            );
            """
        )

    class _InterruptedRegistry(SqliteResourceRegistry):
        @staticmethod
        def _rebuild_fts_rows(conn, *, pot_id=None, doc_slug=None) -> None:
            del conn, pot_id, doc_slug
            raise sqlite3.OperationalError("injected migration interruption")

    with pytest.raises(sqlite3.OperationalError, match="injected migration"):
        _InterruptedRegistry(db_path=db_path).connect()

    rows = SqliteResourceRegistry(db_path=db_path).search_chunks(
        "pot-old", "Retryable Heading"
    )
    assert [(row["doc_slug"], row["section_slug"]) for row in rows] == [
        ("legacy", "body")
    ]


def test_reimport_and_remove_clear_stale_text_title_page_and_metadata(
    tmp_path: Path,
) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "home"))
    first = _pdf_stage(
        tmp_path,
        name="revision-one",
        title="LegacyNeedleOnly",
        content="old-row-token",
        pages=(7,),
    )
    second = _pdf_stage(
        tmp_path,
        name="revision-two",
        title="Current Heading",
        content="new-row-token",
        pages=(8,),
    )
    service.import_staging(pot_id="pot-a", doc_slug="report", staging_dir=first)
    service.import_staging(
        pot_id="pot-a",
        doc_slug="report",
        staging_dir=second,
        force=True,
    )

    assert (
        service.search_chunks(pot_id="pot-a", query="old-row-token", page_number=7)
        == []
    )
    assert service.search_chunks(pot_id="pot-a", query="LegacyNeedleOnly") == []
    assert service.search_chunks(pot_id="pot-a", query="new-row-token", page_number=8)

    service.remove_document(pot_id="pot-a", doc_slug="report")
    assert service.search_chunks(pot_id="pot-a", query="new-row-token") == []
    assert service.list_documents(pot_id="pot-a") == []


def test_source_ref_override_is_used_by_search_and_document_show(
    tmp_path: Path,
) -> None:
    service = ResourceService(store=LocalResourceStore(home=tmp_path / "home"))
    stage = _pdf_stage(
        tmp_path,
        name="source-override",
        title="Reference",
        content="override-token",
        pages=(2,),
    )
    explicit_ref = "https://docs.example.test/report.pdf"

    service.import_staging(
        pot_id="pot-a",
        doc_slug="report",
        staging_dir=stage,
        source_ref=explicit_ref,
    )

    hit = service.search_chunks(pot_id="pot-a", query="override-token")[0]
    shown = service.get_chunk(pot_id="pot-a", uri=hit["uri"])
    assert hit["source_ref"] == explicit_ref
    assert hit["citation"]["source_ref"] == explicit_ref
    assert shown["source_ref"] == explicit_ref
    assert shown["citation"]["source_ref"] == explicit_ref
