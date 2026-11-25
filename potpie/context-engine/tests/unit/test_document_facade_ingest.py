"""The parser command and the canonical resource facade share one document."""

from pathlib import Path
import json
from types import SimpleNamespace

from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
    LocalResourceStore,
)
from potpie_context_engine.adapters.outbound.resources.fact_inbox import FactInbox
from potpie_context_engine.application.services.resource_facade import ResourceFacade
from potpie_context_engine.application.services.resource_service import ResourceService
from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    parse_file_to_staging,
)
from potpie_context_engine.adapters.outbound.resources.snapshot import (
    export_resources,
    restore_resources,
)
from potpie_context_engine.core.ports.resource_store import format_resource_id
from potpie_context_engine.api import DEFAULT_GRAPH_DEFINITION
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.runtime import build_graph_runtime
from potpie_context_engine.testing import (
    InMemoryGraphInboxStore,
    InMemoryGraphPlanStore,
    build_test_backend,
)


def test_document_ingest_writes_revisioned_canonical_resources(tmp_path: Path) -> None:
    source = tmp_path / "note.md"
    source.write_text("# Overview\nThe API uses SQLite.\n", encoding="utf-8")
    canonical = LocalResourceStore(home=tmp_path / "home")
    service = ResourceService(facade=ResourceFacade(store=canonical))

    first = service.ingest_file(pot_id="pot-a", doc_slug="note", source_path=source)
    assert first.sections_added == ["body"]
    assert service.list_documents(pot_id="pot-a")[0]["revision"] == 1
    first_chunk = service.get_chunk(pot_id="pot-a", uri="potpie://res/note/body/0000")
    assert "SQLite" in first_chunk["content"]

    source.write_text("# Overview\nThe API now uses Postgres.\n", encoding="utf-8")
    second = service.ingest_file(pot_id="pot-a", doc_slug="note", source_path=source)
    assert second.sections_changed == ["body"]
    assert service.list_documents(pot_id="pot-a")[0]["revision"] == 2
    assert (
        "Postgres"
        in service.get_chunk(pot_id="pot-a", uri="potpie://res/note/body/0000")[
            "content"
        ]
    )
    assert (
        "SQLite" in service.get_chunk(pot_id="pot-a", uri=first_chunk["uri"])["content"]
    )


def test_provenance_and_binary_artifacts_survive_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("Diagram shows Billing.\n", encoding="utf-8")
    stage = tmp_path / "stage"
    parse_file_to_staging(source, stage)
    (stage / "body" / "0000.prov.json").write_text(
        json.dumps({"provenance": [{"element_id": "figure-1", "page_number": 2}]}),
        encoding="utf-8",
    )
    (stage / "body" / "0000.ocr.txt").write_text("Billing", encoding="utf-8")
    (stage / "elements.jsonl").write_text(
        '{"element_id":"figure-1","element_type":"picture",'
        '"artifact_ref":"artifacts/figure.png","figure_number":1,"page_number":2}\n',
        encoding="utf-8",
    )
    (stage / "artifacts").mkdir()
    (stage / "artifacts" / "figure.png").write_bytes(b"\x89PNG\x00\xff")
    store = LocalResourceStore(home=tmp_path / "first")
    store.import_dir(pot_id="pot", slug="note", source_dir=stage)
    uri = format_resource_id("note", "body", 0, revision=1)
    chunk = store.get(pot_id="pot", resource_id=uri)
    assert chunk.page_numbers == (2,)
    assert chunk.ocr_text == "Billing"
    assert chunk.provenance[0]["element_id"] == "figure-1"
    shown = ResourceService(facade=ResourceFacade(store=store)).get_chunk(
        pot_id="pot", uri=uri
    )
    assert shown["figure_artifacts"] == [
        {
            "figure_number": 1,
            "page_number": 2,
            "path": str(
                store._pot_root("pot") / "note" / "artifacts" / "figure.png"
            ),
            "visual_caption_unverified": True,
        }
    ]

    payload = export_resources(store, pot_id="pot")
    assert payload["binary_files"]
    restored = LocalResourceStore(home=tmp_path / "restored")
    with restore_resources(restored, pot_id="pot", payload=payload):
        pass
    assert restored.get(pot_id="pot", resource_id=uri).provenance == chunk.provenance


def test_facade_projects_document_elements_into_graph(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("Figure caption.\n", encoding="utf-8")
    stage = tmp_path / "stage"
    parse_file_to_staging(source, stage)
    (stage / "elements.jsonl").write_text(
        '{"element_id":"figure-1","element_type":"figure","text":"Figure caption.","page_number":1}\n',
        encoding="utf-8",
    )
    operations = []

    class Graph:
        def mutate(self, request, resolve_operations=None):
            operations.extend(resolve_operations())
            return SimpleNamespace(
                ok=True, status="applied", claim_keys=(), review_required_claim_keys=()
            )

    class Claims:
        def find_claims(self, query):
            return ()

    facade = ResourceFacade(
        store=LocalResourceStore(home=tmp_path / "home"), graph=Graph(), claims=Claims()
    )
    result = facade.import_dir(pot_id="pot", slug="note", source_dir=stage)
    assert result.graph_written
    assert any(op.predicate == "ELEMENT_OF" for op in operations)


def test_document_elements_apply_through_real_graph_service(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("Figure caption.\n", encoding="utf-8")
    stage = tmp_path / "stage"
    parse_file_to_staging(source, stage)
    (stage / "elements.jsonl").write_text(
        '{"element_id":"figure-1","element_type":"figure","text":"Figure caption.","page_number":1}\n',
        encoding="utf-8",
    )
    store = LocalResourceStore(home=tmp_path / "home")
    runtime = build_graph_runtime(
        build_test_backend(),
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        DEFAULT_GRAPH_DEFINITION,
        resource_store=store,
    )
    facade = ResourceFacade.from_runtime(runtime, store=store)
    result = facade.import_dir(pot_id="pot", slug="note", source_dir=stage)
    assert result.graph_written
    rows = runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id="pot",
            predicate_in=("ELEMENT_OF",),
        )
    )
    assert [row.subject_key for row in rows] == ["document-element:note/figure-1"]


def test_document_ingest_without_flag_does_not_start_fact_inbox(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("Billing uses Postgres.\n", encoding="utf-8")
    store = LocalResourceStore(home=tmp_path / "home")
    runtime = build_graph_runtime(
        build_test_backend(),
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        DEFAULT_GRAPH_DEFINITION,
        resource_store=store,
    )
    service = ResourceService(facade=ResourceFacade.from_runtime(runtime, store=store))
    report = service.ingest_file(pot_id="pot", doc_slug="note", source_path=source)
    assert report.graph_written
    assert report.facts_verified == report.facts_proposed == 0
    assert not (store.home / "fact_inbox.sqlite3").exists()

    inbox = FactInbox(home=store.home)
    item = inbox.propose(
        pot_id="pot",
        doc="note",
        revision=1,
        payload={"subject": "Billing", "predicate": "USES", "object": "Postgres"},
        reason="review",
    )
    source.write_text("Billing now uses SQLite.\n", encoding="utf-8")
    service.ingest_file(pot_id="pot", doc_slug="note", source_path=source)
    assert inbox.get(pot_id="pot", item_id=item["id"])["status"] == "closed"
