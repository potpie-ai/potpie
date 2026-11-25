"""The small fact review command surface remains callable."""

from typer.testing import CliRunner

from potpie.cli.commands import graph
from potpie.cli.main import app
from potpie.cli.commands.document import _format_import_report
from potpie.cli.commands.resource import _chunk_payload
from potpie_context_engine.core.resource_models import ResourceImportReport
from potpie_context_engine.core.ports.resource_store import Chunk


def test_fact_inbox_commands_are_wired(monkeypatch) -> None:
    calls = []

    class Inbox:
        def list(self, **kwargs):
            calls.append(("list", kwargs))
            return []

        def clear(self, **kwargs):
            calls.append(("clear", kwargs))
            return 2

        def prune(self, **kwargs):
            calls.append(("prune", kwargs))
            return 1

        def transition(self, **kwargs):
            calls.append(("reject", kwargs))
            return True

        def get(self, **kwargs):
            return {"id": kwargs["item_id"], "status": "rejected"}

    class Extractor:
        def approve(self, **kwargs):
            calls.append(("approve", kwargs))
            return {"id": kwargs["item_id"], "status": "verified"}

    monkeypatch.setattr(graph, "get_root_runtime", lambda: object())
    monkeypatch.setattr(graph, "resolve_pot_id", lambda host, pot: "pot")
    monkeypatch.setattr(graph, "_document_fact_inbox", Inbox)
    monkeypatch.setattr(graph, "_document_fact_extractor", Extractor)
    runner = CliRunner()

    assert runner.invoke(app, ["graph", "inbox"]).exit_code == 0
    assert (
        runner.invoke(app, ["graph", "inbox", "clear", "--older-than", "30d"]).exit_code
        == 0
    )
    assert runner.invoke(app, ["graph", "inbox", "prune"]).exit_code == 0
    assert runner.invoke(app, ["graph", "approve", "fact_1"]).exit_code == 0
    assert runner.invoke(app, ["graph", "reject", "fact_2"]).exit_code == 0
    assert [name for name, _ in calls] == [
        "list",
        "clear",
        "prune",
        "approve",
        "reject",
    ]


def test_document_ingest_human_report_shows_partial_vision_coverage() -> None:
    report = ResourceImportReport(
        pot_id="pot",
        doc_slug="diagrams",
        ingestion_diagnostics={
            "picture_enrichment": {
                "pictures_total": 17,
                "pictures_with_ocr": 16,
                "pictures_with_visual_caption": 13,
                "warnings": ["partial_picture_visual_caption_coverage"],
            }
        },
    )
    rendered = _format_import_report(report)
    assert "figures=17 ocr=16 vision_captions=13" in rendered
    assert "partial_picture_visual_caption_coverage" in rendered


def test_resource_get_suggests_visual_review_for_pdf_figure() -> None:
    uri = "potpie://res/diagrams/company/0000@rev1"
    chunk = Chunk(
        resource_id=uri,
        doc="diagrams",
        section="company",
        text="Figure 1\nVisual caption: A flowchart",
        source_kind="pdf",
    )
    result = _chunk_payload(chunk, (uri,))
    assert result["visual_review_command"] == f"potpie document show {uri}"
