from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from potpie.cli.commands import _common
from potpie.cli.commands import document as document_commands
from potpie.cli.commands import query as query_commands
from potpie_context_engine.domain.resource_models import ResourceImportReport


def test_search_threads_document_metadata_scope_to_engine(monkeypatch) -> None:
    requests = []

    class _Client:
        def search(self, request):
            requests.append(request)
            return SimpleNamespace(
                pot_id="pot-1",
                intent="unknown",
                overall_confidence=0.0,
                items=(),
                coverage=(),
                unsupported_includes=(),
            )

    monkeypatch.setattr(query_commands, "get_engine_client", lambda pot: _Client())
    monkeypatch.setattr(query_commands, "run_engine_operation", lambda result: result)
    monkeypatch.setattr(query_commands, "_capture_context_activation", lambda **_: None)
    monkeypatch.setattr(query_commands, "emit", lambda payload, human: None)

    app = typer.Typer()
    query_commands.register(app)
    result = CliRunner().invoke(
        app,
        [
            "search",
            "needle",
            "--include",
            "docs",
            "--scope",
            "doc_slug:guide,source_kind:pdf,page:3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert requests[0].scope == {
        "doc_slug": "guide",
        "source_kind": "pdf",
        "page": "3",
    }


def test_human_search_displays_document_preview_uri_and_truthful_citation() -> None:
    item = SimpleNamespace(
        include="docs",
        payload={
            "content_preview": "Install dependencies with pip.",
            "chunk_uri": "potpie://res/guide/setup/0000",
            "properties": {
                "source_ref": "file:///guide.pdf",
                "page_numbers": [3],
            },
        },
    )
    env = SimpleNamespace(
        pot_id="pot-1",
        intent="unknown",
        overall_confidence=0.8,
        items=(item,),
        unsupported_includes=(),
    )

    human = query_commands._envelope_human(env)

    assert "Install dependencies with pip." in human
    assert "potpie://res/guide/setup/0000" in human
    assert "file:///guide.pdf" in human
    assert "page 3" in human


def test_document_ingest_does_not_emit_graph_incomplete_report_as_success(capsys) -> None:
    _common.set_json(True)
    report = ResourceImportReport(
        pot_id="pot-1", doc_slug="guide", graph_written=False
    )

    with pytest.raises(typer.Exit) as exc_info:
        document_commands._emit_import_report(report)

    assert exc_info.value.exit_code == 2
    assert '"code": "resource_graph_write_failed"' in capsys.readouterr().out
