"""Unit tests for resource import + FTS registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    LocalResourceStore,
)
from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    parse_file_to_staging,
)

pytestmark = pytest.mark.unit


def test_import_and_search_chunks(tmp_path: Path) -> None:
    home = tmp_path / "potpie-home"
    store = LocalResourceStore(home=home)
    pot_id = "pot_test"
    source = tmp_path / "guide.md"
    source.write_text("## Payments rollback\n\nRollback steps for payments deploy failures.", encoding="utf-8")
    staging = tmp_path / "staging"
    manifest = parse_file_to_staging(source, staging)
    report = store.import_manifest(
        pot_id=pot_id,
        doc_slug="payments-guide",
        staging_dir=staging,
        manifest=manifest,
    )
    assert not report.errors
    hits = store.search_chunks(pot_id=pot_id, query="rollback", limit=5)
    assert hits
    chunk = store.get_chunk_text(
        pot_id=pot_id,
        doc_slug="payments-guide",
        section_slug=manifest.sections[0].slug,
        seq=0,
    )
    assert "rollback" in chunk["content"].lower()
