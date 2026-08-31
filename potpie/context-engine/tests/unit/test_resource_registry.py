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
from potpie_context_engine.application.services.resource_service import (
    ResourceService,
)

pytestmark = pytest.mark.unit


def test_import_and_search_chunks(tmp_path: Path) -> None:
    home = tmp_path / "potpie-home"
    store = LocalResourceStore(home=home)
    service = ResourceService(store=store)
    pot_id = "pot_test"
    source = tmp_path / "guide.md"
    source.write_text("## Payments rollback\n\nRollback steps for payments deploy failures.", encoding="utf-8")
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


def _import_doc(service: ResourceService, tmp_path: Path, pot_id: str, slug: str, body: str) -> None:
    source = tmp_path / f"{slug}.md"
    source.write_text(body, encoding="utf-8")
    staging = tmp_path / f"staging-{slug}"
    parse_file_to_staging(source, staging)
    report = service.import_staging(pot_id=pot_id, doc_slug=slug, staging_dir=staging)
    assert not report.errors


def test_search_chunks_natural_language_query_matches_on_some_terms(tmp_path: Path) -> None:
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


def test_search_chunks_or_fallback_ignores_stopword_only_matches(tmp_path: Path) -> None:
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
    hits = service.search_chunks(pot_id=pot_id, query="how does the deploy work", limit=5)
    assert hits == []


def test_search_chunks_empty_query_does_not_create_registry_db(tmp_path: Path) -> None:
    from potpie_context_engine.adapters.outbound.resources.sqlite_registry import (
        SqliteResourceRegistry,
    )

    db_path = tmp_path / "resources" / "registry.db"
    registry = SqliteResourceRegistry(db_path=db_path)
    assert registry.search_chunks(pot_id="pot_test", query="   ", limit=5) == []
    assert not db_path.exists()
