"""Document fact extraction is opt-in, evidence bound, and reviewable."""

import io
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import time
import urllib.request
from urllib.error import HTTPError

from potpie_context_engine.adapters.outbound.resources.fact_inbox import FactInbox
from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
    LocalResourceStore,
)
from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    parse_file_to_staging,
)
from potpie_context_engine.application.services.fact_extraction import (
    FactExtractor,
    _configured_llm,
)
from potpie_context_engine.application.services.resource_facade import ResourceFacade
from potpie_context_engine.api import DEFAULT_GRAPH_DEFINITION
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.runtime import build_graph_runtime
from potpie_context_engine.testing import (
    InMemoryGraphInboxStore,
    InMemoryGraphPlanStore,
    build_test_backend,
)


def test_gemini_empty_fact_list_is_valid(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_FACTS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("POTPIE_FACTS_LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(
            b'{"choices":[{"message":{"content":"[]"}}]}'
        ),
    )

    assert _configured_llm("A flowchart explains a process.") == {"facts": []}


def test_quota_exhaustion_stops_further_fact_requests(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("## First\nBilling uses Postgres.\n## Second\nAPI uses Redis.\n")
    staging = tmp_path / "staging"
    parse_file_to_staging(source, staging)
    store = LocalResourceStore(home=tmp_path / "home")
    store.import_dir(pot_id="pot", slug="note", source_dir=staging)
    facade = ResourceFacade(store=store, graph=_Graph(), claims=_Claims())
    calls = []

    def exhausted(text: str) -> dict:
        calls.append(text)
        raise HTTPError("https://example.test", 429, "Too Many Requests", {}, None)

    result = FactExtractor(facade, llm=exhausted).run(pot_id="pot", doc="note")
    assert len(calls) == 1
    assert "quota or rate limit" in result["error"]


def test_provider_outage_stops_further_fact_requests(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("## First\nOne.\n## Second\nTwo.\n")
    staging = tmp_path / "staging"
    parse_file_to_staging(source, staging)
    store = LocalResourceStore(home=tmp_path / "home")
    store.import_dir(pot_id="pot", slug="note", source_dir=staging)
    facade = ResourceFacade(store=store, graph=_Graph(), claims=_Claims())
    calls = []

    def unavailable(text: str) -> dict:
        calls.append(text)
        raise HTTPError("https://example.test", 503, "Unavailable", {}, None)

    result = FactExtractor(facade, llm=unavailable).run(pot_id="pot", doc="note")
    assert len(calls) == 1
    assert "temporarily unavailable" in result["error"]


def test_retry_skips_successful_chunks_after_quota_failure(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.md"
    source.write_text("## First\nOne.\n## Second\nTwo.\n")
    staging = tmp_path / "staging"
    parse_file_to_staging(source, staging)
    store = LocalResourceStore(home=tmp_path / "home")
    store.import_dir(pot_id="pot", slug="note", source_dir=staging)
    facade = ResourceFacade(store=store, graph=_Graph(), claims=_Claims())
    monkeypatch.setenv("POTPIE_FACTS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("POTPIE_FACTS_LLM_API_KEY", "test-key")
    calls = []

    def answer(request, timeout):
        calls.append(request)
        if len(calls) == 2:
            raise HTTPError("https://example.test", 429, "quota", {}, None)
        return io.BytesIO(b'{"choices":[{"message":{"content":"[]"}}]}')

    monkeypatch.setattr(urllib.request, "urlopen", answer)
    extractor = FactExtractor(facade)
    first = extractor.run(pot_id="pot", doc="note")
    assert "quota or rate limit" in first["error"]
    second = FactExtractor(facade).run(pot_id="pot", doc="note")
    assert second["error"] is None
    assert len(calls) == 3


class _Graph:
    def __init__(self) -> None:
        self.mutations = []

    def search_entities(self, request):
        return SimpleNamespace(entities=())

    def mutate(self, request, resolve_operations=None):
        self.mutations.append(request)
        if resolve_operations is not None:
            self.mutations.append(SimpleNamespace(operations=resolve_operations()))
        return SimpleNamespace(
            ok=True, status="applied", claim_keys=(), review_required_claim_keys=()
        )


class _Claims:
    def find_claims(self, query):
        return ()


def _facade(tmp_path: Path) -> tuple[ResourceFacade, _Graph]:
    source = tmp_path / "source.md"
    source.write_text("Billing uses Postgres.\n", encoding="utf-8")
    staging = tmp_path / "staging"
    parse_file_to_staging(source, staging)
    store = LocalResourceStore(home=tmp_path / "home")
    store.import_dir(pot_id="pot", slug="note", source_dir=staging)
    graph = _Graph()
    return ResourceFacade(store=store, graph=graph, claims=_Claims()), graph


def _fact(confidence: float = 0.98) -> dict:
    return {
        "subject": "Billing",
        "subject_type": "Service",
        "predicate": "USES",
        "object": "Postgres",
        "object_type": "DataStore",
        "quote": "Billing uses Postgres.",
        "confidence": confidence,
    }


def test_high_confidence_fact_is_written_once_with_revisioned_evidence(
    tmp_path: Path,
) -> None:
    facade, graph = _facade(tmp_path)
    extractor = FactExtractor(facade, llm=lambda text: {"facts": [_fact()]})

    assert extractor.run(pot_id="pot", doc="note") == {
        "verified": 1,
        "proposed": 0,
        "suppressed": 0,
        "error": None,
    }
    operation = graph.mutations[0].operations[0]
    assert operation.predicate == "USES"
    assert operation.evidence[0].source_ref.endswith("@rev1")
    assert extractor.run(pot_id="pot", doc="note")["suppressed"] == 1
    assert len(graph.mutations) == 1


def test_visual_caption_quote_is_staged_even_at_high_confidence(tmp_path: Path) -> None:
    facade, graph = _facade(tmp_path)
    extractor = FactExtractor(facade)
    fact = extractor._fact(
        _fact(), uri="potpie://res/note/body/0000@rev1", page=1
    )
    reason, _, _ = extractor._gate(
        fact,
        chunk_text="Figure 1\nVisual caption: Billing uses Postgres.",
        pot_id="pot",
        available_pages=(1,),
        source_kind="pdf",
    )
    assert reason == "visual or OCR evidence requires review"
    assert graph.mutations == []


def test_pdf_native_text_quote_can_still_pass_gate(tmp_path: Path) -> None:
    facade, _ = _facade(tmp_path)
    extractor = FactExtractor(facade)
    fact = extractor._fact(
        _fact(), uri="potpie://res/note/body/0000@rev1", page=1
    )
    reason, subject, object_ = extractor._gate(
        fact,
        chunk_text="Billing uses Postgres.\n\nFigure 1\nVisual caption: A diagram",
        pot_id="pot",
        available_pages=(1,),
        source_kind="pdf",
    )
    assert reason is None
    assert subject == "service:billing"
    assert object_ == "datastore:postgres"


def test_low_confidence_rejection_stays_suppressed_for_same_revision(
    tmp_path: Path,
) -> None:
    facade, graph = _facade(tmp_path)
    extractor = FactExtractor(facade, llm=lambda text: {"facts": [_fact(0.5)]})
    assert extractor.run(pot_id="pot", doc="note")["proposed"] == 1
    item = extractor.inbox.list(pot_id="pot")[0]
    assert item["reason"] == "low extraction confidence"
    assert extractor.inbox.transition(
        pot_id="pot",
        item_id=item["id"],
        status="rejected",
    )
    assert extractor.run(pot_id="pot", doc="note")["suppressed"] == 1
    assert graph.mutations == []


def test_inbox_closes_superseded_and_prunes_terminal_records(tmp_path: Path) -> None:
    inbox = FactInbox(home=tmp_path)
    item = inbox.propose(
        pot_id="pot",
        doc="note",
        revision=1,
        payload=_fact(),
        reason="review",
    )
    assert inbox.close_superseded(pot_id="pot", doc="note", revision=2) == 1
    assert inbox.get(pot_id="pot", item_id=item["id"])["status"] == "closed"
    assert inbox.prune(pot_id="pot", older_than_days=0) == 1
    assert inbox.get(pot_id="pot", item_id=item["id"]) is None


def test_pending_candidates_expire_after_configured_ttl(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_FACT_INBOX_TTL_DAYS", "30")
    inbox = FactInbox(home=tmp_path)
    item = inbox.propose(
        pot_id="pot",
        doc="note",
        revision=1,
        payload=_fact(),
        reason="review",
    )
    with sqlite3.connect(inbox.path) as db:
        db.execute(
            "UPDATE fact_candidates SET created_at=? WHERE id=?",
            (int(time.time()) - 31 * 86400, item["id"]),
        )
    assert inbox.list(pot_id="pot") == []
    assert inbox.get(pot_id="pot", item_id=item["id"])["status"] == "closed"


def test_fact_uses_graph_service_semantic_mutations(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("Billing uses Postgres.\n", encoding="utf-8")
    stage = tmp_path / "stage"
    parse_file_to_staging(source, stage)
    store = LocalResourceStore(home=tmp_path / "home")
    runtime = build_graph_runtime(
        build_test_backend(),
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        DEFAULT_GRAPH_DEFINITION,
        resource_store=store,
    )
    facade = ResourceFacade.from_runtime(runtime, store=store)
    assert facade.import_dir(pot_id="pot", slug="note", source_dir=stage).graph_written

    result = FactExtractor(facade, llm=lambda text: {"facts": [_fact()]}).run(
        pot_id="pot",
        doc="note",
    )
    assert result["verified"] == 1, result
    rows = runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id="pot",
            predicate_in=("USES",),
        )
    )
    assert len(rows) == 1
    assert rows[0].source_refs[0].endswith("@rev1")
