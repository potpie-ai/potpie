"""``ResourceFacade.from_runtime`` — the supported way to compose documents.

An embedding host builds its ``GraphRuntime`` with ``build_graph_runtime`` and
then the facade from that runtime, so the graph service, claim query, snapshot
port and journal never have to be patched onto the facade by hand.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.api import DEFAULT_GRAPH_DEFINITION, ResourceFacade
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.ports.resource_index import IndexReport
from potpie_context_engine.core.runtime import build_graph_runtime
from potpie_context_engine.testing import (
    InMemoryGraphInboxStore,
    InMemoryGraphPlanStore,
    InMemoryResourceStore,
    build_test_backend,
    write_import_directory,
)

pytestmark = pytest.mark.unit


class _Index:
    profile = "test"

    def __init__(self) -> None:
        self.indexed: list[str] = []

    def index_document(self, *, pot_id, manifest, chunks):
        self.indexed.append(manifest.doc)
        return IndexReport(doc=manifest.doc, profile="test", chunks=len(chunks))


class _Drain:
    def __init__(self) -> None:
        self.signals = 0

    def signal(self) -> None:
        self.signals += 1


def _runtime(store, index):
    return build_graph_runtime(
        build_test_backend(),
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        DEFAULT_GRAPH_DEFINITION,
        resource_index=index,
        resource_store=store,
    )


def test_from_runtime_wires_every_runtime_port(tmp_path) -> None:
    store, index, drain = InMemoryResourceStore(), _Index(), _Drain()
    runtime = _runtime(store, index)

    facade = ResourceFacade.from_runtime(runtime, store=store, index=index, drain=drain)

    assert facade.store is store
    assert facade.index is index
    assert facade.drain is drain
    assert facade.graph is runtime.graph
    assert facade.claims is runtime.backend.claim_query
    assert facade.snapshot is runtime.backend.snapshot
    # The backend hands out a journal view per access; same kind, same backend.
    assert type(facade.journal) is type(runtime.backend.journal)


def test_an_import_through_the_composed_facade_lands_in_the_runtime(tmp_path) -> None:
    store, index = InMemoryResourceStore(), _Index()
    runtime = _runtime(store, index)
    facade = ResourceFacade.from_runtime(runtime, store=store, index=index)
    directory = write_import_directory(
        tmp_path / "in",
        [
            {
                "slug": "body",
                "title": "Body",
                "summary": "what this section covers",
                "ordinal": 0,
                "content_hash": "body-v1",
                "chunks": [{"label": "opening", "text": "alpha"}],
            }
        ],
        source_ref="file:///q3.pdf",
        source_kind="pdf",
    )

    result = facade.import_dir(pot_id="p", slug="q3-review", source_dir=directory)

    assert result.graph_written is True
    assert index.indexed == ["q3-review"]
    rows = runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("SECTION_OF",))
    )
    assert [row.subject_key for row in rows] == ["docsection:q3-review:body"]


@pytest.mark.parametrize("protocols", [False, True])
def test_protocol_source_protection_runs_only_with_the_protocol_extension(
    tmp_path, monkeypatch, protocols
) -> None:
    from potpie_context_engine.application.services import protocol_resources
    from potpie_context_engine.protocols import protocols_definition

    calls: list[str] = []
    monkeypatch.setattr(
        protocol_resources,
        "protect_protocol_source",
        lambda *_a, **kwargs: calls.append(kwargs["slug"]),
    )
    store = InMemoryResourceStore()
    runtime = build_graph_runtime(
        build_test_backend(),
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        protocols_definition() if protocols else DEFAULT_GRAPH_DEFINITION,
        resource_store=store,
    )
    facade = ResourceFacade.from_runtime(runtime, store=store)
    directory = write_import_directory(
        tmp_path / "in",
        [
            {
                "slug": "body",
                "title": "Body",
                "summary": "what this section covers",
                "ordinal": 0,
                "content_hash": "body-v1",
                "chunks": [{"label": "opening", "text": "alpha"}],
            }
        ],
        source_ref="file:///q3.pdf",
        source_kind="pdf",
    )

    facade.import_dir(pot_id="p", slug="q3-review", source_dir=directory)
    facade.delete(pot_id="p", slug="q3-review")

    assert calls == (["q3-review", "q3-review"] if protocols else [])
