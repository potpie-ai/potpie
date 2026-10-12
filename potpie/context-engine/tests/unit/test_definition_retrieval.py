"""Acronym evaluation through the agent read door, real readers and ranker."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from potpie_context_engine.core.agent_context_port import (
    infer_context_intent,
    normalize_context_values,
)
from potpie_context_engine.core.definition_query import definition_subject
from potpie_context_engine.core.ports.agent_context import ResolveRequest, SearchRequest
from potpie_context_engine.core.ports.claim_query import ClaimRow
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    HashingEmbedder,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.envelope_builder import (
    include_rank_weight,
)
from potpie_context_engine.domain.definition_retrieval import has_definition


class _AgentDoor:
    """The agent-facing read door over ``DefaultGraphService``.

    The product layer owns this door. It fills an unset intent from the text and
    widens a ``docs`` filter to passage evidence, then forwards to the graph
    service. The engine's definition ranking is exercised the same way here.
    """

    def __init__(self, graph: DefaultGraphService) -> None:
        self.graph = graph

    def resolve(self, request: ResolveRequest):
        explicit = (request.intent or "").strip()
        intent = explicit or infer_context_intent(request.task)
        return self.graph.resolve(_stamped(request, intent, explicit))

    def search(self, request: SearchRequest):
        explicit = (request.intent or "").strip()
        intent = explicit or (
            "definition" if definition_subject(request.query) else "unknown"
        )
        return self.graph.search(_stamped(request, intent, explicit))


def _stamped(request, intent: str, explicit: str):
    include = tuple(normalize_context_values(request.include))
    if "docs" in include and "resources" not in include:
        include = (*include, "resources")
    return replace(
        request,
        intent=intent,
        include=include,
        metadata={
            **dict(request.metadata),
            "intent_source": "explicit" if explicit else "inferred",
        },
    )


FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/retrieval/definitions.json").read_text()
)


def definition_service(vector=False, resource_index=None):
    backend = InMemoryGraphBackend(embedder=HashingEmbedder() if vector else None)
    store = backend.claim_query
    for claim in FIXTURE["claims"]:
        subject = f"docsection:architecture:{claim['key']}"
        store.set_entity_label(
            pot_id="p", entity_key=subject, labels=("DocumentSection",)
        )
        store.add(
            ClaimRow(
                pot_id="p",
                claim_key=claim["key"],
                predicate="DOCUMENTS",
                subject_key=subject,
                object_key="feature:qme-service-software",
                fact=claim["fact"],
                truth=claim["truth"],
                evidence_strength=claim["strength"],
                source_refs=(claim["ref"],),
                subgraph="knowledge",
            )
        )
    store.add(
        ClaimRow(
            pot_id="p",
            claim_key="topology",
            predicate="DEPENDS_ON",
            subject_key="service:qme",
            object_key="service:redis",
            fact="QME full form deployment uses Redis. QME service depends on Redis.",
            evidence_strength="deterministic",
            properties={"corroboration_count": 20},
        )
    )
    graph = DefaultGraphService(backend=backend, resource_index=resource_index)
    return _AgentDoor(graph), store


@pytest.mark.parametrize("query", FIXTURE["queries"])
@pytest.mark.parametrize("include", [(), ("docs",)])
@pytest.mark.parametrize("vector", [False, True])
def test_definition_ranks_first_with_sources_across_both_agent_paths(
    query, include, vector
):
    service, _ = definition_service(vector)
    resolved = service.resolve(
        ResolveRequest(pot_id="p", task=query, include=include, max_items=1)
    )
    searched = service.search(
        SearchRequest(pot_id="p", query=query, include=include, max_items=1)
    )
    for env in (resolved, searched):
        assert env.intent == "definition"
        assert env.metadata["intent_source"] == "inferred"
        first = env.items[0]
        assert first.candidate_key == "correct", env.to_dict()
        assert first.payload["chunk_ids"] == ["potpie://res/architecture/definition/0"]
        assert first.breakdown["definition_match"] == 1
        assert first.breakdown["definition_authority"] == 1
        assert first.score == pytest.approx(
            first.breakdown["reader_score"] * first.breakdown["include_weight"]
        )


def test_wrong_expansions_do_not_gain_authority_or_corroboration():
    service, _ = definition_service()
    env = service.search(
        SearchRequest(pot_id="p", query="QME full form", include=("docs",))
    )
    items = {item.candidate_key: item for item in env.items}
    wrong = items["incorrect"]
    assert wrong.payload["truth"] == "agent_claim"
    assert wrong.payload["evidence_strength"] == "inferred"
    assert wrong.breakdown["definition_authority"] == 0
    assert wrong.breakdown["corroboration"] == 0.5
    assert wrong.score < items["correct"].score
    assert env.metadata["readers"]["docs"]["warnings"]
    for key in ("negated", "question", "other_acronym", "distractor"):
        if key in items:
            assert items[key].breakdown["definition_match"] == 0


def test_authority_requires_a_reference_and_does_not_flow_from_another_claim():
    service, store = definition_service()
    store.rows = [
        replace(row, source_refs=()) if row.claim_key == "correct" else row
        for row in store.rows
    ]
    env = service.search(
        SearchRequest(pot_id="p", query="QME full form", include=("docs",))
    )
    correct = next(item for item in env.items if item.candidate_key == "correct")
    assert correct.breakdown["definition_authority"] == 0
    assert correct.payload["source_refs"] == []


def test_explicit_intents_and_ordinary_search_keep_existing_ranking():
    service, _ = definition_service()
    for intent in ("feature", "unknown", "docs"):
        env = service.search(
            SearchRequest(
                pot_id="p", query="QME full form", intent=intent, include=("docs",)
            )
        )
        assert env.intent == intent
        assert env.metadata["intent_source"] == "explicit"
        assert all("definition_match" not in item.breakdown for item in env.items)
        expected_weight = 1.0 if intent == "docs" else 0.65
        assert all(
            item.breakdown["include_weight"] == expected_weight for item in env.items
        )
    env = service.search(SearchRequest(pot_id="p", query="Redis deployment"))
    assert env.intent == "unknown"
    assert include_rank_weight("docs") == 0.65
    assert include_rank_weight("resources") == 0.6


@pytest.mark.parametrize(
    "text",
    [
        "QME does not stand for Quality Metrics Engine.",
        "QME stands for Quality Metrics Engine, which is incorrect.",
        "Does QME stand for Quality Metrics Engine?",
        "Perhaps QME stands for Quality Metrics Engine.",
        "XQME stands for Quality Metrics Engine.",
        "ERP stands for Enterprise Resource Planning. QME uses ERP.",
    ],
)
def test_negative_and_other_term_wording_gets_no_definition_signal(text):
    assert not has_definition(text, "QME")


@pytest.mark.parametrize(
    "query",
    [
        "full form validation",
        "add abbreviation support",
        "stands forever",
        "why is QME failing",
    ],
)
def test_non_definition_tasks_are_not_inferred(query):
    assert definition_subject(query) is None


@pytest.mark.parametrize(
    "fact",
    [
        "QME (Quartz/Module/Exchange) is the service software.",
        "We use Quartz Module Exchange (QME).",
        "QME stands for Quartz Module Exchange.",
        "The full form of QME is Quartz Module Exchange.",
    ],
)
def test_source_definition_forms_survive_recall_and_single_item_limit(fact):
    service, store = definition_service()
    store.rows = [
        replace(row, fact=fact) if row.claim_key == "correct" else row
        for row in store.rows
    ]
    env = service.search(SearchRequest(pot_id="p", query="QME full form", max_items=1))
    assert env.items[0].candidate_key == "correct"
    assert env.items[0].payload["chunk_ids"]


def test_explicit_definition_intent_accepts_a_bare_term():
    service, _ = definition_service()
    env = service.search(
        SearchRequest(pot_id="p", query="QME", intent="definition", max_items=1)
    )
    assert env.items[0].candidate_key == "correct"
    assert env.items[0].breakdown["definition_match"] == 1
    assert definition_subject("QME") is None  # a bare search keeps its old routing


def test_competing_authoritative_expansions_remain_separate_evidence():
    service, store = definition_service()
    store.rows = [
        replace(row, truth="authoritative_fact", evidence_strength="deterministic")
        if row.claim_key == "incorrect"
        else row
        for row in store.rows
    ]
    env = service.search(
        SearchRequest(pot_id="p", query="QME full form", include=("docs",))
    )
    competing = [
        item for item in env.items if item.candidate_key in {"correct", "incorrect"}
    ]
    assert len(competing) == 2
    assert all(item.breakdown["corroboration"] == 0.5 for item in competing)
    assert competing[0].payload["source_refs"] != competing[1].payload["source_refs"]
    assert "competing expansions" in env.metadata["readers"]["docs"]["warnings"][0]


@pytest.mark.parametrize("profile", ["sqlite_fts", "sqlite_hybrid"])
@pytest.mark.parametrize("query", ["QME full form", "What does QME stand for?"])
def test_real_passage_index_participates_without_overriding_authoritative_definition(
    tmp_path, profile, query
):
    from potpie_context_engine.adapters.outbound.resources.index import (
        build_resource_index,
    )
    from potpie_context_engine.core.ports.resource_store import (
        Chunk,
        ChunkRef,
        DocumentManifest,
        SectionManifest,
    )

    index = build_resource_index(profile, home=tmp_path)
    sections, chunks = [], []
    for claim in FIXTURE["claims"]:
        section = claim["key"].replace("_", "-")
        sections.append(
            SectionManifest(
                slug=section,
                title=section,
                summary=claim["fact"],
                ordinal=len(sections),
                content_hash=section,
                chunks=(ChunkRef(seq=0, label=section),),
            )
        )
        chunks.append(
            Chunk(
                resource_id=f"potpie://res/architecture/{section}/0000",
                doc="architecture",
                section=section,
                seq=0,
                text=claim["fact"],
                chars=len(claim["fact"]),
                revision=1,
                source_ref=f"file:///architecture/{section}.md",
            )
        )
    index.index_document(
        pot_id="p",
        manifest=DocumentManifest(
            pot_id="p",
            doc="architecture",
            revision=1,
            source_ref="file:///architecture.md",
            source_kind="file",
            sections=tuple(sections),
        ),
        chunks=tuple(chunks),
    )
    index.drain()
    try:
        service, _ = definition_service(resource_index=index)
        for env in (
            service.search(SearchRequest(pot_id="p", query=query, max_items=12)),
            service.resolve(ResolveRequest(pot_id="p", task=query, max_items=12)),
        ):
            assert env.items[0].candidate_key == "correct", env.to_dict()
            passages = [item for item in env.items if item.include == "resources"]
            assert passages
            assert env.metadata["readers"]["resources"]["match_mode"] != "disabled"
            assert all(item.breakdown["definition_authority"] == 0 for item in passages)
            assert all(
                item.payload["resource_id"] in item.payload["fetch"]
                for item in passages
            )
        env = service.search(
            SearchRequest(pot_id="p", query=query, include=("resources",))
        )
        affirmative = next(
            item for item in env.items if "/correct/" in item.candidate_key
        )
        distractor = next(
            item for item in env.items if "/distractor/" in item.candidate_key
        )
        assert affirmative.score > distractor.score
        assert affirmative.breakdown["definition_match"] == 1
        assert distractor.breakdown["definition_match"] == 0
    finally:
        index.close()
