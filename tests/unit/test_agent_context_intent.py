"""The agent door fills an unset intent and says which one it used.

``AgentContextService.resolve`` infers an unset intent from the task text, and
``search`` asks for the definition intent when the query names an acronym. Both
stamp ``intent_source`` (``explicit`` or ``inferred``) into the envelope
metadata. The heuristic lives on this door, not inside ``GraphService``, so the
graph service's internal callers keep the families they ask for.
"""

from __future__ import annotations

import pytest

from potpie.agent_context import AgentContextService
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.core.agent_envelope import AgentEnvelope
from potpie_context_engine.core.ports.agent_context import (
    ResolveRequest,
    SearchRequest,
)
from potpie_context_engine.core.ports.claim_query import ClaimRow

pytestmark = pytest.mark.unit


class _Graph:
    """Records the request the service forwarded; echoes intent + metadata."""

    def __init__(self) -> None:
        self.requests: list[object] = []

    def _answer(self, request) -> AgentEnvelope:
        self.requests.append(request)
        return AgentEnvelope(
            pot_id=request.pot_id,
            intent=request.intent or "unknown",
            items=(),
            coverage=(),
            metadata=dict(request.metadata),
        )

    def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        return self._answer(request)

    def search(self, request: SearchRequest) -> AgentEnvelope:
        return self._answer(request)


def _service() -> tuple[AgentContextService, _Graph]:
    graph = _Graph()
    return AgentContextService(graph=graph, pots=object(), skills=object()), graph


def test_an_unset_intent_is_inferred_from_the_task() -> None:
    service, graph = _service()

    env = service.resolve(ResolveRequest(pot_id="p", task="why is stock stale"))

    assert graph.requests[0].intent == "debugging"
    assert env.intent == "debugging"
    assert env.metadata["intent_source"] == "inferred"


def test_an_explicit_intent_wins_over_the_task_text() -> None:
    service, graph = _service()

    env = service.resolve(
        ResolveRequest(pot_id="p", task="why is stock stale", intent="feature")
    )

    assert graph.requests[0].intent == "feature"
    assert env.metadata["intent_source"] == "explicit"


def test_a_blank_intent_counts_as_unset() -> None:
    service, graph = _service()

    service.resolve(ResolveRequest(pot_id="p", task="why is stock stale", intent="  "))

    assert graph.requests[0].intent == "debugging"


def test_no_task_and_no_intent_stays_broad() -> None:
    """A resolve that names includes and no task has nothing to infer from;
    it keeps the widest family set instead of a guess."""
    service, graph = _service()

    env = service.resolve(ResolveRequest(pot_id="p", include=("raw_graph",)))

    assert graph.requests[0].intent == "unknown"
    assert env.metadata["intent_source"] == "inferred"


def test_caller_metadata_survives_the_stamp() -> None:
    service, graph = _service()

    service.resolve(
        ResolveRequest(pot_id="p", task="add coupons", metadata={"mode": "deep"})
    )

    assert graph.requests[0].metadata == {"mode": "deep", "intent_source": "inferred"}


@pytest.mark.parametrize(
    "query", ["What does QME stand for?", "QME full form", "abbreviation of QME"]
)
def test_an_acronym_search_asks_for_the_definition_intent(query: str) -> None:
    service, graph = _service()

    env = service.search(SearchRequest(pot_id="p", query=query))

    assert graph.requests[0].intent == "definition"
    assert env.metadata["intent_source"] == "inferred"


def test_an_ordinary_search_keeps_the_broad_intent() -> None:
    service, graph = _service()

    env = service.search(SearchRequest(pot_id="p", query="Redis deployment"))

    assert graph.requests[0].intent == "unknown"
    assert env.metadata["intent_source"] == "inferred"


def test_an_explicit_search_intent_passes_through() -> None:
    service, graph = _service()

    env = service.search(
        SearchRequest(pot_id="p", query="QME full form", intent="docs")
    )

    assert graph.requests[0].intent == "docs"
    assert env.metadata["intent_source"] == "explicit"


# --- the real door over the real graph service ----------------------------------


def _definition_door() -> AgentContextService:
    backend = InMemoryGraphBackend()
    store = backend.claim_query
    claims = (
        (
            "correct",
            "QME stands for Quartz Module Exchange.",
            "authoritative_fact",
            "attested",
            "potpie://res/architecture/definition/0",
        ),
        (
            "incorrect",
            "QME stands for Quality Metrics Engine.",
            "agent_claim",
            "inferred",
            "agent:guess",
        ),
        (
            "distractor",
            "QME full form is entered in the deployment configuration.",
            "authoritative_fact",
            "attested",
            "potpie://res/operations/configuration/0",
        ),
    )
    for key, fact, truth, strength, ref in claims:
        subject = f"docsection:architecture:{key}"
        store.set_entity_label(
            pot_id="p", entity_key=subject, labels=("DocumentSection",)
        )
        store.add(
            ClaimRow(
                pot_id="p",
                claim_key=key,
                predicate="DOCUMENTS",
                subject_key=subject,
                object_key="feature:qme-service-software",
                fact=fact,
                truth=truth,
                evidence_strength=strength,
                source_refs=(ref,),
                subgraph="knowledge",
            )
        )
    graph = DefaultGraphService(backend=backend)
    return AgentContextService(graph=graph, pots=object(), skills=object())


@pytest.mark.parametrize("query", ["QME full form", "What does QME stand for?"])
def test_the_attested_definition_ranks_first_through_both_doors(query: str) -> None:
    door = _definition_door()

    resolved = door.resolve(ResolveRequest(pot_id="p", task=query, max_items=1))
    searched = door.search(SearchRequest(pot_id="p", query=query, max_items=1))

    for env in (resolved, searched):
        assert env.intent == "definition"
        assert env.metadata["intent_source"] == "inferred"
        assert env.items[0].candidate_key == "correct", env.to_dict()
