"""``AgentContextService.resolve`` fills an unset intent from the task.

The CLI used to send ``intent="feature"`` whenever the caller said nothing,
so the service never saw an unset intent and never read the task. Now an
unset intent is inferred here — on the agent-facing door, not inside
``GraphService`` — and the envelope says which happened (``intent_source``).
"""

from __future__ import annotations

from potpie_context_core.agent_envelope import AgentEnvelope
from potpie_context_core.ports.agent_context import ResolveRequest
from potpie_context_engine.application.services.agent_context import (
    AgentContextService,
)


class _Graph:
    """Records the request the service forwarded; echoes intent + metadata."""

    def __init__(self) -> None:
        self.requests: list[ResolveRequest] = []

    def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        self.requests.append(request)
        return AgentEnvelope(
            pot_id=request.pot_id,
            intent=request.intent or "unknown",
            items=(),
            coverage=(),
            metadata=dict(request.metadata),
        )


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
