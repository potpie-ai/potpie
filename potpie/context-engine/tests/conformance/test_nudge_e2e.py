"""Step 12a / Step 13 acceptance: the nudge loop end-to-end on the real stack.

Exercises the brain over a real ``DefaultGraphService`` + the bundled local
embedder (no external embedder, no model calls), covering plan acceptance
scenarios #11 (inject on match / silent on none / no dup per session), #12
(an agent-authored description is embedded on write and a *paraphrased* query
retrieves it), and #13 (the trigger + retrieval loop needs no API key).
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    HashingEmbedder,
)
from potpie_context_engine.adapters.outbound.session.injection_ledger import (
    InMemoryInjectionLedger,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.nudge_service import NudgeService
from potpie_context_engine.domain.nudge import GraphNudgeRequest
from potpie_context_engine.domain.reconciliation_flags import agent_planner_enabled
from potpie_context_engine.domain.semantic_mutations import SemanticMutationRequest

pytestmark = pytest.mark.unit

POT = "local/default"


def _stack() -> tuple[NudgeService, DefaultGraphService]:
    backend = InMemoryGraphBackend(embedder=HashingEmbedder())
    svc = DefaultGraphService(backend=backend)
    nudge = NudgeService(graph=svc, ledger=InMemoryInjectionLedger())
    return nudge, svc


def _seed_bug(svc: DefaultGraphService) -> None:
    svc.mutate(
        SemanticMutationRequest.parse(
            {
                "pot_id": POT,
                "operations": [
                    {
                        "op": "assert_claim",
                        "subgraph": "debugging",
                        "subject": {
                            "key": "bug_pattern:settle-deadlock",
                            "type": "BugPattern",
                        },
                        "predicate": "REPRODUCES",
                        "object": {"key": "service:payments-api", "type": "Service"},
                        "truth": "agent_claim",
                        "description": (
                            "payment deadlock on concurrent settle; lock-ordering "
                            "race when two settlements run at once under load"
                        ),
                    }
                ],
            }
        )
    )


def test_paraphrased_query_retrieves_embedded_card_with_no_external_embedder() -> None:
    # #12: description embedded by the local model on write; a paraphrase recalls it.
    nudge, svc = _stack()
    _seed_bug(svc)
    res = nudge.nudge(
        GraphNudgeRequest(
            pot_id=POT,
            event="test_failed",
            session_id="sess-1",
            query="deadlock when settling payments concurrently",  # paraphrase
        )
    )
    assert res.ok and not res.silent
    assert "claim:" in (res.injected_keys[0] if res.injected_keys else "")
    assert "deadlock" in (res.inject_context or "")


def test_no_duplicate_injection_within_session() -> None:
    # #11: the per-session ledger blocks a second injection of the same claim.
    nudge, svc = _stack()
    _seed_bug(svc)
    first = nudge.nudge(
        GraphNudgeRequest(
            pot_id=POT,
            event="test_failed",
            session_id="sess-dup",
            query="settle deadlock",
        )
    )
    assert first.injected_keys
    again = nudge.nudge(
        GraphNudgeRequest(
            pot_id=POT,
            event="test_failed",
            session_id="sess-dup",
            query="settle deadlock",
        )
    )
    assert again.silent  # everything already injected this session
    # A different session still gets it.
    other = nudge.nudge(
        GraphNudgeRequest(
            pot_id=POT,
            event="test_failed",
            session_id="sess-other",
            query="settle deadlock",
        )
    )
    assert other.injected_keys == first.injected_keys


def test_silent_when_graph_is_empty() -> None:
    # #11: nothing to inject → silent, not an empty injection.
    nudge, _ = _stack()
    res = nudge.nudge(
        GraphNudgeRequest(pot_id=POT, event="pre_edit", session_id="s", path="src/x.py")
    )
    assert res.ok and res.silent


def test_loop_runs_without_any_api_key(monkeypatch) -> None:
    # #13: the trigger + retrieval loop is free — local embedder only, planner
    # parked off, no API key required.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert agent_planner_enabled() is False  # LLM reconciliation parked
    embedder = HashingEmbedder()
    assert embedder is not None
    assert embedder.name == "local-hashing-v1"  # bundled, dependency-free

    nudge, svc = _stack()
    _seed_bug(svc)
    res = nudge.nudge(
        GraphNudgeRequest(
            pot_id=POT, event="test_failed", session_id="s", query="settle deadlock"
        )
    )
    assert res.ok  # retrieval succeeded with no API client anywhere on the path


def test_instruction_events_prompt_a_write_capture() -> None:
    # The compounding engine: red→green and end-of-task prompt a capture.
    nudge, _ = _stack()
    for event in ("test_passed", "stop"):
        res = nudge.nudge(GraphNudgeRequest(pot_id=POT, event=event, session_id="s"))
        assert res.ok and not res.silent and res.instruction
