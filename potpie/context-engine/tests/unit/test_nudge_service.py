"""Step 12a: the nudge brain — event→action policy, dedup, budget, formatting.

The read-trunk relevance (ranking/scope/embeddings) is covered by R3/R4/R7;
here we isolate the brain's own responsibilities with a fake reader so the
policy logic is tested deterministically.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.session.injection_ledger import InMemoryInjectionLedger
from potpie_context_engine.application.services.nudge_service import NudgeService
from potpie_context_engine.domain.nudge import (
    NUDGE_POLICIES,
    GraphNudgeRequest,
    canonical_nudge_event,
    is_nudge_event,
)
from potpie_context_core.ports.graph_service import GraphReadResult

pytestmark = pytest.mark.unit

POT = "local/default"


class _FakeReader:
    """Returns canned envelopes per view and records the requests it saw."""

    def __init__(self, by_view: dict[str, list[dict]]):
        self.by_view = by_view
        self.requests: list = []

    def read(self, request) -> GraphReadResult:
        self.requests.append(request)
        view_name = f"{request.subgraph}.{request.view}"
        items = tuple(self.by_view.get(view_name, ()))
        return GraphReadResult(
            graph_contract_version="v1.5",
            ontology_version="2026-06-graph",
            view=view_name,
            subgraph=request.subgraph,
            items=items,
        )


def _item(view_include: str, key: str, score: float, **payload) -> dict:
    del view_include
    return {
        "entity_key": key,
        "score": score,
        "claim": {"claim_key": key},
        **payload,
    }


def _svc(
    by_view=None, ledger=None
) -> tuple[NudgeService, _FakeReader, InMemoryInjectionLedger]:
    reader = _FakeReader(by_view or {})
    led = ledger or InMemoryInjectionLedger()
    return NudgeService(graph=reader, ledger=led), reader, led


# --- event routing ----------------------------------------------------------


def test_unknown_event_is_silent_and_not_ok() -> None:
    svc, reader, _ = _svc()
    res = svc.nudge(GraphNudgeRequest(pot_id=POT, event="explode", session_id="s1"))
    assert res.silent and not res.ok
    assert "unknown nudge event" in (res.detail or "")
    assert reader.requests == []  # never touched the graph


def test_nudge_event_accepts_dash_aliases() -> None:
    assert canonical_nudge_event("pre-edit") == "pre_edit"
    assert canonical_nudge_event("test-passed") == "test_passed"
    assert is_nudge_event("pre-edit")
    assert not is_nudge_event("preflight-edit")


def test_dash_alias_routes_to_canonical_policy() -> None:
    by_view = {
        "decisions.preferences_for_scope": [
            _item(
                "coding_preferences", "claim:pref:retry", 0.4, description="retry pref"
            ),
        ],
        "debugging.prior_occurrences": [],
    }
    svc, reader, _ = _svc(by_view)
    res = svc.nudge(
        GraphNudgeRequest(
            pot_id=POT,
            event="pre-edit",
            session_id="s1",
            path="src/payments/client.py",
        )
    )
    assert res.ok and not res.silent
    assert res.event == "pre_edit"
    assert {f"{req.subgraph}.{req.view}" for req in reader.requests} == {
        "decisions.preferences_for_scope",
        "debugging.prior_occurrences",
    }


@pytest.mark.parametrize("event", ["test_passed", "stop"])
def test_instruction_events_return_directive_without_reading(event: str) -> None:
    svc, reader, _ = _svc()
    res = svc.nudge(GraphNudgeRequest(pot_id=POT, event=event, session_id="s1"))
    assert res.ok and not res.silent
    assert res.instruction and "graph mutate" in res.instruction
    assert res.inject_context is None
    assert reader.requests == []  # instruction direction never reads


# --- data direction ---------------------------------------------------------


def test_pre_edit_injects_ranked_deduped_budgeted() -> None:
    by_view = {
        "decisions.preferences_for_scope": [
            _item(
                "coding_preferences",
                "claim:pref:retry",
                0.4,
                description="wrap calls in tenacity retry",
                source_refs=["repo:prefs"],
            ),
        ],
        "debugging.prior_occurrences": [
            _item(
                "prior_bugs",
                "claim:bug:deadlock",
                0.9,
                fact="payment deadlock on concurrent settle",
            ),
            _item("prior_bugs", "claim:bug:timeout", 0.2, fact="timeout under load"),
        ],
    }
    svc, reader, _ = _svc(by_view)
    res = svc.nudge(
        GraphNudgeRequest(
            pot_id=POT,
            event="pre_edit",
            session_id="s1",
            path="src/payments/client.py",
            limit=2,
        )
    )
    assert res.ok and not res.silent
    # Budgeted to 2, globally ranked by score (bug 0.9 first, pref 0.4 next).
    assert res.injected_keys == ("claim:bug:deadlock", "claim:pref:retry")
    assert "payment deadlock on concurrent settle" in res.inject_context
    assert "wrap calls in tenacity retry" in res.inject_context
    assert set(res.views_read) == {
        "decisions.preferences_for_scope",
        "debugging.prior_occurrences",
    }


def test_path_is_mapped_to_file_path_filter_for_scoped_views() -> None:
    svc, reader, _ = _svc(
        {"decisions.preferences_for_scope": [], "debugging.prior_occurrences": []}
    )
    svc.nudge(
        GraphNudgeRequest(
            pot_id=POT, event="pre_edit", session_id="s1", path="src/a.py"
        )
    )
    for req in reader.requests:
        assert req.scope.get("file_path") == "src/a.py"


def test_empty_graph_is_silent() -> None:
    svc, _, _ = _svc({})
    res = svc.nudge(
        GraphNudgeRequest(
            pot_id=POT, event="pre_edit", session_id="s1", path="src/a.py"
        )
    )
    assert res.ok and res.silent
    assert res.inject_context is None


def test_session_dedup_never_injects_same_key_twice() -> None:
    by_view = {
        "decisions.preferences_for_scope": [
            _item(
                "coding_preferences", "claim:pref:retry", 0.5, description="retry pref"
            ),
        ],
        "debugging.prior_occurrences": [],
    }
    svc, _, led = _svc(by_view)
    first = svc.nudge(
        GraphNudgeRequest(
            pot_id=POT, event="pre_edit", session_id="sess-A", path="src/a.py"
        )
    )
    assert "claim:pref:retry" in first.injected_keys
    assert led.was_injected("sess-A", "claim:pref:retry")
    # Same session → already injected → silent.
    second = svc.nudge(
        GraphNudgeRequest(
            pot_id=POT, event="pre_edit", session_id="sess-A", path="src/a.py"
        )
    )
    assert second.silent
    # Different session → fresh injection.
    other = svc.nudge(
        GraphNudgeRequest(
            pot_id=POT, event="pre_edit", session_id="sess-B", path="src/a.py"
        )
    )
    assert "claim:pref:retry" in other.injected_keys


def test_min_score_threshold_filters(monkeypatch) -> None:
    import dataclasses

    from potpie_context_engine.domain import nudge as nudge_mod

    strict = dataclasses.replace(NUDGE_POLICIES["pre_edit"], min_score=0.8)
    monkeypatch.setitem(nudge_mod.NUDGE_POLICIES, "pre_edit", strict)
    by_view = {
        "decisions.preferences_for_scope": [
            _item("coding_preferences", "claim:weak", 0.3, fact="weak")
        ],
        "debugging.prior_occurrences": [
            _item("prior_bugs", "claim:strong", 0.95, fact="strong")
        ],
    }
    svc, _, _ = _svc(by_view)
    res = svc.nudge(
        GraphNudgeRequest(pot_id=POT, event="pre_edit", session_id="s1", path="x")
    )
    assert res.injected_keys == ("claim:strong",)  # 0.3 below threshold dropped


def test_inject_context_carries_scope_and_source() -> None:
    by_view = {
        "infra_topology.service_neighborhood": [
            _item(
                "infra_topology",
                "claim:dep",
                0.7,
                fact="payments depends on ledger",
                environment="prod",
                source_refs=["repo:manifest:svc.yaml"],
            )
        ]
    }
    svc, _, _ = _svc(by_view)
    res = svc.nudge(
        GraphNudgeRequest(
            pot_id=POT,
            event="pre_deploy",
            session_id="s1",
            scope={"service": "payments-api"},
        )
    )
    assert "environment=prod" in res.inject_context
    assert "src=repo:manifest:svc.yaml" in res.inject_context
    assert "[infra_topology.service_neighborhood]" in res.inject_context
