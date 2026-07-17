"""The nudge trigger model — *what* to inject, *when*, and *whether to prompt a
write* (Graph V1.5 Step 12a / Trigger Model).

A harness hook is a dumb three-line adapter: it forwards a harness event + path
to ``potpie graph nudge`` and injects whatever comes back. All the policy — which
view to read for which event, the relevance gate, and when to emit a
write-instruction instead of data — lives here, deterministically, so the hook
never reasons and never calls a model.

Two directions (see the plan's event→nudge map):

- **data**: run a token-budgeted ``graph read`` over the event's views and inject
  ranked, deduped context (the agent gets context whether or not it asked).
- **instruction**: inject a short "consider recording X" directive on a strong
  signal (a red→green test, end of task); the in-session agent decides truth
  class, resolves identity, and calls ``graph mutate`` — never an auto-write.

This module owns only declarative policy + DTOs (no IO, no model). The
:class:`~potpie_context_engine.application.services.nudge_service.NudgeService` executes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from potpie_context_engine.domain.graph_contract import GRAPH_CONTRACT_VERSION, ONTOLOGY_VERSION


class NudgeEvent(StrEnum):
    """The harness events V1.5 carries (the four UC1–UC4 triggers + bookends)."""

    session_start = "session_start"
    pre_edit = "pre_edit"  # PreToolUse(Write|Edit)
    pre_deploy = "pre_deploy"  # PreToolUse(Bash: deploy/infra)
    test_failed = "test_failed"  # PostToolUse(Bash test/run fails)
    test_passed = "test_passed"  # PostToolUse(test red→green)
    stop = "stop"  # end of task


NUDGE_EVENTS: tuple[str, ...] = tuple(e.value for e in NudgeEvent)


def canonical_nudge_event(value: str | None) -> str | None:
    if value is None:
        return None
    event = str(value).strip().replace("-", "_")
    return event if event in NUDGE_EVENTS else None


def is_nudge_event(value: str | None) -> bool:
    return canonical_nudge_event(value) is not None


NUDGE_EVENT_HELP = (
    "session_start|pre_edit|pre_deploy|test_failed|test_passed|stop "
    "(dash aliases accepted, e.g. pre-edit)"
)


class NudgeDirection(StrEnum):
    data = "data"  # inject ranked read results
    instruction = "instruction"  # inject a write-prompt directive


@dataclass(frozen=True, slots=True)
class NudgeViewSpec:
    """One view the policy reads, and how the event's inputs map onto it."""

    view: str
    pass_query: bool = False
    pass_scope: bool = True
    limit: int = 8


@dataclass(frozen=True, slots=True)
class NudgePolicy:
    """The deterministic action for one event."""

    event: str
    direction: NudgeDirection
    views: tuple[NudgeViewSpec, ...] = ()
    instruction: str | None = None
    # Soft relevance gate: a read item must score at/above this to be eligible.
    # 0.0 defers to the read trunk's own ranking + coverage; the per-session
    # dedup ledger is the primary noise control.
    min_score: float = 0.0
    # SessionStart additionally wants a deterministic source ingest; the hook
    # runs `ingest ... --since` separately, so the nudge itself only reads.
    triggers_ingest: bool = False


# The event → action map (the plan's "Event → nudge" table). Views reference
# names in ``potpie_context_engine.domain.graph_views.GRAPH_VIEWS``.
NUDGE_POLICIES: dict[str, NudgePolicy] = {
    NudgeEvent.session_start.value: NudgePolicy(
        event=NudgeEvent.session_start.value,
        direction=NudgeDirection.data,
        views=(
            NudgeViewSpec(
                "decisions.active_decisions", pass_query=False, pass_scope=True
            ),
            NudgeViewSpec(
                "decisions.preferences_for_scope", pass_query=False, pass_scope=True
            ),
        ),
        triggers_ingest=True,
    ),
    NudgeEvent.pre_edit.value: NudgePolicy(
        event=NudgeEvent.pre_edit.value,
        direction=NudgeDirection.data,
        views=(
            NudgeViewSpec(
                "decisions.preferences_for_scope", pass_query=False, pass_scope=True
            ),
            NudgeViewSpec(
                "debugging.prior_occurrences", pass_query=True, pass_scope=True
            ),
        ),
    ),
    NudgeEvent.pre_deploy.value: NudgePolicy(
        event=NudgeEvent.pre_deploy.value,
        direction=NudgeDirection.data,
        views=(
            NudgeViewSpec(
                "infra_topology.service_neighborhood", pass_query=False, pass_scope=True
            ),
        ),
    ),
    NudgeEvent.test_failed.value: NudgePolicy(
        event=NudgeEvent.test_failed.value,
        direction=NudgeDirection.data,
        views=(
            NudgeViewSpec(
                "debugging.prior_occurrences", pass_query=True, pass_scope=True
            ),
            NudgeViewSpec("recent_changes.timeline", pass_query=False, pass_scope=True),
        ),
    ),
    NudgeEvent.test_passed.value: NudgePolicy(
        event=NudgeEvent.test_passed.value,
        direction=NudgeDirection.instruction,
        instruction=(
            "You just turned a failing test green. If the bug + fix is non-obvious, "
            "capture it so a future searcher recalls it: call `potpie graph mutate` "
            "with an assert_claim REPRODUCES (the bug pattern) and a RESOLVED (the "
            "fix), each with a retrieval-grade description — symptoms, synonyms, and "
            "scope a future query would use, not display text."
        ),
    ),
    NudgeEvent.stop.value: NudgePolicy(
        event=NudgeEvent.stop.value,
        direction=NudgeDirection.instruction,
        instruction=(
            "End of task. Capture durable learnings as graph claims via "
            "`potpie graph mutate`: new preferences (POLICY_APPLIES_TO), decisions "
            "(DECIDED), and fixes (RESOLVED). Decide the truth class, resolve entity "
            "identity with `graph search-entities` first, and write descriptions for "
            "retrieval, not display."
        ),
    ),
}


def _check_nudge_policies_coherent() -> None:
    """Import-time guard: the hand-maintained policy table must cover every
    nudge event exactly once, with each key matching its policy's ``event``.

    Mirrors :func:`potpie_context_engine.domain.graph_views._check_views_coherent` — a missing or
    mismatched key silently disables the nudge for that event.
    """
    errors: list[str] = []
    expected = set(NUDGE_EVENTS)
    actual = set(NUDGE_POLICIES)
    missing = expected - actual
    extra = actual - expected
    if missing:
        errors.append(f"missing nudge policy for events: {sorted(missing)!r}")
    if extra:
        errors.append(f"unknown nudge policy events: {sorted(extra)!r}")
    for key, policy in NUDGE_POLICIES.items():
        if policy.event != key:
            errors.append(
                f"nudge policy key {key!r} does not match policy.event {policy.event!r}"
            )
    if errors:
        raise RuntimeError("nudge policies incoherent:\n  - " + "\n  - ".join(errors))


_check_nudge_policies_coherent()


@dataclass(frozen=True, slots=True)
class GraphNudgeRequest:
    """Input to the nudge brain (one harness event)."""

    pot_id: str
    event: str
    session_id: str
    scope: Mapping[str, Any] = field(default_factory=dict)
    path: str | None = None  # file path scope (PreToolUse Write/Edit)
    query: str | None = None  # symptom/intent text (test_failed, etc.)
    limit: int = 5  # total injected-item budget across views


@dataclass(frozen=True, slots=True)
class GraphNudgeResult:
    """The hook's payload: inject context, prompt a write, or stay silent."""

    ok: bool
    silent: bool
    event: str
    pot_id: str
    inject_context: str | None = None
    instruction: str | None = None
    injected_keys: tuple[str, ...] = ()
    views_read: tuple[str, ...] = ()
    detail: str | None = None
    graph_contract_version: str = GRAPH_CONTRACT_VERSION
    ontology_version: str = ONTOLOGY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "silent": self.silent,
            "event": self.event,
            "pot_id": self.pot_id,
            "inject_context": self.inject_context,
            "instruction": self.instruction,
            "injected_keys": list(self.injected_keys),
            "views_read": list(self.views_read),
            "detail": self.detail,
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
        }


__all__ = [
    "NUDGE_EVENTS",
    "NUDGE_EVENT_HELP",
    "NUDGE_POLICIES",
    "GraphNudgeRequest",
    "GraphNudgeResult",
    "NudgeDirection",
    "NudgeEvent",
    "NudgePolicy",
    "NudgeViewSpec",
    "canonical_nudge_event",
    "is_nudge_event",
]
