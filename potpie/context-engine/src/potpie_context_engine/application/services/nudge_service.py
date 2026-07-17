"""The nudge brain — execute the deterministic trigger policy (Step 12a).

Composes two primitives: the read trunk (``graph read`` over a named view) and
the per-session injection ledger. For a **data** event it reads the event's
views, ranks across them, drops anything already injected this session, budgets
to top-K, and returns compact source-ref-first context. For an **instruction**
event it returns the policy's write-prompt directive.

Hard rule (Trigger Model): this makes **no model calls**. The only intelligence
is the local embedder behind ``graph read`` (similarity, not generation) and the
in-session agent that later consumes the nudge. No API client is constructed on
this path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

from potpie_context_engine.domain.nudge import (
    NUDGE_POLICIES,
    GraphNudgeRequest,
    GraphNudgeResult,
    NudgeDirection,
    NudgePolicy,
    canonical_nudge_event,
)
from potpie_context_engine.domain.ports.injection_ledger import InjectionLedgerPort
from potpie_context_core.domain.ports.services.graph_service import GraphReadRequest, GraphReadResult


class GraphReadPort(Protocol):
    """The narrow slice of ``GraphService`` the nudge brain needs."""

    def read(self, request: GraphReadRequest) -> GraphReadResult: ...


# Scope keys worth surfacing inline in injected context.
_SCOPE_KEYS = (
    "repo",
    "service",
    "file_path",
    "path",
    "environment",
    "language",
    "framework",
)


@dataclass(slots=True)
class NudgeService:
    """Event → action policy executor over the read trunk + injection ledger."""

    graph: GraphReadPort
    ledger: InjectionLedgerPort

    def nudge(self, request: GraphNudgeRequest) -> GraphNudgeResult:
        canonical_event = canonical_nudge_event(request.event)
        if canonical_event is None:
            return GraphNudgeResult(
                ok=False,
                silent=True,
                event=request.event,
                pot_id=request.pot_id,
                detail=(
                    f"unknown nudge event {request.event!r}; "
                    f"known: {', '.join(sorted(NUDGE_POLICIES))}"
                ),
            )
        if canonical_event != request.event:
            request = replace(request, event=canonical_event)
        policy = NUDGE_POLICIES[canonical_event]
        if policy.direction == NudgeDirection.instruction:
            return self._instruction(request, policy)
        return self._data(request, policy)

    # -- instruction direction ------------------------------------------------
    def _instruction(
        self, request: GraphNudgeRequest, policy: NudgePolicy
    ) -> GraphNudgeResult:
        return GraphNudgeResult(
            ok=True,
            silent=False,
            event=request.event,
            pot_id=request.pot_id,
            instruction=policy.instruction,
        )

    # -- data direction -------------------------------------------------------
    def _data(
        self, request: GraphNudgeRequest, policy: NudgePolicy
    ) -> GraphNudgeResult:
        scope = dict(request.scope)
        if request.path and "file_path" not in scope:
            scope["file_path"] = request.path

        scored: list[tuple[float, str, dict[str, Any]]] = []
        views_read: list[str] = []
        for spec in policy.views:
            subgraph, view = spec.view.split(".", 1)
            result = self.graph.read(
                GraphReadRequest(
                    pot_id=request.pot_id,
                    subgraph=subgraph,
                    view=view,
                    query=request.query if spec.pass_query else None,
                    scope=scope if spec.pass_scope else {},
                    limit=spec.limit,
                )
            )
            views_read.append(spec.view)
            for item in result.items:
                payload = dict(item)
                score = float(payload.get("score") or 0.0)
                if score >= policy.min_score:
                    scored.append((score, spec.view, payload))

        # Global rank across views, then dedup vs the session ledger + within
        # this nudge, budgeted to the caller's top-K.
        scored.sort(key=lambda t: t[0], reverse=True)
        fresh: list[tuple[float, str, dict[str, Any]]] = []
        chosen: set[str] = set()
        for score, view, item in scored:
            key = _injection_key(item)
            if not key or key in chosen:
                continue
            if self.ledger.was_injected(request.session_id, key):
                continue
            chosen.add(key)
            fresh.append((score, view, item))
            if len(fresh) >= request.limit:
                break

        if not fresh:
            return GraphNudgeResult(
                ok=True,
                silent=True,
                event=request.event,
                pot_id=request.pot_id,
                views_read=tuple(views_read),
                detail="nothing relevant above threshold or all already injected",
            )

        injected_keys = tuple(_injection_key(item) for _, _, item in fresh)
        self.ledger.record(request.session_id, injected_keys)
        return GraphNudgeResult(
            ok=True,
            silent=False,
            event=request.event,
            pot_id=request.pot_id,
            inject_context=_format_inject_context(fresh),
            injected_keys=injected_keys,
            views_read=tuple(views_read),
        )


def _format_inject_context(items: list[tuple[float, str, dict[str, Any]]]) -> str:
    """Compact, source-ref-first context block for the session.

    Leads with the agent-authored retrieval card (``description``) when present,
    falls back to ``fact``/``summary``; appends scope + source so the agent can
    treat each line as cited graph truth.
    """
    lines = ["Relevant project memory (Potpie graph):"]
    for _score, view, item in items:
        payload: dict[str, Any] = dict(item)
        text = (
            payload.get("description")
            or payload.get("summary")
            or payload.get("fact")
            or payload.get("entity_key")
        )
        scope_bits: list[str] = []
        code_scope = (
            payload.get("code_scope")
            if isinstance(payload.get("code_scope"), dict)
            else {}
        )
        for key in _SCOPE_KEYS:
            val = payload.get(key) or (code_scope.get(key) if code_scope else None)
            if isinstance(val, str) and val.strip():
                scope_bits.append(f"{key}={val.strip()}")
        source = None
        refs = payload.get("source_refs")
        if isinstance(refs, (list, tuple)) and refs:
            source = refs[0]
        meta: list[str] = []
        if scope_bits:
            meta.append(", ".join(scope_bits))
        if source:
            meta.append(f"src={source}")
        suffix = f" ({'; '.join(meta)})" if meta else ""
        lines.append(f"- [{view}] {text}{suffix}")
    return "\n".join(lines)


def _injection_key(item: Mapping[str, Any]) -> str:
    payload = dict(item)
    claim = payload.get("claim") if isinstance(payload.get("claim"), dict) else {}
    claim_key = claim.get("claim_key") or payload.get("claim_key")
    if isinstance(claim_key, str) and claim_key:
        return claim_key
    relations = payload.get("relations")
    if isinstance(relations, list):
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            rel_key = rel.get("claim_key")
            if isinstance(rel_key, str) and rel_key:
                return rel_key
            claim = rel.get("claim")
            if not isinstance(claim, dict):
                continue
            rel_key = claim.get("candidate_key")
            if isinstance(rel_key, str) and rel_key:
                return rel_key
    return str(payload.get("entity_key") or "")


__all__ = ["GraphReadPort", "NudgeService"]
