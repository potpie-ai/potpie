"""Parked agentic-reconciliation domain types (Graph V1.5 Step 11).

The service-side **LLM reconciliation** path is *non-canonical* in V1.5:
canonical graph writes go through semantic mutations
(:mod:`potpie_context_core.semantic_mutations`) lowered to
:class:`~potpie_context_core.reconciliation.MutationBatch` — never a Potpie-owned planner.
These types are used **only** by the parked LLM planner adapter
(:mod:`potpie_context_engine.adapters.outbound.reconciliation`) and its read-tool port
(:mod:`potpie_context_engine.domain.ports.reconciliation_tools`).

They live apart from :mod:`potpie_context_core.reconciliation` so the canonical structural
write tier (``MutationBatch``) carries no agentic vocabulary. The planner is
opt-in (``CONTEXT_ENGINE_AGENT_PLANNER_ENABLED=1``); these types come along
with it. ``potpie_context_core.reconciliation`` re-exports both names as a thin back-compat
shim for one iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from potpie_context_core.context_events import ContextEvent


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Pointer to evidence cited by the (parked) LLM planner.

    The canonical write tier records evidence as structured dicts on each
    edge's ``properties`` (see the semantic lowerer); this DTO is the
    planner-only citation shape that ``MutationBatch.evidence`` accepts when
    the agentic path is enabled.
    """

    kind: str
    ref: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReconciliationRequest:
    """Input to the parked ``ReconciliationAgentPort`` read-tool dispatch."""

    event: ContextEvent
    pot_id: str
    repo_name: str
    prior_attempts: list[dict[str, Any]] = field(default_factory=list)


__all__ = ["EvidenceRef", "ReconciliationRequest"]
