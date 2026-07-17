"""Feature flags for reconciliation (env-backed; no Potpie imports)."""

from __future__ import annotations

import os


def _truthy(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in ("0", "false", "no", "off", ""):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    return default


def reconciliation_enabled() -> bool:
    """Master switch for reconciliation lifecycle (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_RECONCILIATION_ENABLED"), True)


def agent_planner_enabled() -> bool:
    """LLM-backed reconciliation planner — **opt-in** (default: off).

    Graph V1.5 Step 11 parks service-side LLM reconciliation as non-canonical:
    canonical graph writes go through the harness-facing semantic mutation
    surface, never a Potpie-owned planner. The agentic path stays available for
    experiments/backfill — set ``CONTEXT_ENGINE_AGENT_PLANNER_ENABLED=1`` to opt
    back in. Event playbooks (:func:`potpie_context_core.event_playbooks.playbooks_enable_planner`)
    can still enable it per-batch for one-shot backfill without the global flag.
    """
    return _truthy(os.getenv("CONTEXT_ENGINE_AGENT_PLANNER_ENABLED"), False)


def infer_canonical_labels_enabled() -> bool:
    """Infer ontology labels from episodic edges and enrich reconciliation plans (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_INFER_LABELS"), True)


def conflict_detection_enabled() -> bool:
    """Predicate-family conflict detection + ``QualityIssue`` persistence (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_CONFLICT_DETECT"), True)


def auto_supersede_enabled() -> bool:
    """Predicate-family auto-invalidate older episodic edges (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_AUTO_SUPERSEDE"), True)


def causal_expand_enabled() -> bool:
    """One-hop causal expansion in semantic ``search`` (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_CAUSAL_EXPAND"), True)


def strict_extraction_enabled() -> bool:
    """Strict episodic edge extraction guard (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_STRICT_EXTRACTION"), True)


def ontology_soft_fail_enabled() -> bool:
    """Coerce LLM-extracted labels / edge types / lifecycle values instead of rejecting (default: on).

    The reconciliation agent is an LLM whose output drifts in small ways from the
    canonical ontology (e.g. ``status='active'`` for a Decision, or an edge type
    rendered as ``AUTHORED_BY`` instead of ``AUTHORED``). Hard-rejecting the batch
    causes large, observable HTTP 500 storms because *every* extracted event
    gets dropped. We let the downgrade machinery normalize what it can, record
    what it changed, and surface that on the QualityIssue stream — so callers
    still see the drift but the graph keeps moving forward.

    Set ``CONTEXT_ENGINE_ONTOLOGY_STRICT=1`` to force the legacy hard-fail
    behaviour (useful for tests / golden fixtures that want byte-exact plans).
    """
    return _truthy(os.getenv("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL"), True)


def ontology_strict_enabled() -> bool:
    """Force strict ontology validation (disables soft downgrade; default: off)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_ONTOLOGY_STRICT"), False)
