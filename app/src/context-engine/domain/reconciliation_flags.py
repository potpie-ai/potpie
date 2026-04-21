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


def compat_pr_reconciler_enabled() -> bool:
    """Use compatibility planner path for merged GitHub PRs (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_COMPAT_PR_RECONCILER_ENABLED"), True)


def agent_planner_enabled() -> bool:
    """LLM-backed planner (default: on). Set false to disable the agent path."""
    return _truthy(os.getenv("CONTEXT_ENGINE_AGENT_PLANNER_ENABLED"), True)


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


def classify_modified_edges_enabled() -> bool:
    """Allow classify-modified-edges maintenance to apply rewrites (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES"), True)


def allow_edge_classify_write_enabled() -> bool:
    """Allow non-dry-run writes for classify-modified-edges (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE"), True)


def strict_extraction_enabled() -> bool:
    """Strict episodic edge extraction guard (default: on)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_STRICT_EXTRACTION"), True)


def ontology_soft_fail_enabled() -> bool:
    """When true, unknown labels / edge types / invalid lifecycle coerce instead of rejecting."""
    return _truthy(os.getenv("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL"), False)


def ontology_strict_enabled() -> bool:
    """Force strict ontology validation (disables soft downgrade; default: off)."""
    return _truthy(os.getenv("CONTEXT_ENGINE_ONTOLOGY_STRICT"), False)
