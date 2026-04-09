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
