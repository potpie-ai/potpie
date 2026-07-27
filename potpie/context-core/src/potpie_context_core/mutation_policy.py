"""Runtime-scoped graph mutation policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GraphMutationPolicy:
    """Composition-time defaults shared by direct mutation and workbench writes."""

    default_plan_ttl_seconds: int = 3600
    allow_auto_commit: bool = True
    require_approval_for_review: bool = True

    def __post_init__(self) -> None:
        if self.default_plan_ttl_seconds < 1:
            raise ValueError("default_plan_ttl_seconds must be at least 1")


DEFAULT_MUTATION_POLICY = GraphMutationPolicy()


__all__ = ["DEFAULT_MUTATION_POLICY", "GraphMutationPolicy"]
