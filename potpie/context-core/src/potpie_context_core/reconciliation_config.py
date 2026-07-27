"""Immutable reconciliation behavior injected by runtime composition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReconciliationConfig:
    """All reconciliation feature choices for one composed runtime.

    The core never reads process environment variables to make these decisions.
    Hosts may translate environment, files, or managed configuration into this
    value once and inject it at their composition root.
    """

    enabled: bool = True
    agent_planner_enabled: bool = False
    infer_canonical_labels: bool = True
    conflict_detection: bool = True
    auto_supersede: bool = True
    causal_expand: bool = True
    strict_extraction: bool = True
    ontology_soft_fail: bool = True
    ontology_strict: bool = False

    def __post_init__(self) -> None:
        if self.ontology_soft_fail and self.ontology_strict:
            raise ValueError(
                "ontology_soft_fail and ontology_strict cannot both be enabled"
            )


DEFAULT_RECONCILIATION_CONFIG = ReconciliationConfig()


__all__ = ["DEFAULT_RECONCILIATION_CONFIG", "ReconciliationConfig"]
