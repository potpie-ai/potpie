"""Immutable reconciliation behavior selected by runtime composition."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReconciliationConfig:
    """All reconciliation choices for one composed graph runtime."""

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

_ACTIVE_RECONCILIATION_CONFIG: ContextVar[ReconciliationConfig | None] = ContextVar(
    "potpie_active_reconciliation_config",
    default=None,
)


def current_reconciliation_config() -> ReconciliationConfig | None:
    """Return the config bound to the current mutation call, if any."""

    return _ACTIVE_RECONCILIATION_CONFIG.get()


@contextmanager
def reconciliation_config_scope(
    config: ReconciliationConfig,
) -> Iterator[None]:
    """Bind one runtime's config without changing the backend call signature."""

    token = _ACTIVE_RECONCILIATION_CONFIG.set(config)
    try:
        yield
    finally:
        _ACTIVE_RECONCILIATION_CONFIG.reset(token)


__all__ = [
    "DEFAULT_RECONCILIATION_CONFIG",
    "ReconciliationConfig",
    "current_reconciliation_config",
    "reconciliation_config_scope",
]
