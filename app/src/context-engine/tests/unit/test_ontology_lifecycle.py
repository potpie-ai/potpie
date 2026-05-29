"""Ontology lifecycle and extraction edge registration."""

from __future__ import annotations

from domain.ontology import EDGE_TYPES, LifecycleStatus


def test_lifecycle_status_enum_values() -> None:
    assert LifecycleStatus.planned.value == "planned"
    assert LifecycleStatus.completed.value == "completed"


def test_lifecycle_transition_in_canonical_edges() -> None:
    # v2 collapses GENERIC_ACTION / RELATED_TO / PLANNED / DELIVERED / etc.
    # into the single LIFECYCLE_TRANSITION edge with a ``verb`` property.
    assert "LIFECYCLE_TRANSITION" in EDGE_TYPES
