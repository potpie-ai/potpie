"""Ontology lifecycle and extraction edge registration."""

from __future__ import annotations

from domain.ontology import EDGE_TYPES, LifecycleStatus


def test_lifecycle_status_enum_values() -> None:
    assert LifecycleStatus.planned.value == "planned"
    assert LifecycleStatus.completed.value == "completed"


def test_generic_action_in_canonical_edges() -> None:
    assert "GENERIC_ACTION" in EDGE_TYPES
