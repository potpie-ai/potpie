"""Ontology lifecycle and extraction edge registration."""

from __future__ import annotations

from domain.ontology import EDGE_TYPES, LifecycleStatus


def test_lifecycle_status_enum_values() -> None:
    assert LifecycleStatus.planned.value == "planned"
    assert LifecycleStatus.completed.value == "completed"


def test_related_to_is_the_generic_fallback_edge() -> None:
    # The minimal topology ontology has no lifecycle edges; the only generic
    # edge is the RELATED_TO soft-fail fallback.
    assert "RELATED_TO" in EDGE_TYPES
