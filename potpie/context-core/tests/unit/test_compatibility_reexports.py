"""Temporary package forwards existing imports to Context Engine-owned types."""

from __future__ import annotations

# ruff: noqa: S101 - pytest compatibility tests use assertions intentionally.


def test_curated_api_reexports_canonical_engine_types() -> None:
    from potpie_context_core import api as compatibility
    from potpie_context_core.ports.graph.backend import GraphBackend as LegacyBackend
    from potpie_context_engine.core import api as canonical
    from potpie_context_engine.core.ports.graph.backend import GraphBackend

    assert compatibility.GraphBackend is canonical.GraphBackend
    assert LegacyBackend is GraphBackend
    assert compatibility.__all__ == canonical.__all__


def test_representative_domain_modules_reexport_same_class_objects() -> None:
    from potpie_context_core.context_events import ContextEvent as LegacyEvent
    from potpie_context_core.graph_mutations import EntityUpsert as LegacyEntityUpsert
    from potpie_context_core.lifecycle import SetupPlan as LegacySetupPlan
    from potpie_context_engine.core.context_events import ContextEvent
    from potpie_context_engine.core.graph_mutations import EntityUpsert
    from potpie_context_engine.core.lifecycle import SetupPlan

    assert LegacyEvent is ContextEvent
    assert LegacyEntityUpsert is EntityUpsert
    assert LegacySetupPlan is SetupPlan
