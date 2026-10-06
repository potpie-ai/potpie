"""Gate: the curated api module re-exports the internal contracts."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_every_declared_export_resolves() -> None:
    from potpie_context_engine.core import api

    missing = [name for name in api.__all__ if not hasattr(api, name)]
    assert missing == []


def test_api_reexports_are_the_internal_contracts() -> None:
    from potpie_context_engine.core import api
    from potpie_context_engine.core.reconciliation_config import ReconciliationConfig
    from potpie_context_engine.core.workbench_service import (
        GraphWorkbenchService,
    )
    from potpie_context_engine.core.ports.graph.backend import GraphBackend
    from potpie_context_engine.core.ports.graph_service import GraphService

    assert api.GraphBackend is GraphBackend
    assert api.GraphService is GraphService
    assert api.GraphWorkbenchService is GraphWorkbenchService
    assert api.ReconciliationConfig is ReconciliationConfig
    assert api.DEFAULT_RECONCILIATION_CONFIG == ReconciliationConfig()


def test_runtime_builder_is_exported_by_the_engine_api_not_the_core_api() -> None:
    """The builder composes the engine's default graph service, so it is an
    engine-level export; the core surface never reaches that implementation."""
    from potpie_context_engine import api as engine_api
    from potpie_context_engine.core import api as core_api

    assert {"GraphRuntime", "build_graph_runtime"}.isdisjoint(core_api.__all__)
    assert {"GraphRuntime", "build_graph_runtime"} <= set(engine_api.__all__)
    assert {
        "GraphObserver",
        "NoOpGraphObserver",
        "RuntimeCompositionError",
    } <= set(core_api.__all__)
