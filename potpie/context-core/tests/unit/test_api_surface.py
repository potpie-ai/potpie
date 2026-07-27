"""Gate: the curated api module re-exports the internal contracts."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_every_declared_export_resolves() -> None:
    from potpie_context_core import api

    missing = [name for name in api.__all__ if not hasattr(api, name)]
    assert missing == []


def test_api_reexports_are_the_internal_contracts() -> None:
    from potpie_context_core import api
    from potpie_context_core.workbench_service import (
        GraphWorkbenchService,
    )
    from potpie_context_core.ports.graph.backend import GraphBackend
    from potpie_context_core.ports.graph_service import GraphService

    assert api.GraphBackend is GraphBackend
    assert api.GraphService is GraphService
    assert api.GraphWorkbenchService is GraphWorkbenchService
