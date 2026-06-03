"""The ``GraphBackend`` capability ports.

One swappable backend = six narrow capability ports. ``mutation`` +
``claim_query`` are the canonical source of truth; ``semantic`` ``inspection``
``analytics`` ``snapshot`` are rebuildable projections. See ``backend.py``.
"""

from __future__ import annotations

from domain.ports.graph.analytics import GraphAnalyticsPort, RepairReport
from domain.ports.graph.backend import (
    BackendCapabilities,
    BackendReadiness,
    GraphBackend,
)
from domain.ports.graph.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
)
from domain.ports.graph.inspection import (
    GraphEdge,
    GraphInspectionPort,
    GraphNode,
    GraphSlice,
)
from domain.ports.graph.mutation import GraphMutationPort
from domain.ports.graph.semantic import SemanticSearchPort
from domain.ports.graph.snapshot import GraphSnapshotPort, SnapshotManifest

__all__ = [
    "BackendCapabilities",
    "BackendReadiness",
    "ClaimQueryFilter",
    "ClaimQueryPort",
    "ClaimRow",
    "GraphAnalyticsPort",
    "GraphBackend",
    "GraphEdge",
    "GraphInspectionPort",
    "GraphMutationPort",
    "GraphNode",
    "GraphSlice",
    "GraphSnapshotPort",
    "RepairReport",
    "SemanticSearchPort",
    "SnapshotManifest",
]
