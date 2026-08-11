"""Default implementation hook used by ``potpie_context_core.build_graph_runtime``."""

from __future__ import annotations

from typing import Any

from potpie_context_core.definition import GraphDefinition
from potpie_context_core.mutation_policy import GraphMutationPolicy
from potpie_context_core.ports.graph.backend import GraphBackend
from potpie_context_core.reconciliation_config import ReconciliationConfig
from potpie_context_core.reconciliation_flags import reconciliation_config_from_env
from potpie_context_engine.application.services.graph_service import DefaultGraphService


def build_graph_service(
    *,
    backend: GraphBackend,
    definition: GraphDefinition,
    policy: GraphMutationPolicy,
    reconciliation_config: ReconciliationConfig | None = None,
    resource_index: Any = None,
) -> DefaultGraphService:
    return DefaultGraphService(
        backend=backend,
        definition=definition,
        policy=policy,
        reconciliation_config=(
            reconciliation_config or reconciliation_config_from_env()
        ),
        # A ``ResourceIndexPort``, or ``None`` where the host has no document
        # store: the ``resources`` family then answers ``match_mode="disabled"``
        # rather than disappearing from the contract.
        resource_index=resource_index,
    )


__all__ = ["build_graph_service"]
