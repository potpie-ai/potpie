"""Default implementation hook used by ``potpie_context_core.build_graph_runtime``."""

from __future__ import annotations

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
) -> DefaultGraphService:
    return DefaultGraphService(
        backend=backend,
        definition=definition,
        policy=policy,
        reconciliation_config=(
            reconciliation_config or reconciliation_config_from_env()
        ),
    )


__all__ = ["build_graph_service"]
