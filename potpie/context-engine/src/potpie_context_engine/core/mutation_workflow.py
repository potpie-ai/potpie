"""Dependencies shared by mutation preparation and execution coordination."""

from collections.abc import Callable
from typing import Any, Protocol

from .definition import GraphDefinition
from .graph_plans import GraphIngestionVerificationResult
from .mutation_policy import GraphMutationPolicy
from .ports.graph.backend import GraphBackend
from .ports.graph.plan_store import GraphPlanStorePort
from .reconciliation_config import ReconciliationConfig
from .semantic_mutations import SemanticMutationRequest


class MutationWorkflow(Protocol):
    backend: GraphBackend
    plan_store: GraphPlanStorePort
    definition: GraphDefinition
    policy: GraphMutationPolicy
    reconciliation_config: ReconciliationConfig
    validator: Callable[[SemanticMutationRequest], Any]
    lowerer: Callable[[SemanticMutationRequest, Any], Any]
    default_plan_ttl_seconds: int
    commit_claim_timeout_seconds: int

    def verify_commit(
        self, plan_id: str, *, pot_id: str
    ) -> GraphIngestionVerificationResult: ...
