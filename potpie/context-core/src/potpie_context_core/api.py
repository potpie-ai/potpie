"""The supported import surface of potpie-context-core.

Consumers embedding the graph core — the potpie product distribution, the
context engine, or a headless deployment — import from here instead of from
internal module paths, so internal moves do not break them:

    from potpie_context_core.api import GraphBackend, GraphWorkbenchService

Groups, top to bottom: the ontology catalog, context records, semantic
mutations and their transforms, the workbench, and the graph ports a
runtime implements.
"""

from __future__ import annotations

# --- Ontology: the single catalog of labels, predicates, and record types.
from potpie_context_core.ontology import (
    EDGE_TYPES,
    ENTITY_TYPES,
    PUBLIC_RECORD_TYPES,
    RECORD_TYPES,
)

# --- Context records: the typed durable-memory vocabulary.
from potpie_context_core.context_records import (
    BugPatternRecord,
    ContextRecordValidationError,
    DecisionRecord,
    FixRecord,
    FreeFormRecord,
    PreferenceRecord,
)

# --- Semantic mutations: how records become graph writes.
from potpie_context_core.semantic_mutations import (
    SemanticMutation,
    SemanticMutationParseError,
    SemanticMutationPlan,
    SemanticMutationRequest,
    SemanticMutationResult,
)
from potpie_context_core.record_to_semantic import (
    record_to_semantic_request,
)
from potpie_context_core.semantic_mutation_lowering import (
    lower_semantic_request,
)
from potpie_context_core.semantic_mutation_validator import (
    validate_semantic_request,
)

# --- Workbench: the catalog/read/search/mutate command surface.
from potpie_context_core.workbench_service import (
    GraphWorkbenchService,
)
from potpie_context_core.graph_workbench import GraphCommandEnvelope

# --- Ports: the contracts a graph runtime implements.
from potpie_context_core.agent_envelope import AgentEnvelope
from potpie_context_core.ports.claim_query import ClaimQueryPort
from potpie_context_core.ports.graph.backend import GraphBackend
from potpie_context_core.ports.graph.inbox_store import GraphInboxStorePort
from potpie_context_core.ports.graph.mutation import GraphMutationPort
from potpie_context_core.ports.graph.plan_store import GraphPlanStorePort
from potpie_context_core.ports.graph_service import GraphService

__all__ = [
    # ontology
    "EDGE_TYPES",
    "ENTITY_TYPES",
    "PUBLIC_RECORD_TYPES",
    "RECORD_TYPES",
    # records
    "BugPatternRecord",
    "ContextRecordValidationError",
    "DecisionRecord",
    "FixRecord",
    "FreeFormRecord",
    "PreferenceRecord",
    # mutations + transforms
    "SemanticMutation",
    "SemanticMutationParseError",
    "SemanticMutationPlan",
    "SemanticMutationRequest",
    "SemanticMutationResult",
    "record_to_semantic_request",
    "lower_semantic_request",
    "validate_semantic_request",
    # workbench
    "GraphWorkbenchService",
    "GraphCommandEnvelope",
    # ports
    "AgentEnvelope",
    "ClaimQueryPort",
    "GraphBackend",
    "GraphInboxStorePort",
    "GraphMutationPort",
    "GraphPlanStorePort",
    "GraphService",
]
