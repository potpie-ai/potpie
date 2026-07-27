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
    EdgeTypeSpec,
    EntityTypeSpec,
    RecordTypeSpec,
)
from potpie_context_core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
    GraphDefinitionError,
    GraphExtension,
    GraphReader,
    GraphReaderFactory,
    GraphReaderSpec,
)
from potpie_context_core.graph_views import GraphViewSpec
from potpie_context_core.identity import (
    IdentityClass,
    IdentityError,
    IdentitySpec,
    mint_entity_key,
    validate_entity_key,
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
    GraphEntityRef,
    GraphEvidenceRef,
    LoweredOperation,
    MutationActor,
    SemanticMutation,
    SemanticMutationParseError,
    SemanticMutationPlan,
    SemanticMutationRequest,
    SemanticMutationResult,
)
from potpie_context_core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_core.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
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
from potpie_context_core.agent_envelope import (
    CoverageReport as AgentCoverageReport,
    EvidenceItem,
    UnsupportedInclude,
)
from potpie_context_core.graph_workbench import (
    GraphCommandEnvelope,
    GraphCommandError,
    GraphRecommendedAction,
    GraphUnsupported,
    GraphUnsupportedResult,
    GraphWorkbenchStatus,
)
from potpie_context_core.graph_history import (
    GraphHistoryEntry,
    GraphHistoryRequest,
    GraphHistoryResult,
)
from potpie_context_core.graph_inbox import (
    GraphInboxItem,
    GraphInboxResult,
    GraphInboxStatus,
)
from potpie_context_core.graph_plans import (
    GraphIngestionVerificationResult,
    GraphMutationApproval,
    GraphMutationCommitResult,
    GraphMutationDiff,
    GraphMutationPlanRecord,
    GraphMutationPlanStatus,
    GraphMutationProposal,
)
from potpie_context_core.graph_quality import (
    GraphQualityFinding,
    GraphQualityIssue,
    GraphQualityReport,
    GraphQualityResult,
)
from potpie_context_core.mutation_policy import (
    DEFAULT_MUTATION_POLICY,
    GraphMutationPolicy,
)
from potpie_context_core.reconciliation_config import (
    DEFAULT_RECONCILIATION_CONFIG,
    ReconciliationConfig,
)
from potpie_context_core.runtime import (
    GraphObserver,
    GraphRuntime,
    NoOpGraphObserver,
    RuntimeCompositionError,
    build_graph_runtime,
)
from potpie_context_core.lifecycle import SetupPlan, StepResult
from potpie_context_core.serialization import from_transport, to_transport

# --- Ports: the contracts a graph runtime implements.
from potpie_context_core.agent_envelope import AgentEnvelope
from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    StatusReport,
    StatusRequest,
)
from potpie_context_core.ports.claim_query import (
    AsyncClaimQueryPort,
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
)
from potpie_context_core.ports.graph.analytics import (
    GraphAnalyticsPort,
    RepairReport,
)
from potpie_context_core.ports.graph.backend import (
    BackendCapabilities,
    GraphBackend,
)
from potpie_context_core.ports.graph.inbox_store import (
    AsyncGraphInboxStorePort,
    GraphInboxStorePort,
)
from potpie_context_core.ports.graph.inspection import (
    GraphEdge,
    GraphInspectionPort,
    GraphNode,
    GraphSlice,
)
from potpie_context_core.ports.graph.mutation import (
    BackendReadiness,
    GraphMutationPort,
)
from potpie_context_core.ports.graph.plan_store import (
    AsyncGraphPlanStorePort,
    GraphPlanStorePort,
)
from potpie_context_core.ports.graph.semantic import SemanticSearchPort
from potpie_context_core.ports.graph.snapshot import (
    GraphSnapshotPort,
    SnapshotManifest,
)
from potpie_context_core.ports.graph_service import (
    DataPlaneStatus,
    GraphCatalogRequest,
    GraphCatalogResult,
    GraphDescribeRequest,
    GraphEntityCandidate,
    GraphEntitySearchRequest,
    GraphEntitySearchResult,
    GraphReadRequest,
    GraphReadResult,
    GraphService,
)

__all__ = [
    # ontology
    "EDGE_TYPES",
    "ENTITY_TYPES",
    "PUBLIC_RECORD_TYPES",
    "RECORD_TYPES",
    "DEFAULT_GRAPH_DEFINITION",
    "GraphDefinition",
    "GraphDefinitionError",
    "GraphExtension",
    "GraphReader",
    "GraphReaderFactory",
    "GraphReaderSpec",
    "EntityTypeSpec",
    "EdgeTypeSpec",
    "RecordTypeSpec",
    "GraphViewSpec",
    "IdentityClass",
    "IdentityError",
    "IdentitySpec",
    "mint_entity_key",
    "validate_entity_key",
    # records
    "BugPatternRecord",
    "ContextRecordValidationError",
    "DecisionRecord",
    "FixRecord",
    "FreeFormRecord",
    "PreferenceRecord",
    # mutations + transforms
    "SemanticMutation",
    "MutationBatch",
    "MutationResult",
    "MutationSummary",
    "EntityUpsert",
    "EdgeUpsert",
    "EdgeDelete",
    "InvalidationOp",
    "ProvenanceContext",
    "GraphEntityRef",
    "GraphEvidenceRef",
    "LoweredOperation",
    "MutationActor",
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
    "GraphCommandError",
    "GraphRecommendedAction",
    "GraphUnsupported",
    "GraphUnsupportedResult",
    "GraphWorkbenchStatus",
    "AgentCoverageReport",
    "EvidenceItem",
    "UnsupportedInclude",
    "GraphQualityFinding",
    "GraphQualityIssue",
    "GraphQualityReport",
    "GraphQualityResult",
    "GraphHistoryEntry",
    "GraphHistoryRequest",
    "GraphHistoryResult",
    "GraphInboxItem",
    "GraphInboxResult",
    "GraphInboxStatus",
    "GraphIngestionVerificationResult",
    "GraphMutationApproval",
    "GraphMutationCommitResult",
    "GraphMutationDiff",
    "GraphMutationPlanRecord",
    "GraphMutationPlanStatus",
    "GraphMutationProposal",
    "DEFAULT_MUTATION_POLICY",
    "DEFAULT_RECONCILIATION_CONFIG",
    "GraphMutationPolicy",
    "ReconciliationConfig",
    "GraphObserver",
    "GraphRuntime",
    "NoOpGraphObserver",
    "RuntimeCompositionError",
    "SetupPlan",
    "StepResult",
    "build_graph_runtime",
    "from_transport",
    "to_transport",
    # ports
    "AgentEnvelope",
    "ClaimQueryPort",
    "AsyncClaimQueryPort",
    "ClaimQueryFilter",
    "ClaimRow",
    "CapabilityNotImplemented",
    "BackendCapabilities",
    "BackendReadiness",
    "GraphBackend",
    "AsyncGraphInboxStorePort",
    "GraphInboxStorePort",
    "GraphMutationPort",
    "AsyncGraphPlanStorePort",
    "GraphPlanStorePort",
    "GraphAnalyticsPort",
    "RepairReport",
    "GraphInspectionPort",
    "GraphNode",
    "GraphEdge",
    "GraphSlice",
    "SemanticSearchPort",
    "GraphSnapshotPort",
    "SnapshotManifest",
    "GraphService",
    "DataPlaneStatus",
    "GraphCatalogRequest",
    "GraphCatalogResult",
    "GraphDescribeRequest",
    "GraphEntityCandidate",
    "GraphEntitySearchRequest",
    "GraphEntitySearchResult",
    "GraphReadRequest",
    "GraphReadResult",
    "RecordReceipt",
    "RecordRequest",
    "ResolveRequest",
    "SearchRequest",
    "StatusReport",
    "StatusRequest",
]
