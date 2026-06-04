"""Pydantic models for specification generation system."""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# Enums
class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GuardrailType(str, Enum):
    MUST = "must"
    MUST_NOT = "must_not"
    SHOULD = "should"
    SHOULD_NOT = "should_not"


class FileImpactAction(str, Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


class EvidenceType(str, Enum):
    FILE_REFERENCE = "file_reference"
    URL = "url"
    CODE_PATTERN = "code_pattern"
    AGENT_RESULT = "agent_result"


# Core nested models
class Guardrail(BaseModel):
    type: GuardrailType
    statement: str
    rationale: str
    consequences: List[str] = Field(default_factory=list)


class ImplementationRecommendation(BaseModel):
    area: str
    recommendation: str
    rationale: str
    examples: Optional[str] = None
    libraries: List[str] = Field(default_factory=list)


class ExternalDependency(BaseModel):
    name: str
    version: str
    license: str
    purpose: str
    security_status: Optional[str] = None
    source: str
    required_by: List[str] = Field(default_factory=list)


class FileImpact(BaseModel):
    path: str
    purpose: str
    action: FileImpactAction


class Evidence(BaseModel):
    type: EvidenceType
    source: str
    relevance: str


class RequirementAppendix(BaseModel):
    evidence: List[Evidence] = Field(default_factory=list)
    edge_cases: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    research_sources: List[str] = Field(default_factory=list)


# Requirements models
class FunctionalRequirementCore(BaseModel):
    id: str  # e.g., "FR-001"
    title: str
    description: str
    acceptance_criteria: List[str]
    priority: Priority
    dependencies: List[str] = Field(default_factory=list)


class NonFunctionalRequirementCore(BaseModel):
    id: str  # e.g., "NFR-001"
    title: str
    category: str  # performance, security, scalability, etc.
    description: str
    acceptance_criteria: List[str]
    priority: Priority
    dependencies: List[str] = Field(default_factory=list)
    measurement_methodology: str


class FunctionalRequirement(FunctionalRequirementCore):
    guardrails: List[Guardrail] = Field(default_factory=list)
    implementation_recommendations: List[ImplementationRecommendation] = Field(default_factory=list)
    external_dependencies: List[ExternalDependency] = Field(default_factory=list)
    file_impact: List[FileImpact] = Field(default_factory=list)
    appendix: RequirementAppendix = Field(default_factory=RequirementAppendix)


class NonFunctionalRequirement(NonFunctionalRequirementCore):
    guardrails: List[Guardrail] = Field(default_factory=list)
    implementation_recommendations: List[ImplementationRecommendation] = Field(default_factory=list)
    external_dependencies: List[ExternalDependency] = Field(default_factory=list)
    file_impact: List[FileImpact] = Field(default_factory=list)
    appendix: RequirementAppendix = Field(default_factory=RequirementAppendix)


# Research models
class ResearchSource(BaseModel):
    query: str
    findings: str
    source: str  # "explore_agent" or "librarian_agent"


class ResearchFindings(BaseModel):
    sources: List[ResearchSource] = Field(min_length=5)
    summary: str  # 500-1000 words


# Requirements output models
class RequirementsOutputCore(BaseModel):
    functional_requirements: List[FunctionalRequirementCore] = Field(min_length=3, max_length=5)
    non_functional_requirements: List[NonFunctionalRequirementCore] = Field(min_length=2, max_length=3)
    success_metrics: List[str] = Field(min_length=3, max_length=4)


class RequirementEnrichment(BaseModel):
    requirement_id: str
    guardrails: List[Guardrail] = Field(default_factory=list)
    implementation_recommendations: List[ImplementationRecommendation] = Field(default_factory=list)
    external_dependencies: List[ExternalDependency] = Field(default_factory=list)
    file_impact: List[FileImpact] = Field(default_factory=list)
    appendix: RequirementAppendix = Field(default_factory=RequirementAppendix)


class RequirementEnrichmentOutput(BaseModel):
    enrichments: List[RequirementEnrichment]


class RequirementsOutput(BaseModel):
    functional_requirements: List[FunctionalRequirement]
    non_functional_requirements: List[NonFunctionalRequirement]
    success_metrics: List[str]


# Architecture models
class AlternativeConsidered(BaseModel):
    option: str
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    why_rejected: str


class ArchitecturalDecisionRecord(BaseModel):
    id: str  # e.g., "ADR-001"
    title: str
    decision: str
    rationale: str
    alternatives_considered: List[AlternativeConsidered] = Field(min_length=2)
    consequences: List[str] = Field(min_length=2)
    file_impact: List[FileImpact] = Field(default_factory=list)
    appendix: Optional[RequirementAppendix] = None


class ArchitectureOutput(BaseModel):
    adrs: List[ArchitecturalDecisionRecord] = Field(min_length=3, max_length=5)


# Technical design models
class ModelField(BaseModel):
    name: str
    type: str
    required: bool
    description: str
    constraints: Optional[str] = None


class DataModel(BaseModel):
    name: str
    purpose: str
    fields: List[ModelField] = Field(max_length=10)
    location: str
    database_table: Optional[str] = None
    indexes: List[str] = Field(default_factory=list)


class Request(BaseModel):
    headers: dict = Field(default_factory=dict)
    body: dict = Field(default_factory=dict)
    rate_limiting: Optional[str] = None


class Response(BaseModel):
    status: int
    description: str
    body: dict = Field(default_factory=dict)


class Interface(BaseModel):
    name: str
    method: str  # GET, POST, PUT, DELETE, PATCH
    endpoint: str
    description: str
    request: Request
    responses: List[Response] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    file_location: str


class TechnicalDesignOutput(BaseModel):
    data_models: List[DataModel] = Field(min_length=3, max_length=5)
    interfaces: List[Interface] = Field(min_length=5, max_length=10)
    external_dependencies: List[ExternalDependency] = Field(min_length=5, max_length=15)


# Validation models
class ValidationReport(BaseModel):
    passed: bool
    feedback: str
    target_step: Optional[str] = None  # 'requirements', 'architecture', 'technical_design'
    iteration: int


# Specification context
class SpecContext(BaseModel):
    original_request: str
    janus_analysis: str = ""
    qa_answers: str = ""
    research_findings: str = ""


# Final specification
class Specification(BaseModel):
    tl_dr: str  # 3-4 sentence executive summary
    context: SpecContext
    success_metrics: List[str]
    functional_requirements: List[FunctionalRequirement]
    non_functional_requirements: List[NonFunctionalRequirement]
    architectural_decisions: List[ArchitecturalDecisionRecord]
    data_models: List[DataModel]
    interfaces: List[Interface]
    external_dependencies_summary: List[ExternalDependency]
