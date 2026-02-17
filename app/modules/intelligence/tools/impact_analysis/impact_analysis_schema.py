from typing import List, Literal, Optional

from pydantic import BaseModel, Field


ConfidenceLevel = Literal["high", "medium", "low"]


class ImpactAnalysisRequest(BaseModel):
    changed_file: str = Field(
        ...,
        description="Repo-relative file path containing the changed function(s).",
    )
    function_name: Optional[str] = Field(
        default=None,
        description="Function or method name that changed. Optional when analyzing the whole file.",
    )
    module_hint: Optional[str] = Field(
        default=None,
        description="Optional module/package hint to disambiguate similarly named symbols.",
    )
    strict_mode: bool = Field(
        default=True,
        description="When true, low-confidence recommendations are removed.",
    )
    change_notes: Optional[str] = Field(
        default=None,
        description="Optional user-provided notes to enrich identifier discovery.",
    )


class RecommendedTest(BaseModel):
    name: str = Field(description="Test file name (e.g., test_service.py)")
    file_path: str = Field(description="Repo-relative path to the test file")
    test_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Runnable test identifiers, e.g. tests/test_foo.py::test_bar (pytest), "
            "path::keyword:StepLoginApplication_FlaUI (Robot/FlaUI), or "
            "path::scenario:Scenario Name (SpecFlow)."
        ),
    )
    confidence: ConfidenceLevel
    reason: str
    evidence_ids: List[str] = Field(default_factory=list)


class TracePath(BaseModel):
    source: str
    target: str
    relation: str
    confidence: ConfidenceLevel
    evidence_ids: List[str] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str
    type: str
    file_path: str
    detail: str
    source: str
    confidence: ConfidenceLevel
    matched_text: Optional[str] = None
    original_identifier: Optional[str] = None
    normalized_identifier: Optional[str] = None


class Ambiguity(BaseModel):
    field: str
    message: str
    candidates: List[str] = Field(default_factory=list)


class BlockedByScope(BaseModel):
    file_path: str
    reason: str
    matched_identifier: Optional[str] = None


class ImpactAnalysisResponse(BaseModel):
    recommended_tests: List[RecommendedTest] = Field(default_factory=list)
    trace_paths: List[TracePath] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    ambiguities: List[Ambiguity] = Field(default_factory=list)
    blocked_by_scope: List[BlockedByScope] = Field(default_factory=list)


class ImpactTraceAnalysisInput(ImpactAnalysisRequest):
    project_id: str = Field(..., description="Project ID (UUID).")
