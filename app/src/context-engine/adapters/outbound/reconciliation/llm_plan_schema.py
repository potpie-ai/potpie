"""Pydantic shapes for LLM structured output (pydantic-ai / pydantic-deep)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LlmEpisodeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    episode_body: str
    source_description: str
    reference_time: datetime | None = None


class LlmEntityUpsert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_key: str
    labels: list[str]
    properties: dict[str, Any] = Field(default_factory=dict)


class LlmEdgeUpsert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_type: str
    from_entity_key: str
    to_entity_key: str
    properties: dict[str, Any] = Field(default_factory=dict)


class LlmEdgeDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_type: str
    from_entity_key: str
    to_entity_key: str


class LlmInvalidationOp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    target_entity_key: str | None = None
    edge_type: str | None = None
    from_entity_key: str | None = None
    to_entity_key: str | None = None


class LlmEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    ref: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LlmReconciliationPlan(BaseModel):
    """Structured plan returned by the planner model (no compat PR bundle)."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    episodes: list[LlmEpisodeDraft] = Field(default_factory=list)
    entity_upserts: list[LlmEntityUpsert] = Field(default_factory=list)
    edge_upserts: list[LlmEdgeUpsert] = Field(default_factory=list)
    edge_deletes: list[LlmEdgeDelete] = Field(default_factory=list)
    invalidations: list[LlmInvalidationOp] = Field(default_factory=list)
    evidence: list[LlmEvidenceRef] = Field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
