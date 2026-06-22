"""Graph V2 history DTOs.

History is a read-only workbench view over mutation plans and committed claim
metadata. It deliberately reuses the plan store and ``ClaimQueryPort`` instead
of introducing a second mutation ledger in the first V2 cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from domain.graph_workbench import GRAPH_WORKBENCH_CONTRACT_VERSION
from domain.graph_workbench import GRAPH_WORKBENCH_ONTOLOGY_VERSION


@dataclass(frozen=True, slots=True)
class GraphHistoryRequest:
    pot_id: str
    entity_key: str | None = None
    claim_key: str | None = None
    subgraph: str | None = None
    plan_id: str | None = None
    mutation_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 50

    def filters(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (
            ("entity", self.entity_key),
            ("claim", self.claim_key),
            ("subgraph", self.subgraph),
            ("plan", self.plan_id),
            ("mutation", self.mutation_id),
        ):
            if value:
                out[key] = value
        if self.since:
            out["since"] = self.since.isoformat()
        if self.until:
            out["until"] = self.until.isoformat()
        out["limit"] = self.limit
        return out


@dataclass(frozen=True, slots=True)
class GraphHistoryEntry:
    kind: str
    id: str
    status: str | None = None
    occurred_at: datetime | None = None
    plan_id: str | None = None
    mutation_id: str | None = None
    claim_key: str | None = None
    entity_keys: tuple[str, ...] = ()
    subgraph: str | None = None
    truth: str | None = None
    source_refs: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    summary: str | None = None
    detail: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "id": self.id,
            "entity_keys": list(self.entity_keys),
            "source_refs": list(self.source_refs),
            "evidence": [dict(item) for item in self.evidence],
            "payload": dict(self.payload),
        }
        values = (
            ("status", self.status),
            ("occurred_at", self.occurred_at.isoformat() if self.occurred_at else None),
            ("plan_id", self.plan_id),
            ("mutation_id", self.mutation_id),
            ("claim_key", self.claim_key),
            ("subgraph", self.subgraph),
            ("truth", self.truth),
            ("summary", self.summary),
            ("detail", self.detail),
        )
        for key, value in values:
            if value is not None:
                out[key] = value
        return out


@dataclass(frozen=True, slots=True)
class GraphHistoryResult:
    ok: bool
    pot_id: str
    filters: Mapping[str, Any]
    entries: tuple[GraphHistoryEntry, ...] = ()
    warnings: tuple[str, ...] = ()
    unsupported: tuple[Mapping[str, Any], ...] = ()
    detail: str | None = None
    recommended_next_action: str | None = None
    subgraph_versions: Mapping[str, int] = field(default_factory=dict)
    graph_contract_version: str = GRAPH_WORKBENCH_CONTRACT_VERSION
    ontology_version: str = GRAPH_WORKBENCH_ONTOLOGY_VERSION

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "pot_id": self.pot_id,
            "filters": dict(self.filters),
            "entries": [entry.to_dict() for entry in self.entries],
            "entry_count": len(self.entries),
            "warnings": list(self.warnings),
            "unsupported": [dict(item) for item in self.unsupported],
            "subgraph_versions": dict(self.subgraph_versions),
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
        }
        if self.detail:
            out["detail"] = self.detail
        if self.recommended_next_action:
            out["recommended_next_action"] = self.recommended_next_action
        return out


__all__ = [
    "GraphHistoryEntry",
    "GraphHistoryRequest",
    "GraphHistoryResult",
]
