"""Graph V2 inbox DTOs.

Inbox items are pending graph work. They intentionally do not become graph
facts until a harness processes them through read/search, propose, and commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping

from context_engine.domain.graph_workbench import GRAPH_WORKBENCH_CONTRACT_VERSION
from context_engine.domain.graph_workbench import GRAPH_WORKBENCH_ONTOLOGY_VERSION


class GraphInboxStatus(StrEnum):
    pending = "pending"
    claimed = "claimed"
    applied = "applied"
    rejected = "rejected"
    closed = "closed"


TERMINAL_INBOX_STATUSES: frozenset[str] = frozenset(
    {
        GraphInboxStatus.applied.value,
        GraphInboxStatus.rejected.value,
        GraphInboxStatus.closed.value,
    }
)


@dataclass(frozen=True, slots=True)
class GraphInboxItem:
    item_id: str
    pot_id: str
    status: str
    summary: str
    details: str | None = None
    evidence: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    suspected_subgraphs: tuple[str, ...] = ()
    created_by: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    claimed_by: str | None = None
    claimed_at: datetime | None = None
    closed_by: str | None = None
    closed_at: datetime | None = None
    linked_plan_id: str | None = None
    linked_mutation_id: str | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "item_id": self.item_id,
            "inbox_id": self.item_id,
            "pot_id": self.pot_id,
            "status": self.status,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "source_refs": list(self.source_refs),
            "suspected_subgraphs": list(self.suspected_subgraphs),
            "suggested_subgraphs": list(self.suspected_subgraphs),
            "created_by": _json_safe_mapping(self.created_by),
            "created_at": self.created_at.isoformat(),
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
            "closed_by": self.closed_by,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "linked_plan_id": self.linked_plan_id,
            "linked_mutation_id": self.linked_mutation_id,
            "rejection_reason": self.rejection_reason,
        }
        if self.details:
            out["details"] = self.details
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphInboxItem":
        item_id = str(raw.get("item_id") or raw.get("inbox_id") or "")
        return cls(
            item_id=item_id,
            pot_id=str(raw.get("pot_id") or ""),
            status=str(raw.get("status") or GraphInboxStatus.pending.value),
            summary=str(raw.get("summary") or ""),
            details=str(raw.get("details") or "") or None,
            evidence=_string_tuple(raw.get("evidence")),
            source_refs=_string_tuple(raw.get("source_refs")),
            suspected_subgraphs=_string_tuple(
                raw.get("suspected_subgraphs") or raw.get("suggested_subgraphs")
            ),
            created_by=_mapping(raw.get("created_by")),
            created_at=_parse_datetime(raw.get("created_at"))
            or datetime.now(timezone.utc),
            claimed_by=str(raw.get("claimed_by") or "") or None,
            claimed_at=_parse_datetime(raw.get("claimed_at")),
            closed_by=str(raw.get("closed_by") or "") or None,
            closed_at=_parse_datetime(raw.get("closed_at")),
            linked_plan_id=str(raw.get("linked_plan_id") or "") or None,
            linked_mutation_id=str(raw.get("linked_mutation_id") or "") or None,
            rejection_reason=str(raw.get("rejection_reason") or "") or None,
        )


@dataclass(frozen=True, slots=True)
class GraphInboxResult:
    ok: bool
    pot_id: str
    action: str
    item: GraphInboxItem | None = None
    items: tuple[GraphInboxItem, ...] = ()
    filters: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    detail: str | None = None
    recommended_next_action: str | None = None
    graph_contract_version: str = GRAPH_WORKBENCH_CONTRACT_VERSION
    ontology_version: str = GRAPH_WORKBENCH_ONTOLOGY_VERSION

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "pot_id": self.pot_id,
            "action": self.action,
            "filters": dict(self.filters),
            "items": [item.to_dict() for item in self.items],
            "item_count": len(self.items),
            "warnings": list(self.warnings),
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
        }
        if self.item is not None:
            out["item"] = self.item.to_dict()
        if self.detail:
            out["detail"] = self.detail
        if self.recommended_next_action:
            out["recommended_next_action"] = self.recommended_next_action
        return out


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item)
    return (str(value),)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return {}


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "GraphInboxItem",
    "GraphInboxResult",
    "GraphInboxStatus",
    "TERMINAL_INBOX_STATUSES",
]
