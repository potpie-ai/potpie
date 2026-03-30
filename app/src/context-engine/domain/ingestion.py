"""Ingestion result types (application/domain boundary)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class IngestionResult:
    """Structured return from merged-PR ingestion."""

    episode_uuid: str | None
    pr_entity_key: str
    already_existed: bool = False
    stamp_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class BridgeResult:
    """Counts of structural edges written."""

    touched_by: int = 0
    modified_in: int = 0
    has_decision: int = 0

    def total(self) -> int:
        return self.touched_by + self.modified_in + self.has_decision

    def as_dict(self) -> dict[str, int]:
        return {
            "touched_by": self.touched_by,
            "modified_in": self.modified_in,
            "has_decision": self.has_decision,
        }
