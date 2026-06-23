"""Graph V2 workbench command envelope DTOs.

These DTOs describe the CLI workbench protocol. They deliberately sit above the
V1.5 data plane: the outer envelope can be V2 while a wrapped result still
contains data-plane fields produced by the existing graph service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from context_engine.domain.graph_contract import ONTOLOGY_VERSION

GRAPH_WORKBENCH_CONTRACT_VERSION = "v2"
GRAPH_WORKBENCH_ONTOLOGY_VERSION = ONTOLOGY_VERSION

GRAPH_WORKBENCH_COMMANDS: tuple[str, ...] = (
    "status",
    "catalog",
    "describe",
    "search-entities",
    "read",
    "neighborhood",
    "propose",
    "commit",
    "bulk",
    "history",
    "inbox",
    "quality",
)

GRAPH_WORKBENCH_ADMIN_COMMANDS: tuple[str, ...] = (
    "repair",
    "export",
    "import",
)

GRAPH_WORKBENCH_LEGACY_COMMANDS: tuple[str, ...] = (
    "mutate",
    "inspect",
    "mutation-template",
    "nudge",
)


@dataclass(frozen=True, slots=True)
class GraphCommandError:
    code: str
    message: str
    detail: Any = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }
        return out


@dataclass(frozen=True, slots=True)
class GraphUnsupported:
    name: str
    reason: str
    detail: Any = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "reason": self.reason,
        }
        if self.detail is not None:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True, slots=True)
class GraphRecommendedAction:
    action: str
    command: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.action}
        if self.command:
            out["command"] = self.command
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True, slots=True)
class GraphWorkbenchStatus:
    host: Mapping[str, Any] = field(default_factory=dict)
    pot: Mapping[str, Any] = field(default_factory=dict)
    graph_service: Mapping[str, Any] = field(default_factory=dict)
    backend: Mapping[str, Any] = field(default_factory=dict)
    ledger: Mapping[str, Any] = field(default_factory=dict)
    skills: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": dict(self.host),
            "pot": dict(self.pot),
            "graph_service": dict(self.graph_service),
            "backend": dict(self.backend),
            "ledger": dict(self.ledger),
            "skills": dict(self.skills),
            "quality": dict(self.quality),
        }


@dataclass(frozen=True, slots=True)
class GraphUnsupportedResult:
    status: str
    command: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"status": self.status, "command": self.command}
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True, slots=True)
class GraphCommandEnvelope:
    ok: bool
    command: str
    request_id: str
    pot_id: str | None
    graph_contract_version: str = GRAPH_WORKBENCH_CONTRACT_VERSION
    ontology_version: str = GRAPH_WORKBENCH_ONTOLOGY_VERSION
    subgraph_versions: Mapping[str, int] = field(default_factory=dict)
    result: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    unsupported: tuple[GraphUnsupported, ...] = ()
    recommended_next_action: str | Mapping[str, Any] | None = None
    error: GraphCommandError | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "command": self.command,
            "request_id": self.request_id,
            "pot_id": self.pot_id,
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
            "subgraph_versions": dict(self.subgraph_versions),
            "result": dict(self.result) if self.result is not None else None,
            "warnings": list(self.warnings),
            "unsupported": [u.to_dict() for u in self.unsupported],
            "recommended_next_action": (
                dict(self.recommended_next_action)
                if isinstance(self.recommended_next_action, Mapping)
                else self.recommended_next_action
            ),
        }
        if self.error is not None:
            out["error"] = self.error.to_dict()
        return out


__all__ = [
    "GRAPH_WORKBENCH_ADMIN_COMMANDS",
    "GRAPH_WORKBENCH_COMMANDS",
    "GRAPH_WORKBENCH_CONTRACT_VERSION",
    "GRAPH_WORKBENCH_LEGACY_COMMANDS",
    "GRAPH_WORKBENCH_ONTOLOGY_VERSION",
    "GraphCommandEnvelope",
    "GraphCommandError",
    "GraphRecommendedAction",
    "GraphUnsupported",
    "GraphUnsupportedResult",
    "GraphWorkbenchStatus",
]
