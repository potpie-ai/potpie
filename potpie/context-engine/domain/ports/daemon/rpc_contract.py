"""Daemon RPC contract owned by the context-engine HostShell boundary."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Final, Mapping


RPC_DTO_MODULES: Final[frozenset[str]] = frozenset(
    {
        "domain.actor",
        "domain.agent_envelope",
        "domain.context_events",
        "domain.context_records",
        "domain.context_status",
        "domain.event_playbooks",
        "domain.graph_contract",
        "domain.graph_history",
        "domain.graph_inbox",
        "domain.graph_mutations",
        "domain.graph_plans",
        "domain.graph_quality",
        "domain.graph_query",
        "domain.graph_views",
        "domain.graph_workbench",
        "domain.graph_workbench_ontology",
        "domain.identity",
        "domain.ingestion_event_models",
        "domain.lifecycle",
        "domain.llm_reconciliation",
        "domain.nudge",
        "domain.ontology",
        "domain.ontology_classifier",
        "domain.ports.agent_context",
        "domain.ports.agent_execution_log",
        "domain.ports.claim_query",
        "domain.ports.daemon.operations",
        "domain.ports.daemon.service",
        "domain.ports.daemon.shell",
        "domain.ports.graph.analytics",
        "domain.ports.graph.backend",
        "domain.ports.graph.inspection",
        "domain.ports.graph.mutation",
        "domain.ports.graph.snapshot",
        "domain.ports.ingestion_config",
        "domain.ports.ingestion_ledger",
        "domain.ports.ledger.client",
        "domain.ports.policy",
        "domain.ports.pot_resolution",
        "domain.ports.reconciliation_ledger",
        "domain.ports.services.auth",
        "domain.ports.services.graph_service",
        "domain.ports.services.pot_management",
        "domain.ports.services.skill_manager",
        "domain.ports.telemetry",
        "domain.ranking",
        "domain.reconciliation",
        "domain.reconciliation_batch",
        "domain.semantic_mutations",
        "domain.source_connector",
        "domain.source_references",
        "domain.source_resolution",
    }
)


def _set(*items: str) -> frozenset[str]:
    return frozenset(items)


@dataclass(frozen=True, slots=True)
class RpcSurfaceSpec:
    """Explicit daemon RPC contract for one remotely exposed host surface."""

    methods: frozenset[str] = frozenset()
    attrs: frozenset[str] = frozenset()
    children: Mapping[str, str] = field(default_factory=dict)


RPC_SURFACES: Mapping[str, RpcSurfaceSpec] = {
    "agent_context": RpcSurfaceSpec(
        methods=_set("record", "resolve", "search", "status"),
    ),
    "auth": RpcSurfaceSpec(methods=_set("whoami")),
    "backend": RpcSurfaceSpec(
        methods=_set("capabilities", "provision"),
        attrs=_set("profile"),
        children={
            "analytics": "backend.analytics",
            "claim_query": "backend.claim_query",
            "inspection": "backend.inspection",
            "mutation": "backend.mutation",
            "semantic": "backend.semantic",
            "snapshot": "backend.snapshot",
        },
    ),
    "backend.analytics": RpcSurfaceSpec(
        methods=_set("counts", "freshness", "quality", "repair"),
    ),
    "backend.claim_query": RpcSurfaceSpec(
        methods=_set("entity_labels", "entity_properties", "find_claims"),
    ),
    "backend.inspection": RpcSurfaceSpec(
        methods=_set("labels", "neighborhood", "path", "slice"),
    ),
    "backend.mutation": RpcSurfaceSpec(
        methods=_set("apply", "invalidate", "readiness", "reset_pot"),
    ),
    "backend.semantic": RpcSurfaceSpec(methods=_set("search")),
    "backend.snapshot": RpcSurfaceSpec(methods=_set("export", "import_")),
    "config": RpcSurfaceSpec(methods=_set("get", "set")),
    "graph": RpcSurfaceSpec(
        methods=_set(
            "catalog",
            "data_plane_status",
            "mutate",
            "read",
            "resolve",
            "search",
            "search_entities",
            "record",
        ),
    ),
    "graph_workbench": RpcSurfaceSpec(
        methods=_set(
            "commit",
            "history",
            "inbox_add",
            "inbox_claim",
            "inbox_close",
            "inbox_list",
            "inbox_mark_applied",
            "inbox_mark_rejected",
            "inbox_show",
            "propose",
            "quality",
        ),
    ),
    "installer": RpcSurfaceSpec(methods=_set("ensure_cli", "install", "status")),
    "ledger": RpcSurfaceSpec(methods=_set("pull", "query", "sources", "status")),
    "nudge": RpcSurfaceSpec(methods=_set("nudge")),
    "pots": RpcSurfaceSpec(
        methods=_set(
            "active_pot",
            "add_source",
            "aggregate_status",
            "archive_pot",
            "clear_repo_default",
            "create_pot",
            "init",
            "list_pots",
            "list_repo_defaults",
            "list_sources",
            "remove_source",
            "rename_pot",
            "repo_default",
            "reset_pot",
            "set_repo_default",
            "source_status",
            "use_pot",
        ),
    ),
    "setup": RpcSurfaceSpec(methods=_set("plan", "preview", "run")),
    "skills": RpcSurfaceSpec(
        methods=_set("add", "install", "list", "nudge", "remove", "status", "update"),
    ),
}


def class_ref(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def assert_rpc_class_allowed(cls: type) -> None:
    ref = class_ref(cls)
    if not _is_rpc_dto_class(cls):
        raise TypeError(f"RPC class not allowed: {ref}")


def load_rpc_class(ref: str) -> type:
    module_name, qualname = _split_class_ref(ref)
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise TypeError(f"RPC class not allowed: {ref}") from exc
    if not isinstance(obj, type) or class_ref(obj) != ref or not _is_rpc_dto_class(obj):
        raise TypeError(f"RPC class not allowed: {ref}")
    return obj


def _split_class_ref(ref: str) -> tuple[str, str]:
    try:
        module_name, qualname = ref.split(":", 1)
    except ValueError as exc:
        raise TypeError(f"RPC class not allowed: {ref}") from exc
    if module_name not in RPC_DTO_MODULES or not qualname:
        raise TypeError(f"RPC class not allowed: {ref}")
    return module_name, qualname


def _is_rpc_dto_class(cls: type) -> bool:
    if cls.__module__ not in RPC_DTO_MODULES:
        return False
    return is_dataclass(cls) or issubclass(cls, Enum)


__all__ = [
    "RPC_DTO_MODULES",
    "RPC_SURFACES",
    "RpcSurfaceSpec",
    "assert_rpc_class_allowed",
    "class_ref",
    "load_rpc_class",
]
