"""Executable Graph V2 ontology/workbench contract.

This module is the static contract layer above the V1.5 data plane. It turns
the existing view map, ontology catalogs, and mutation-operation partitions into
objects that ``graph catalog`` and ``graph describe`` can return directly.
Reader behavior still lives in :mod:`potpie_context_engine.application.services.graph_service`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from potpie_context_engine.domain.agent_context_port import CONTEXT_INCLUDE_VALUES
from potpie_context_engine.domain.graph_contract import (
    APPLICABLE_MUTATION_OPS,
    DEFERRED_OPS,
    KNOWN_MUTATION_OPS,
    ONTOLOGY_VERSION,
    REVIEW_REQUIRED_OPS,
    SOURCE_AUTHORITIES,
    STRONG_AUTHORITIES,
    TRUTH_CLASSES,
)
from potpie_context_engine.domain.graph_views import (
    GRAPH_VIEWS,
    GraphViewSpec,
    UnknownGraphViewError,
    include_guess_guidance,
)
from potpie_context_engine.domain.ontology import EDGE_TYPES, ENTITY_TYPES

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True, slots=True)
class ExampleCommand:
    command: str
    description: str
    json: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "description": self.description,
            "json": self.json,
        }


@dataclass(frozen=True, slots=True)
class IdentityPolicy:
    entity_type: str
    key_prefix: str
    identity_class: str
    identity_policy: str
    authoritative_source: str | None = None
    scope: bool = False
    canonical_prefix_rule: str = (
        "Entity-key prefixes are exact; underscore prefixes are canonical and "
        "hyphenated prefixes are not aliases."
    )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "entity_type": self.entity_type,
            "key_prefix": self.key_prefix,
            "identity_class": self.identity_class,
            "identity_policy": self.identity_policy,
            "scope": self.scope,
            "canonical_prefix_rule": self.canonical_prefix_rule,
        }
        if self.authoritative_source:
            out["authoritative_source"] = self.authoritative_source
        return out


@dataclass(frozen=True, slots=True)
class EntityTypeContract:
    label: str
    category: str
    description: str
    identity: IdentityPolicy
    patchable_properties: tuple[str, ...] = ()
    lifecycle_states: tuple[str, ...] = ()
    lifecycle_transitions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "identity": self.identity.to_dict(),
            "patchable_properties": list(self.patchable_properties),
            "lifecycle_states": list(self.lifecycle_states),
            "lifecycle_transitions": {
                key: list(values) for key, values in self.lifecycle_transitions.items()
            },
        }


@dataclass(frozen=True, slots=True)
class RelationTypeContract:
    name: str
    category: str
    description: str
    allowed_pairs: tuple[tuple[str, str], ...]
    required_properties: tuple[str, ...] = ()
    singleton: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "allowed_pairs": [list(pair) for pair in self.allowed_pairs],
            "required_properties": list(self.required_properties),
            "singleton": self.singleton,
        }


@dataclass(frozen=True, slots=True)
class SourceAuthorityPolicy:
    authority: str
    strength: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "strength": self.strength,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class MutationPolicy:
    operation: str
    availability: str
    description: str
    requires_review: bool = False
    applies_to_subgraphs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "availability": self.availability,
            "description": self.description,
            "requires_review": self.requires_review,
            "applies_to_subgraphs": list(self.applies_to_subgraphs),
        }


@dataclass(frozen=True, slots=True)
class ViewContract:
    name: str
    subgraph: str
    view: str
    purpose: str
    when_to_use: tuple[str, ...]
    v1_include: str
    backed: bool
    required_scope: tuple[str, ...] = ()
    required_any_scope: tuple[str, ...] = ()
    optional_scope: tuple[str, ...] = ()
    result_shape: str = "flat_claims"
    ranking_inputs: tuple[str, ...] = ()
    supported_filters: tuple[str, ...] = ()
    inline_relations: tuple[str, ...] = ()
    traversal: bool = False
    examples: tuple[ExampleCommand, ...] = ()
    keywords: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_examples: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "subgraph": self.subgraph,
            "view": self.view,
            "purpose": self.purpose,
            "when_to_use": list(self.when_to_use),
            "v1_include": self.v1_include,
            "backed": self.backed,
            "required_scope": list(self.required_scope),
            "required_any_scope": list(self.required_any_scope),
            "optional_scope": list(self.optional_scope),
            "result_shape": self.result_shape,
            "ranking_inputs": list(self.ranking_inputs),
            "supported_filters": list(self.supported_filters),
            "inline_relations": list(self.inline_relations),
            "traversal": self.traversal,
            "extra": dict(self.extra),
        }
        if include_examples:
            out["examples"] = [example.to_dict() for example in self.examples]
        else:
            out["example_count"] = len(self.examples)
        return out


@dataclass(frozen=True, slots=True)
class SubgraphContract:
    name: str
    purpose: str
    when_to_use: tuple[str, ...]
    entity_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    views: tuple[ViewContract, ...]
    keywords: tuple[str, ...] = ()
    examples: tuple[ExampleCommand, ...] = ()

    def to_dict(
        self,
        *,
        include_examples: bool = False,
        view_filter: str | None = None,
    ) -> dict[str, Any]:
        views = self.views
        if view_filter:
            views = tuple(view for view in views if view.name == view_filter)
        out: dict[str, Any] = {
            "name": self.name,
            "purpose": self.purpose,
            "when_to_use": list(self.when_to_use),
            "entity_types": [
                _entity_type_contract(label).to_dict()
                for label in self.entity_types
                if label in ENTITY_TYPES
            ],
            "relation_types": [
                _relation_type_contract(name).to_dict()
                for name in self.relation_types
                if name in EDGE_TYPES
            ],
            "views": [
                view.to_dict(include_examples=include_examples) for view in views
            ],
            "mutation_policies": [
                policy.to_dict() for policy in mutation_policies_for_subgraph(self.name)
            ],
            "source_authority_rules": [
                policy.to_dict() for policy in source_authority_policies()
            ],
            "truth_classes": list(TRUTH_CLASSES),
        }
        if include_examples:
            out["examples"] = [example.to_dict() for example in self.examples]
        else:
            out["example_count"] = len(self.examples)
        return out


@dataclass(frozen=True, slots=True)
class WorkbenchOntologyContract:
    ontology_version: str
    subgraphs: tuple[SubgraphContract, ...]
    mutation_policies: tuple[MutationPolicy, ...]
    source_authority_policies: tuple[SourceAuthorityPolicy, ...]
    identity_policies: tuple[IdentityPolicy, ...]

    def subgraph(self, name: str) -> SubgraphContract | None:
        normalized = (name or "").strip()
        return next((s for s in self.subgraphs if s.name == normalized), None)

    def view(self, name: str) -> ViewContract | None:
        normalized = (name or "").strip()
        return next(
            (
                view
                for subgraph in self.subgraphs
                for view in subgraph.views
                if view.name == normalized
            ),
            None,
        )

    def to_dict(self, *, include_examples: bool = False) -> dict[str, Any]:
        return {
            "ontology_version": self.ontology_version,
            "subgraphs": [
                subgraph.to_dict(include_examples=include_examples)
                for subgraph in self.subgraphs
            ],
            "mutation_policies": [
                policy.to_dict() for policy in self.mutation_policies
            ],
            "source_authority_rules": [
                policy.to_dict() for policy in self.source_authority_policies
            ],
            "identity_policies": [
                policy.to_dict() for policy in self.identity_policies
            ],
        }


_SUBGRAPH_DEFINITIONS: dict[str, dict[str, Any]] = {
    "debugging": {
        "purpose": "Recall prior failures, symptoms, attempted fixes, and verifications.",
        "when_to_use": (
            "Use before investigating bugs, failing tests, alerts, incidents, or regressions.",
        ),
        "entity_types": ("BugPattern", "Fix", "Activity", "Service", "CodeAsset"),
        "relation_types": (
            "REPRODUCES",
            "RESOLVED",
            "ATTEMPTED_FIX_FAILED",
            "VERIFIED",
        ),
        "keywords": (
            "debug",
            "debugging",
            "bug",
            "failure",
            "failing",
            "fix",
            "incident",
            "regression",
            "timeout",
            "error",
            "exception",
            "flaky",
            "test",
        ),
        "examples": (
            ExampleCommand(
                command='potpie graph read --subgraph debugging --view prior_occurrences --query "timeout" --json',
                description="Find prior bug/fix memories for a symptom.",
            ),
        ),
    },
    "recent_changes": {
        "purpose": "Read recent activity such as PRs, deploys, incidents, and decisions.",
        "when_to_use": (
            "Use when a task asks what changed recently or a regression may have followed a change.",
        ),
        "entity_types": (
            "Activity",
            "Period",
            "Person",
            "Team",
            "Service",
            "CodeAsset",
        ),
        "relation_types": ("TOUCHED", "PERFORMED", "AUTHORED", "IN_PERIOD", "MENTIONS"),
        "keywords": (
            "recent",
            "change",
            "changed",
            "after",
            "before",
            "deploy",
            "deployment",
            "rollback",
            "pr",
            "merged",
            "commit",
            "release",
            "timeline",
        ),
        "examples": (
            ExampleCommand(
                command="potpie graph read --subgraph recent_changes --view timeline --time-window 7d --json",
                description="Read the recent project timeline.",
            ),
        ),
    },
    "infra_topology": {
        "purpose": "Inspect services, environments, dependencies, adapters, config, and deployment edges.",
        "when_to_use": (
            "Use before changes that touch services, environments, integrations, dependencies, or deployment behavior.",
        ),
        "entity_types": (
            "Service",
            "Repository",
            "Environment",
            "DataStore",
            "Cluster",
            "Dependency",
            "APIContract",
            "Adapter",
            "ConfigVariable",
            "DeploymentTarget",
        ),
        "relation_types": (
            "DEFINED_IN",
            "DEPLOYED_TO",
            "DEPENDS_ON",
            "USES",
            "USES_ADAPTER",
            "CONFIGURES",
            "DEPLOYED_WITH",
            "EXPOSES",
            "HOSTED_ON",
            "OWNED_BY",
            "PROVIDES",
        ),
        "keywords": (
            "infra",
            "topology",
            "service",
            "dependency",
            "depends",
            "environment",
            "env",
            "staging",
            "production",
            "deploy",
            "deployment",
            "config",
            "adapter",
            "backend",
            "api",
        ),
        "examples": (
            ExampleCommand(
                command="potpie graph read --subgraph infra_topology --view service_neighborhood --scope service:payments-api --depth 2 --json",
                description="Read a bounded service neighborhood.",
            ),
        ),
    },
    "decisions": {
        "purpose": "Retrieve preferences, policies, and architectural decisions that apply to a scope.",
        "when_to_use": (
            "Use before writing code, reviewing risky changes, or checking policy and decision context.",
        ),
        "entity_types": (
            "Preference",
            "Policy",
            "Decision",
            "Repository",
            "Service",
            "CodeAsset",
        ),
        "relation_types": ("POLICY_APPLIES_TO", "DECIDED", "AFFECTS"),
        "keywords": (
            "decision",
            "decisions",
            "preference",
            "preferences",
            "policy",
            "standard",
            "guideline",
            "architecture",
            "choice",
            "risk",
            "debug",
            "deployment",
            "timeout",
        ),
        "examples": (
            ExampleCommand(
                command='potpie graph read --subgraph decisions --view preferences_for_scope --scope path:src/app.py --query "testing" --json',
                description="Read active preferences for a coding scope.",
            ),
        ),
    },
    "features": {
        "purpose": "Describe product or system capabilities a repo or service provides.",
        "when_to_use": (
            "Use when learning what a repository/service does or locating implementation context for a feature.",
        ),
        "entity_types": ("Feature", "Repository", "Service", "CodeAsset"),
        "relation_types": ("PROVIDES", "IMPLEMENTED_IN"),
        "keywords": (
            "feature",
            "capability",
            "product",
            "behavior",
            "implements",
            "functionality",
        ),
        "examples": (
            ExampleCommand(
                command="potpie graph read --subgraph features --view feature_context --scope repo:github.com/acme/app --json",
                description="Read capabilities provided by a repo.",
            ),
        ),
    },
    "code_topology": {
        "purpose": "Resolve ownership and code-anchor context.",
        "when_to_use": (
            "Use when routing code ownership questions or checking who owns a service/repo/path.",
        ),
        "entity_types": ("Repository", "Service", "CodeAsset", "Team", "Person"),
        "relation_types": ("OWNED_BY", "MEMBER_OF", "DEFINED_IN"),
        "keywords": ("owner", "ownership", "team", "person", "code", "path", "repo"),
        "examples": (
            ExampleCommand(
                command="potpie graph read --subgraph code_topology --view ownership_by_path --scope path:src/app.py --json",
                description="Read owners for a path or service scope.",
            ),
        ),
    },
    "knowledge": {
        "purpose": "Locate durable documentation and notes for a scope.",
        "when_to_use": (
            "Use when looking for docs, runbooks, design notes, or written references.",
        ),
        "entity_types": (
            "Document",
            "Observation",
            "Repository",
            "Service",
            "CodeAsset",
        ),
        "relation_types": ("RELATED_TO", "MENTIONS", "AFFECTS"),
        "keywords": (
            "doc",
            "docs",
            "document",
            "runbook",
            "note",
            "reference",
            "knowledge",
        ),
        "examples": (
            ExampleCommand(
                command="potpie graph read --subgraph knowledge --view document_context --scope service:payments-api --json",
                description="Read documentation pointers for a scope.",
            ),
        ),
    },
    "admin": {
        "purpose": "Operator-only inspection of the canonical graph slice.",
        "when_to_use": (
            "Use for graph explorer/admin inspection, not ordinary agent memory reads.",
        ),
        "entity_types": tuple(ENTITY_TYPES),
        "relation_types": tuple(EDGE_TYPES),
        "keywords": ("admin", "inspect", "raw", "graph", "operator", "visualization"),
        "examples": (
            ExampleCommand(
                command="potpie graph read --subgraph admin --view inspection_slice --json",
                description="Read an operator inspection slice.",
            ),
        ),
    },
}

_VIEW_OVERRIDES: dict[str, dict[str, Any]] = {
    "decisions.preferences_for_scope": {
        "purpose": "Return active preferences and policy claims that apply to a repo, service, path, or code asset.",
        "when_to_use": (
            "Use before writing or reviewing code so local project preferences are visible.",
        ),
        "result_shape": "entity_relations",
        "required_any_scope": ("repo", "scope", "service", "path", "query", "language"),
        "optional_scope": (
            "repo",
            "scope",
            "service",
            "path",
            "file_path",
            "language",
            "framework",
            "query",
        ),
        "supported_filters": (
            "repo",
            "scope",
            "service",
            "path",
            "file_path",
            "language",
            "framework",
            "audience",
            "query",
        ),
        "keywords": ("preference", "policy", "scope", "coding", "style"),
        "examples": (
            ExampleCommand(
                command='potpie graph read --subgraph decisions --view preferences_for_scope --scope path:potpie/context-engine --query "testing" --json',
                description="Read scoped coding preferences.",
            ),
        ),
    },
    "debugging.prior_occurrences": {
        "purpose": "Return prior matching bug patterns with fix and verification relations inlined.",
        "when_to_use": (
            "Use before debugging a symptom so old fixes and failed attempts are not missed.",
        ),
        "result_shape": "entity_relations",
        "required_any_scope": ("query", "service", "repo"),
        "optional_scope": ("query", "service", "repo", "time_window"),
        "supported_filters": (
            "query",
            "service",
            "repo",
            "since",
            "until",
            "time_window",
            "path",
            "file_path",
        ),
        "keywords": ("bug", "debug", "failure", "symptom", "fix", "timeout"),
        "examples": (
            ExampleCommand(
                command='potpie graph read --subgraph debugging --view prior_occurrences --query "staging timeout" --json',
                description="Read prior failures matching a symptom.",
            ),
        ),
    },
    "recent_changes.timeline": {
        "purpose": "Return recent project activity ordered by event time.",
        "when_to_use": (
            "Use when diagnosing regressions or checking what changed in a time window.",
        ),
        "result_shape": "events",
        "optional_scope": ("scope", "time_window", "query"),
        "supported_filters": (
            "scope",
            "query",
            "since",
            "until",
            "time_window",
            "service",
            "repo",
            "path",
            "file_path",
        ),
        "keywords": ("timeline", "recent", "change", "deploy", "merged", "after"),
        "examples": (
            ExampleCommand(
                command="potpie graph read --subgraph recent_changes --view timeline --time-window 7d --json",
                description="Read recent activity.",
            ),
        ),
    },
    "infra_topology.service_neighborhood": {
        "purpose": "Return a bounded service neighborhood with dependency, deployment, config, ownership, and adapter edges.",
        "when_to_use": (
            "Use before touching service boundaries, deployment behavior, dependencies, adapters, or environment-specific config.",
        ),
        "result_shape": "entity_relations",
        "required_any_scope": ("service", "anchor_entity_key"),
        "optional_scope": (
            "service",
            "depth",
            "direction",
            "environment",
            "include_unqualified_environment",
            "query",
        ),
        "supported_filters": (
            "service",
            "anchor_entity_key",
            "depth",
            "direction",
            "environment",
            "include_unqualified_environment",
            "query",
        ),
        "keywords": (
            "service",
            "dependency",
            "environment",
            "staging",
            "deploy",
            "config",
        ),
        "examples": (
            ExampleCommand(
                command="potpie graph read --subgraph infra_topology --view service_neighborhood --scope service:payments-api --environment staging --depth 2 --json",
                description="Read a service neighborhood in one environment.",
            ),
        ),
    },
    "features.feature_context": {
        "purpose": "Return capabilities a repository or service provides and where they are implemented.",
        "when_to_use": (
            "Use to learn what a repo/service does or to locate feature implementation anchors.",
        ),
        "result_shape": "entity_relations",
        "required_any_scope": (
            "scope",
            "service",
            "repo",
            "anchor_entity_key",
            "query",
        ),
        "optional_scope": ("scope", "service", "query"),
        "supported_filters": (
            "scope",
            "service",
            "repo",
            "anchor_entity_key",
            "query",
        ),
        "keywords": ("feature", "capability", "implements", "repo", "service"),
    },
    "decisions.active_decisions": {
        "purpose": "Return active architectural decisions affecting a scope.",
        "when_to_use": (
            "Use before architecture-sensitive work, migrations, or behavior changes.",
        ),
        "result_shape": "flat_claims",
        "required_any_scope": ("scope", "query", "service", "repo", "path"),
        "optional_scope": ("scope", "query"),
        "supported_filters": ("scope", "query", "service", "repo", "path", "file_path"),
        "keywords": ("decision", "architecture", "adr", "choice", "migration"),
    },
    "code_topology.ownership_by_path": {
        "purpose": "Return service/repo/path ownership and team context.",
        "when_to_use": (
            "Use before routing reviews, asking owners, or changing owned code.",
        ),
        "result_shape": "entity_relations",
        "required_any_scope": ("scope", "path", "repo", "service", "anchor_entity_key"),
        "optional_scope": ("scope",),
        "supported_filters": (
            "scope",
            "path",
            "file_path",
            "repo",
            "service",
            "anchor_entity_key",
        ),
        "keywords": ("owner", "ownership", "team", "path", "repo"),
    },
    "knowledge.document_context": {
        "purpose": "Return documentation pointers and reference notes for a scope.",
        "when_to_use": (
            "Use when source docs or runbooks may hold the authoritative answer.",
        ),
        "result_shape": "flat_claims",
        "required_any_scope": ("scope", "query", "service", "repo", "path"),
        "optional_scope": ("scope", "query"),
        "supported_filters": ("scope", "query", "service", "repo", "path", "file_path"),
        "keywords": ("docs", "document", "runbook", "reference", "note"),
    },
    "admin.inspection_slice": {
        "purpose": "Return the raw canonical graph slice for operator inspection.",
        "when_to_use": (
            "Use for admin/debug visualization, not for ordinary task context.",
        ),
        "result_shape": "raw_graph",
        "optional_scope": (),
        "supported_filters": (),
        "keywords": ("admin", "raw", "inspect", "graph"),
    },
}

_SUBGRAPH_TIE_BREAKER: dict[str, int] = {
    "debugging": 0,
    "recent_changes": 1,
    "infra_topology": 2,
    "decisions": 3,
    "features": 4,
    "code_topology": 5,
    "knowledge": 6,
    "admin": 7,
}


def ontology_contract() -> WorkbenchOntologyContract:
    return _ONTOLOGY_CONTRACT


def _unknown_describe_error(
    message: str, *, subgraph: str, view: str | None
) -> UnknownGraphViewError:
    guidance = include_guess_guidance(subgraph, view)
    try_command = None
    if guidance:
        try_command = (
            f"potpie graph describe {guidance['subgraph']} "
            f"--view {guidance['view_name']}"
        )
        if guidance.get("matched_include"):
            message += (
                f" The context include family {guidance['matched_include']!r} is "
                f"served by view {guidance['view']!r}; try `{try_command}`."
            )
        else:
            message += f" Did you mean {guidance['view']!r}? Try `{try_command}`."
    return UnknownGraphViewError(
        message,
        did_you_mean=guidance,
        recommended_next_action=try_command,
    )


def describe_contract(
    *,
    subgraph: str,
    view: str | None = None,
    include_examples: bool = False,
) -> dict[str, Any]:
    """Return the executable contract for a subgraph or one of its views."""
    contract = ontology_contract()
    subgraph_name = (subgraph or "").strip()
    if not subgraph_name:
        raise ValueError("subgraph is required")
    subgraph_contract = contract.subgraph(subgraph_name)
    if subgraph_contract is None:
        known = ", ".join(sorted(s.name for s in contract.subgraphs))
        raise _unknown_describe_error(
            f"unknown graph subgraph {subgraph_name!r}. Known subgraphs: {known}.",
            subgraph=subgraph_name,
            view=view,
        )

    view_name = _resolve_view_name(subgraph_name, view)
    view_contract = None
    if view_name:
        view_contract = contract.view(view_name)
        if view_contract is None or view_contract.subgraph != subgraph_name:
            known = ", ".join(sorted(v.name for v in subgraph_contract.views))
            raise _unknown_describe_error(
                f"unknown graph view {view_name!r} for subgraph {subgraph_name!r}. "
                f"Known views: {known}.",
                subgraph=subgraph_name,
                view=view,
            )

    payload = {
        "contract_kind": "graph_workbench_ontology",
        "ontology_version": contract.ontology_version,
        "subgraph": subgraph_contract.to_dict(
            include_examples=include_examples,
            view_filter=view_contract.name if view_contract else None,
        ),
    }
    if view_contract is not None:
        payload["view"] = view_contract.to_dict(include_examples=include_examples)
    return payload


def rank_views_for_task(
    task: str | None,
    *,
    views: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Deterministically rank view contracts for a task string."""
    task_tokens = _tokens(task or "")
    if not task_tokens:
        return []
    selected_views = list(_catalog_view_entries() if views is None else views)
    by_name = {
        view.name: view
        for subgraph in _ONTOLOGY_CONTRACT.subgraphs
        for view in subgraph.views
    }
    ranked: list[dict[str, Any]] = []
    for index, entry in enumerate(selected_views):
        name = str(entry.get("name") or "")
        view = by_name.get(name)
        if view is None:
            continue
        score, matched = _score_view_for_task(view, task_tokens)
        ranked.append(
            {
                "view": view.name,
                "subgraph": view.subgraph,
                "score": score,
                "matched_terms": sorted(matched),
                "reason": _rank_reason(view, matched),
                "_index": index,
            }
        )
    ranked.sort(
        key=lambda item: (
            -int(item["score"]),
            _SUBGRAPH_TIE_BREAKER.get(str(item["subgraph"]), 99),
            int(item["_index"]),
            str(item["view"]),
        )
    )
    for item in ranked:
        item.pop("_index", None)
    return ranked


def ranked_catalog_views(
    views: Sequence[Mapping[str, Any]],
    task: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``views`` sorted by task ranking plus the ranking explanation."""
    view_entries = [dict(view) for view in views]
    ranking = rank_views_for_task(task, views=view_entries)
    if not ranking:
        return view_entries, []
    order = {str(entry["view"]): index for index, entry in enumerate(ranking)}
    view_entries.sort(key=lambda view: order.get(str(view.get("name")), len(order)))
    return view_entries, ranking


def mutation_policies_for_subgraph(subgraph: str) -> tuple[MutationPolicy, ...]:
    return tuple(
        replace(policy, applies_to_subgraphs=(subgraph,))
        for policy in _MUTATION_POLICIES
    )


def source_authority_policies() -> tuple[SourceAuthorityPolicy, ...]:
    return _SOURCE_AUTHORITY_POLICIES


def assert_ontology_contract_coherent(
    contract: WorkbenchOntologyContract | None = None,
) -> None:
    errors = _ontology_contract_errors(contract or _ONTOLOGY_CONTRACT)
    if errors:
        raise RuntimeError(
            "graph_workbench_ontology incoherent:\n  - " + "\n  - ".join(errors)
        )


def _build_contract() -> WorkbenchOntologyContract:
    views_by_subgraph: dict[str, list[ViewContract]] = {
        name: [] for name in _SUBGRAPH_DEFINITIONS
    }
    for spec in GRAPH_VIEWS.values():
        views_by_subgraph.setdefault(spec.subgraph, []).append(_view_contract(spec))

    subgraphs: list[SubgraphContract] = []
    for name, data in _SUBGRAPH_DEFINITIONS.items():
        subgraphs.append(
            SubgraphContract(
                name=name,
                purpose=str(data["purpose"]),
                when_to_use=tuple(data["when_to_use"]),
                entity_types=tuple(data["entity_types"]),
                relation_types=tuple(data["relation_types"]),
                views=tuple(
                    sorted(views_by_subgraph.get(name, ()), key=lambda v: v.name)
                ),
                keywords=tuple(data["keywords"]),
                examples=tuple(data["examples"]),
            )
        )

    return WorkbenchOntologyContract(
        ontology_version=ONTOLOGY_VERSION,
        subgraphs=tuple(subgraphs),
        mutation_policies=_MUTATION_POLICIES,
        source_authority_policies=_SOURCE_AUTHORITY_POLICIES,
        identity_policies=_IDENTITY_POLICIES,
    )


def _view_contract(spec: GraphViewSpec) -> ViewContract:
    override = _VIEW_OVERRIDES.get(spec.name, {})
    optional_scope = tuple(override.get("optional_scope", spec.inputs))
    supported_filters = tuple(override.get("supported_filters", spec.inputs))
    if spec.backed and "source_ref" not in supported_filters:
        supported_filters = (*supported_filters, "source_ref")
    if spec.backed and "source_ref" not in optional_scope:
        optional_scope = (*optional_scope, "source_ref")
    return ViewContract(
        name=spec.name,
        subgraph=spec.subgraph,
        view=spec.view,
        purpose=str(override.get("purpose") or spec.description),
        when_to_use=tuple(override.get("when_to_use") or (spec.description,)),
        v1_include=spec.v1_include,
        backed=spec.backed,
        required_scope=tuple(override.get("required_scope") or ()),
        required_any_scope=tuple(override.get("required_any_scope") or ()),
        optional_scope=optional_scope,
        result_shape=str(override.get("result_shape") or "flat_claims"),
        ranking_inputs=tuple(spec.ranking_inputs),
        supported_filters=supported_filters,
        inline_relations=tuple(spec.inline_relations),
        traversal=spec.traversal,
        examples=tuple(override.get("examples") or ()),
        keywords=tuple(override.get("keywords") or ()),
        extra=dict(spec.extra),
    )


def _identity_policy(label: str) -> IdentityPolicy:
    spec = ENTITY_TYPES[label]
    identity_class = getattr(spec.identity_class, "value", str(spec.identity_class))
    return IdentityPolicy(
        entity_type=label,
        key_prefix=spec.key_prefix,
        identity_class=str(identity_class),
        identity_policy=spec.identity_policy,
        authoritative_source=spec.authoritative_source,
        scope=spec.scope,
    )


def _entity_type_contract(label: str) -> EntityTypeContract:
    spec = ENTITY_TYPES[label]
    return EntityTypeContract(
        label=label,
        category=spec.category,
        description=spec.description,
        identity=_identity_policy(label),
        patchable_properties=tuple(sorted(spec.patchable_properties)),
        lifecycle_states=tuple(sorted(spec.lifecycle_states)),
        lifecycle_transitions={
            key: tuple(sorted(values))
            for key, values in sorted(spec.lifecycle_transitions.items())
        },
    )


def _relation_type_contract(name: str) -> RelationTypeContract:
    spec = EDGE_TYPES[name]
    return RelationTypeContract(
        name=name,
        category=spec.category,
        description=spec.description,
        allowed_pairs=tuple(spec.allowed_pairs),
        required_properties=tuple(sorted(spec.required_properties)),
        singleton=spec.singleton,
    )


def _resolve_view_name(subgraph: str, view: str | None) -> str | None:
    if not view:
        return None
    value = view.strip()
    if not value:
        return None
    return value if "." in value else f"{subgraph}.{value}"


def _catalog_view_entries() -> list[dict[str, Any]]:
    return [
        view.to_dict(include_examples=False)
        for subgraph in _ONTOLOGY_CONTRACT.subgraphs
        for view in subgraph.views
    ]


def _score_view_for_task(
    view: ViewContract, task_tokens: set[str]
) -> tuple[int, set[str]]:
    subgraph_keywords = set(_SUBGRAPH_DEFINITIONS[view.subgraph]["keywords"])
    view_keywords = set(view.keywords)
    description_tokens = _tokens(
        " ".join(
            (
                view.name,
                view.purpose,
                " ".join(view.when_to_use),
                " ".join(view.inline_relations),
            )
        )
    )
    matched: set[str] = set()
    score = 0
    for token in task_tokens:
        if _matches_any(token, view_keywords):
            score += 6
            matched.add(token)
            continue
        if _matches_any(token, subgraph_keywords):
            score += 3
            matched.add(token)
            continue
        if _matches_any(token, description_tokens):
            score += 1
            matched.add(token)
    if matched and view.backed:
        score += 1
    if view.subgraph == "admin":
        score -= 2
    return max(score, 0), matched


def _rank_reason(view: ViewContract, matched: set[str]) -> str:
    if matched:
        return (
            f"Matched {', '.join(sorted(matched))} against the "
            f"{view.subgraph} contract and {view.name} view."
        )
    return (
        f"No direct task keyword match; kept after more relevant {view.subgraph} views."
    )


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}


def _matches_any(token: str, candidates: set[str]) -> bool:
    if token in candidates:
        return True
    return any(
        (len(token) >= 5 and candidate.startswith(token))
        or (len(candidate) >= 5 and token.startswith(candidate))
        for candidate in candidates
    )


def _ontology_contract_errors(contract: WorkbenchOntologyContract) -> list[str]:
    errors: list[str] = []
    policy_by_op = {policy.operation: policy for policy in contract.mutation_policies}
    for subgraph in contract.subgraphs:
        for label in subgraph.entity_types:
            if label not in ENTITY_TYPES:
                errors.append(
                    f"subgraph {subgraph.name!r} references unknown entity type {label!r}"
                )
        for relation in subgraph.relation_types:
            if relation not in EDGE_TYPES:
                errors.append(
                    f"subgraph {subgraph.name!r} references unknown relation type {relation!r}"
                )
        for view in subgraph.views:
            if view.v1_include not in CONTEXT_INCLUDE_VALUES:
                errors.append(
                    f"view {view.name!r} routes to unsupported include {view.v1_include!r}"
                )
            for relation in view.inline_relations:
                if relation not in EDGE_TYPES:
                    errors.append(
                        f"view {view.name!r} inlines unknown relation {relation!r}"
                    )
            if view.subgraph != subgraph.name:
                errors.append(
                    f"view {view.name!r} is attached to subgraph {subgraph.name!r} "
                    f"but declares {view.subgraph!r}"
                )

    for op in APPLICABLE_MUTATION_OPS:
        if policy_by_op.get(op, None) is None:
            errors.append(f"applicable mutation op {op!r} has no policy")
        elif policy_by_op[op].availability != "applicable":
            errors.append(
                f"applicable mutation op {op!r} has wrong policy availability"
            )
    for op in REVIEW_REQUIRED_OPS:
        if policy_by_op.get(op, None) is None:
            errors.append(f"review-required mutation op {op!r} has no policy")
        elif policy_by_op[op].availability != "review_required":
            errors.append(
                f"review-required mutation op {op!r} has wrong policy availability"
            )
    for op in DEFERRED_OPS:
        if policy_by_op.get(op, None) is None:
            errors.append(f"deferred mutation op {op!r} has no policy")
        elif policy_by_op[op].availability != "deferred":
            errors.append(f"deferred mutation op {op!r} has wrong policy availability")
    for policy in contract.mutation_policies:
        if policy.operation not in KNOWN_MUTATION_OPS:
            errors.append(f"mutation policy advertises unknown op {policy.operation!r}")
    return errors


def _mutation_policy(
    operation: str,
    *,
    availability: str,
    description: str,
    requires_review: bool = False,
) -> MutationPolicy:
    return MutationPolicy(
        operation=operation,
        availability=availability,
        description=description,
        requires_review=requires_review,
    )


_MUTATION_POLICIES: tuple[MutationPolicy, ...] = (
    tuple(
        _mutation_policy(
            op,
            availability="applicable",
            description="Accepted by the semantic validator and directly applicable through the V1.5 write door when risk policy allows.",
        )
        for op in APPLICABLE_MUTATION_OPS
    )
    + tuple(
        _mutation_policy(
            op,
            availability="review_required",
            description="Recognized by the semantic validator but requires review; V1.5 has no auto-apply path.",
            requires_review=True,
        )
        for op in REVIEW_REQUIRED_OPS
    )
    + tuple(
        _mutation_policy(
            op,
            availability="deferred",
            description="Reserved vocabulary; not applied by the current V1.5 data plane.",
            requires_review=True,
        )
        for op in DEFERRED_OPS
    )
)

_SOURCE_AUTHORITY_POLICIES: tuple[SourceAuthorityPolicy, ...] = tuple(
    SourceAuthorityPolicy(
        authority=authority,
        strength="strong" if authority in STRONG_AUTHORITIES else "supporting",
        description=(
            "Can satisfy durable-write evidence requirements."
            if authority in STRONG_AUTHORITIES
            else "Useful as supporting evidence, but not strong enough alone for objective durable facts."
        ),
    )
    for authority in sorted(SOURCE_AUTHORITIES)
)

_IDENTITY_POLICIES: tuple[IdentityPolicy, ...] = tuple(
    _identity_policy(label) for label, spec in ENTITY_TYPES.items() if spec.public
)

_ONTOLOGY_CONTRACT = _build_contract()
assert_ontology_contract_coherent(_ONTOLOGY_CONTRACT)


__all__ = [
    "EntityTypeContract",
    "ExampleCommand",
    "IdentityPolicy",
    "MutationPolicy",
    "RelationTypeContract",
    "SourceAuthorityPolicy",
    "SubgraphContract",
    "ViewContract",
    "WorkbenchOntologyContract",
    "assert_ontology_contract_coherent",
    "describe_contract",
    "ontology_contract",
    "rank_views_for_task",
    "ranked_catalog_views",
]
