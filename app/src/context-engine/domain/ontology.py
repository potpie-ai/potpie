"""Canonical Potpie context graph ontology — v2, extensibility-first.

The ontology is a *declarative* catalog. Each entity and edge type carries the
metadata its downstream consumers need (project-map family, fact family,
source-of-truth policy, freshness TTL, classifier text cues, property
signatures). All other modules — the classifier, structural reader, hybrid
graph, graph-quality policy, query helpers — derive their behavior from this
catalog rather than hardcoding label strings.

**Adding or renaming an entity:** edit a single row in :data:`ENTITY_TYPES`.
**Adding an edge:** edit a single row in :data:`EDGE_TYPES`.

Design pillars:

1. **Intent / state / evidence** are three orthogonal axes. Each fact carries
   ``confidence`` and ``evidence_strength`` properties so agents can weight
   deterministic facts (a merged PR) above hypotheses (an LLM extraction over
   chat).
2. **Time is a property of facts, not a side subgraph.** Lifecycle and
   validity belong on edges. The :class:`Activity` label is the unified
   timeline view; rich change events (``PullRequest``, ``Commit``,
   ``Deployment``, ``Incident``, ``Alert``, ``Release``, ``Migration``,
   ``Issue``) carry the ``Activity`` label themselves rather than spawning a
   shadow node.
3. **Wildcard edges are an admission of failure** and are reserved for
   genuinely polymorphic predicates (evidence, identity, supersession,
   lifecycle). Everything else gets typed endpoint pairs the validator can
   enforce.
4. **A ``Scope`` interface, not a base class.** Entities flagged
   ``scope=True`` (``Pot``, ``Repository``, ``System``, ``Service``,
   ``Environment``, ``Component``, ``CodeAsset``) act as scope endpoints for
   ``APPLIES_TO`` / ``CONFIGURED_BY`` / ``INFORMS`` without each edge having
   to enumerate them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from domain.graph_mutations import EdgeDelete, EdgeUpsert, EntityUpsert
from domain.source_references import validate_source_reference_properties

ONTOLOGY_VERSION = "2026-05-phase-9-extensibility"


# --- Lifecycle vocabulary ---------------------------------------------------


class LifecycleStatus(StrEnum):
    """Edge-level lifecycle for episodic facts (``lifecycle_status`` on edge props)."""

    proposed = "proposed"
    planned = "planned"
    in_progress = "in_progress"
    completed = "completed"
    deprecated = "deprecated"
    decommissioned = "decommissioned"
    unknown = "unknown"


COMMON_LIFECYCLE_STATES = frozenset(
    {"proposed", "active", "deprecated", "retired", "unknown"}
)
DECISION_STATES = frozenset(
    {"proposed", "accepted", "superseded", "rejected", "unknown"}
)
ISSUE_STATES = frozenset({"open", "closed", "triaged", "blocked", "unknown"})
INCIDENT_STATES = frozenset({"open", "mitigated", "resolved", "postmortem", "unknown"})
RISK_STATES = frozenset({"open", "mitigated", "accepted", "materialized", "unknown"})
QUESTION_STATES = frozenset({"open", "resolved", "abandoned", "unknown"})
MIGRATION_STATES = frozenset(
    {
        "planned",
        "in_progress",
        "backfilling",
        "cutover",
        "completed",
        "rolled_back",
        "unknown",
    }
)
FLAG_STATES = frozenset({"off", "ramping", "on", "deprecated", "unknown"})


# --- Source-of-truth families ----------------------------------------------
# Used by graph-quality policy. Each entity declares which family it belongs
# to via ``fact_family``; the freshness TTL and source-of-truth policy are
# looked up here.

SOURCE_OF_TRUTH_AUTHORITATIVE_EXTERNAL = "authoritative_external_truth"
SOURCE_OF_TRUTH_AUTHORITATIVE_CODE = "authoritative_code_truth"
SOURCE_OF_TRUTH_CANONICAL_MEMORY = "canonicalized_memory"
SOURCE_OF_TRUTH_SOFT_INFERENCE = "soft_inference"


# --- Evidence strength vocabulary ------------------------------------------
# Carried on entity/edge properties so agents can weight facts. Modules that
# rank or filter facts should consult these.

EVIDENCE_STRENGTHS = ("deterministic", "attested", "inferred", "hypothesized")
DEFAULT_EVIDENCE_STRENGTH = "inferred"


# --- Graph label / wildcard constants --------------------------------------

BASE_GRAPH_LABELS = frozenset({"Entity"})
# Legacy code-graph labels still in the data; ``CodeAsset`` is the canonical
# bridge. The validator treats any of these as a CodeAsset endpoint.
CODE_GRAPH_LABELS = frozenset({"CodeAsset", "FILE", "FUNCTION", "CLASS", "NODE"})
WILDCARD_ENDPOINT = "*"
SCOPE_ENDPOINT = "@Scope"  # resolves to any entity flagged ``scope=True``
ACTIVITY_ENDPOINT = "@Activity"  # resolves to Activity or any entity flagged ``activity=True``


# --- Spec dataclasses -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntityTypeSpec:
    """Declarative metadata for one canonical entity label.

    Every downstream consumer derives its behavior from these fields rather
    than hardcoding label strings. To add a new entity, append a new spec to
    :data:`ENTITY_TYPES`.
    """

    label: str
    category: str
    description: str
    identity_policy: str
    required_properties: frozenset[str] = frozenset()
    lifecycle_states: frozenset[str] = frozenset()
    public: bool = True

    # --- Structural traits ---------------------------------------------------
    scope: bool = False
    """True for Pot/Repo/Service/Env/Component/CodeAsset — endpoints of APPLIES_TO etc."""

    is_activity: bool = False
    """True for PullRequest/Commit/Deployment/etc. — multi-labels with Activity for timeline."""

    # --- Agent-facing family mapping ----------------------------------------
    project_map_family: str | None = None
    """Which family this label answers in the project_map response (service_map, deployments, …)."""

    debugging_family: str | None = None
    """Which family in the debugging_memory response (prior_fixes, incidents, alerts, …)."""

    include_keys: tuple[str, ...] = ()
    """Agent-contract ``include`` keys that should surface this label.

    Stable contract: scenarios and the public include set are keyed by these
    strings (``preferences``, ``diagnostic_signals``, etc). Internal labels
    can change without breaking the agent contract by remapping families."""

    # --- Source-of-truth / freshness ---------------------------------------
    fact_family: str = "unknown"
    """Identifier for the SoT family (ownership, change, decision, …)."""

    source_of_truth: str = SOURCE_OF_TRUTH_CANONICAL_MEMORY
    freshness_ttl_hours: int = 24 * 30

    # --- Classification cues (used by ontology_classifier) -----------------
    text_patterns: tuple[str, ...] = ()
    """Regex patterns over name/title/summary/statement/fact/rationale that classify text → this label."""

    property_signatures: tuple[str, ...] = ()
    """Property names whose non-empty string value forces this label."""


@dataclass(frozen=True, slots=True)
class EdgeTypeSpec:
    edge_type: str
    description: str
    allowed_pairs: tuple[tuple[str, str], ...]
    required_properties: frozenset[str] = frozenset()
    public: bool = True
    category: str = "structural"

    # --- Temporal traits ---------------------------------------------------
    lifecycle_carrier: bool = False
    """Edge carries ``lifecycle_status`` (proposed/in_progress/completed/...)."""

    predicate_family: str | None = None
    """Membership in a predicate family for auto-supersession and conflict detection."""

    # --- Edge cardinality (rebuild plan P2 / F3) ---------------------------
    singleton: bool = False
    """When True, ``(subject, predicate)`` admits one live object at a time.

    The canonical writer auto-stamps ``invalid_at`` on any prior live
    claim with the same subject + predicate but a different object when
    a new deterministic claim lands. Multi-source corroboration on the
    same object is preserved (singleton applies to objects, not sources).
    """

    # --- Endpoint disambiguation -------------------------------------------
    source_inferred_labels: tuple[str, ...] = ()
    target_inferred_labels: tuple[str, ...] = ()
    """When this edge type appears, the classifier may add these labels to source/target."""

    def allows(self, from_labels: Iterable[str], to_labels: Iterable[str]) -> bool:
        from_set = _normalized_label_set(from_labels)
        to_set = _normalized_label_set(to_labels)
        return any(
            _endpoint_matches(left, from_set) and _endpoint_matches(right, to_set)
            for left, right in self.allowed_pairs
        )


# --- Entity catalog (v2) ----------------------------------------------------
# Conventions:
#   * ``scope=True``  → endpoint for cross-cutting edges (APPLIES_TO etc.)
#   * ``is_activity=True`` → rich change event; carries ``Activity`` label too
#   * ``project_map_family``, ``debugging_family`` → drive the structural reader
#   * ``include_keys`` → stable agent contract; remap to relabel internals safely
#   * ``fact_family``, ``source_of_truth``, ``freshness_ttl_hours`` → drive graph_quality
#   * ``text_patterns`` / ``property_signatures`` → drive ontology_classifier


def _e(label: str, category: str, description: str, *, identity: str, **kwargs) -> EntityTypeSpec:
    """Helper for declaring an entity spec with concise call sites."""
    required = kwargs.pop("required", ())
    lifecycle = kwargs.pop("lifecycle", frozenset())
    return EntityTypeSpec(
        label=label,
        category=category,
        description=description,
        identity_policy=identity,
        required_properties=frozenset(required),
        lifecycle_states=frozenset(lifecycle),
        **kwargs,
    )


# Source-of-truth presets to make rows concise.
SOT_EXTERNAL = SOURCE_OF_TRUTH_AUTHORITATIVE_EXTERNAL
SOT_CODE = SOURCE_OF_TRUTH_AUTHORITATIVE_CODE
SOT_MEMORY = SOURCE_OF_TRUTH_CANONICAL_MEMORY
SOT_SOFT = SOURCE_OF_TRUTH_SOFT_INFERENCE

H = 1
DAY = 24
WEEK = 7 * DAY


ENTITY_TYPES: dict[str, EntityTypeSpec] = {
    # ============================================================
    # Layer 1 — Scope (the "where"; endpoints for cross-cutting edges)
    # ============================================================
    "Pot": _e(
        "Pot",
        "scope_identity",
        "Isolation boundary for project context.",
        identity="pot slug",
        required=("name",),
        scope=True,
        project_map_family="purpose",
        fact_family="scope",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=30 * DAY,
    ),
    "Repository": _e(
        "Repository",
        "scope_identity",
        "Source repository mapped to a pot.",
        identity="provider host plus owner/name",
        required=("name", "provider"),
        lifecycle=COMMON_LIFECYCLE_STATES,
        scope=True,
        project_map_family="repo_map",
        fact_family="scope",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
    ),
    "System": _e(
        "System",
        "scope_identity",
        "Product or platform boundary.",
        identity="pot-scoped slug",
        required=("name",),
        lifecycle=COMMON_LIFECYCLE_STATES,
        scope=True,
        project_map_family="purpose",
        fact_family="scope",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=30 * DAY,
    ),
    "Service": _e(
        "Service",
        "scope_identity",
        "Deployable runtime unit.",
        identity="pot-scoped service slug",
        required=("name", "criticality", "lifecycle_state"),
        lifecycle=COMMON_LIFECYCLE_STATES,
        scope=True,
        project_map_family="service_map",
        fact_family="service",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=14 * DAY,
    ),
    "Environment": _e(
        "Environment",
        "scope_identity",
        "Local, staging, production, preview, or regional runtime environment.",
        identity="pot-scoped environment slug",
        required=("name", "environment_type"),
        lifecycle=COMMON_LIFECYCLE_STATES,
        scope=True,
        project_map_family="deployments",
        fact_family="environment",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=7 * DAY,
        property_signatures=("environment_type",),
    ),
    "Component": _e(
        "Component",
        "product_architecture",
        "Logical subsystem, module, package, or bounded context.",
        identity="repo/service-scoped semantic key",
        required=("name", "component_type"),
        lifecycle=COMMON_LIFECYCLE_STATES,
        scope=True,
        project_map_family="service_map",
        fact_family="service",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=14 * DAY,
        property_signatures=("component_type",),
    ),
    "CodeAsset": _e(
        "CodeAsset",
        "code_topology",
        "File, function, class, module, or package referenced by context facts. "
        "Bridge to legacy code-graph labels (FILE/FUNCTION/CLASS).",
        identity="provider/repo/path/symbol identity",
        required=("name", "asset_type"),
        scope=True,
        project_map_family="repo_map",
        fact_family="code",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=7 * DAY,
        property_signatures=("asset_type",),
    ),
    # ============================================================
    # Layer 2 — Intent (what we want, what we're worried about)
    # ============================================================
    "Initiative": _e(
        "Initiative",
        "intent",
        "Named multi-PR, multi-quarter effort. Parent for grouped PRs, "
        "Issues, Decisions, and Features.",
        identity="initiative slug or planning-source id",
        required=("name", "status"),
        lifecycle=COMMON_LIFECYCLE_STATES,
        project_map_family="initiatives",
        include_keys=("initiatives",),
        fact_family="initiative",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=30 * DAY,
        text_patterns=(
            r"^\s*(initiative|workstream|epic|program(me)?)\s*[:\-]",
            r"\b("
            r"initiative|workstream|program(me)?|epic|big[- ]rock|"
            r"q[1-4]\s+(launch|rollout|effort|push)|"
            r"(auth|payments?|search|context[- ]engine)\s+(rewrite|migration|overhaul)"
            r")\b",
        ),
    ),
    "Capability": _e(
        "Capability",
        "product_architecture",
        "External product behavior or capability.",
        identity="system-scoped capability slug",
        required=("name",),
        lifecycle=COMMON_LIFECYCLE_STATES,
        project_map_family="feature_map",
        include_keys=("feature_map",),
        fact_family="feature",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=30 * DAY,
    ),
    "Feature": _e(
        "Feature",
        "product_architecture",
        "Concrete deliverable area within a capability.",
        identity="capability/system-scoped feature slug",
        required=("name",),
        lifecycle=COMMON_LIFECYCLE_STATES,
        project_map_family="feature_map",
        include_keys=("feature_map",),
        fact_family="feature",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=30 * DAY,
    ),
    "Requirement": _e(
        "Requirement",
        "product_architecture",
        "Expected behavior or acceptance criterion.",
        identity="source ref or scoped requirement slug",
        required=("statement", "status"),
        project_map_family="feature_map",
        include_keys=("feature_map",),
        fact_family="feature",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=30 * DAY,
    ),
    "RoadmapItem": _e(
        "RoadmapItem",
        "product_architecture",
        "Planned-but-not-started evolution or future direction.",
        identity="planning-source id or scoped slug",
        required=("title", "status"),
        project_map_family="feature_map",
        include_keys=("feature_map", "roadmap"),
        fact_family="feature",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
    ),
    "Risk": _e(
        "Risk",
        "intent",
        "Thing we're worried about. Distinct from Issue (tracked work) and "
        "Policy (decided rule). Carries severity and status.",
        identity="risk slug or scoped id",
        required=("summary", "severity", "status"),
        lifecycle=RISK_STATES,
        project_map_family="risks",
        include_keys=("risks",),
        fact_family="risk",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=30 * DAY,
        text_patterns=(
            r"^\s*risk\s*[:\-]",
            r"\b("
            r"risk(\s+(of|that))?|concerned\s+about|worried\s+about|"
            r"failure\s+mode|attack\s+surface|hazard"
            r")\b",
        ),
    ),
    "OpenQuestion": _e(
        "OpenQuestion",
        "intent",
        "Recorded unknown the team has not yet decided. Resolved when a "
        "Decision answers it.",
        identity="question slug or scoped id",
        required=("question", "status"),
        lifecycle=QUESTION_STATES,
        project_map_family="open_questions",
        include_keys=("open_questions",),
        fact_family="open_question",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=30 * DAY,
        text_patterns=(
            r"^\s*(open\s*question|question|q)\s*[:\-]",
            r"\b("
            r"open\s+question|unresolved\s+(question|issue|decision)|"
            r"tbd|not\s+yet\s+(decided|determined)|"
            r"do\s+we\s+(want|need)|should\s+we\b"
            r")",
        ),
    ),
    # ============================================================
    # Layer 3 — Norms (what we've agreed)
    # ============================================================
    "Decision": _e(
        "Decision",
        "change_decision",
        "Canonical engineering or product decision. Past-tense, dated act.",
        identity="source ref or scoped decision slug",
        required=("title", "summary", "status"),
        lifecycle=DECISION_STATES,
        project_map_family="decisions",
        include_keys=("decisions",),
        fact_family="decision",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=180 * DAY,
        text_patterns=(
            r"\b("
            r"(we|the\s+team|engineering|product)\s+(decided|chose|adopted|agreed)|"
            r"(adopted|selected|chose|going\s+with)\s+[\w\-./]+\s+(over|instead\s+of|as)|"
            r"architecture\s+decision|"
            r"design\s+decision|"
            r"decision\s+record|"
            r"\badr[- ]?\d*\b"
            r")",
        ),
    ),
    "Policy": _e(
        "Policy",
        "change_decision",
        "Normative rule that shapes future work. Replaces Constraint, "
        "Preference, and AgentInstruction. ``strength`` (required/preferred/"
        "optional) and ``audience`` (humans/agents/both) carry the variance. "
        "Most Policy facts are softly inferred from chat / docs; per-fact "
        "``evidence_strength`` distinguishes deterministic-from-AGENTS.md "
        "from hypothesized-from-Slack.",
        identity="scope plus policy slug",
        required=("statement", "strength"),
        lifecycle=COMMON_LIFECYCLE_STATES,
        project_map_family="policies",
        include_keys=("policies", "preferences", "agent_instructions", "constraints"),
        fact_family="policy",
        source_of_truth=SOT_SOFT,
        freshness_ttl_hours=60 * DAY,
        # ``constraint_type`` / ``instruction_type`` / ``preference_type`` are
        # legacy properties from the old Constraint / AgentInstruction /
        # Preference extractors; route any entity with one to Policy.
        property_signatures=(
            "strength",
            "audience",
            "constraint_type",
            "instruction_type",
            "preference_type",
        ),
        text_patterns=(
            # Explicit labels (extractors and humans use these as headers)
            r"^\s*(policy|rule|convention|constraint|preference|agent\s+instruction)\s*[:\-]",
            # Strength/audience metadata (very strong signal)
            r"\bstrength\s*[:=]\s*(required|preferred|optional|recommended)\b",
            r"\baudience\s*[:=]\s*(humans?|agents?|both)\b",
            # Phrasings that humans use in docs / chat
            r"\b("
            r"team\s+prefers|we\s+prefer|preferred\s+approach|style\s+preference|"
            r"hard\s+constraint|architectural\s+constraint|compliance\s+requirement|"
            r"must\s+not\s+(be\s+)?(use|used|stored|exposed|committed|logged)|"
            r"never\s+(commit|log|store|expose|call)\s+\w+|"
            r"do\s+not\s+(call|use|import|modify)\s+\w+|"
            r"agents?\.md|claude\.md|cursor\.md|\.cursorrules|"
            r"agent\s+instruction|skill\s+definition|mcp\s+guidance"
            r")",
        ),
    ),
    # ============================================================
    # Layer 4 — Code, contracts, configuration
    # ============================================================
    "APIContract": _e(
        "APIContract",
        "product_architecture",
        "REST, RPC, GraphQL, CLI, or webhook surface exposed by a service.",
        identity="component/service-scoped contract slug",
        required=("name", "contract_type"),
        project_map_family="service_map",
        include_keys=("service_map", "contracts", "interfaces"),
        fact_family="contract",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=14 * DAY,
        # ``interface_type`` is the legacy property emitted by the old
        # Interface extractor; route those to APIContract.
        property_signatures=("contract_type", "interface_type"),
    ),
    "DataContract": _e(
        "DataContract",
        "product_architecture",
        "Schema, event payload, message format, or migration boundary.",
        identity="component/store-scoped schema slug",
        required=("name", "schema_type"),
        project_map_family="service_map",
        include_keys=("service_map", "schemas"),
        fact_family="contract",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=14 * DAY,
        property_signatures=("schema_type",),
    ),
    "DataStore": _e(
        "DataStore",
        "product_architecture",
        "Database, cache, object store, or SaaS storage.",
        identity="service/system-scoped datastore slug",
        required=("name", "store_type"),
        project_map_family="service_map",
        include_keys=("service_map", "datastores"),
        fact_family="datastore",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
        property_signatures=("store_type",),
    ),
    "Dependency": _e(
        "Dependency",
        "product_architecture",
        "External system, library, runtime service, API integration, or "
        "platform with operational significance. ``dependency_kind`` "
        "distinguishes library/runtime_service/external_api/platform.",
        identity="package/system scoped dependency id",
        required=("name", "dependency_kind"),
        project_map_family="service_map",
        include_keys=("service_map", "dependencies", "integrations"),
        fact_family="dependency",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
        property_signatures=("dependency_kind", "dependency_type", "integration_type"),
    ),
    "FeatureFlag": _e(
        "FeatureFlag",
        "delivery_operations",
        "Runtime feature gate with rollout state. Critical for debugging "
        "and impact analysis.",
        identity="provider plus flag key",
        required=("name", "status"),
        lifecycle=FLAG_STATES,
        project_map_family="config",
        include_keys=("feature_flags", "config"),
        fact_family="feature_flag",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=1 * DAY,
        property_signatures=("flag_status", "rollout"),
        text_patterns=(
            r"^\s*(feature\s*flag|flag)\s*[:\-]",
            r"\b(feature\s+flag|launchdarkly|growthbook|flagged\s+behind|gated\s+on)\b",
        ),
    ),
    "ConfigVariable": _e(
        "ConfigVariable",
        "delivery_operations",
        "Important config variable or secret reference (never secret value).",
        identity="scope plus variable name",
        required=("name", "scope_kind"),
        project_map_family="config",
        include_keys=("config",),
        fact_family="config",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=7 * DAY,
    ),
    # ============================================================
    # Layer 5 — Change pipeline (rich activities)
    # ============================================================
    "PullRequest": _e(
        "PullRequest",
        "change_decision",
        "Source-control pull request.",
        identity="provider/repo/pr number",
        required=("pr_number", "title"),
        is_activity=True,
        project_map_family="changes",
        include_keys=("changes", "pull_requests"),
        fact_family="change",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=30 * DAY,
        property_signatures=("pr_number",),
    ),
    "Commit": _e(
        "Commit",
        "change_decision",
        "Source-control commit.",
        identity="provider/repo/sha",
        required=("sha",),
        is_activity=True,
        project_map_family="changes",
        include_keys=("changes", "commits"),
        fact_family="change",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=30 * DAY,
        property_signatures=("sha",),
    ),
    "Branch": _e(
        "Branch",
        "delivery_operations",
        "Git branch with operational meaning.",
        identity="provider/repo/branch name",
        required=("name",),
        project_map_family="changes",
        include_keys=("branches",),
        fact_family="change",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=7 * DAY,
    ),
    "Release": _e(
        "Release",
        "delivery_operations",
        "Tagged version that fans out into one or more deployments.",
        identity="repo plus version tag",
        required=("version",),
        is_activity=True,
        project_map_family="releases",
        include_keys=("releases", "deployments"),
        fact_family="release",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
        text_patterns=(
            r"\b(release\s+\d|v\d+\.\d+(\.\d+)?|tagged\s+version|cut\s+a\s+release)\b",
        ),
    ),
    "Deployment": _e(
        "Deployment",
        "delivery_operations",
        "Version or branch promoted into an environment.",
        identity="environment plus deployment id",
        required=("version", "deployed_at"),
        is_activity=True,
        project_map_family="deployments",
        include_keys=("deployments",),
        fact_family="deployment",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=3 * DAY,
    ),
    "Migration": _e(
        "Migration",
        "delivery_operations",
        "Schema, data, or code migration with a multi-phase lifecycle.",
        identity="migration slug or source id",
        required=("name", "migration_kind", "phase"),
        lifecycle=MIGRATION_STATES,
        is_activity=True,
        project_map_family="changes",
        include_keys=("migrations", "changes"),
        fact_family="migration",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=7 * DAY,
        property_signatures=("migration_kind", "phase"),
        text_patterns=(
            r"^\s*migration\s*[:\-]",
            r"\b("
            r"(schema|data|code)\s+migration|"
            r"backfill(ing)?|cut\s+over|cutover|"
            r"alembic|flyway|liquibase|"
            r"renamed?\s+(table|column)|added\s+column|dropped\s+column"
            r")\b",
        ),
    ),
    "Issue": _e(
        "Issue",
        "change_decision",
        "Ticket, issue, bug, or planning item.",
        identity="provider issue key/id",
        required=("title", "status"),
        lifecycle=ISSUE_STATES,
        is_activity=True,
        project_map_family="tickets",
        include_keys=("tickets", "issues"),
        fact_family="change",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
        property_signatures=("issue_number",),
    ),
    # ============================================================
    # Layer 6 — Operations & reliability
    # ============================================================
    "Incident": _e(
        "Incident",
        "delivery_operations",
        "Operational issue with timeline and severity.",
        identity="source incident id or scoped slug",
        required=("title", "severity", "status"),
        lifecycle=INCIDENT_STATES,
        is_activity=True,
        project_map_family="incidents",
        debugging_family="incidents",
        include_keys=("incidents",),
        fact_family="incident",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=12 * H,
        text_patterns=(
            r"\b("
            r"incident|outage|downtime|postmortem|post[- ]mortem|"
            r"p[0-4]\s+(incident|event|issue)|"
            r"sev[- ]?[0-4]\b"
            r")",
        ),
    ),
    "Alert": _e(
        "Alert",
        "delivery_operations",
        "Monitoring or incident signal.",
        identity="source alert id or fingerprint",
        required=("title", "severity", "status"),
        is_activity=True,
        project_map_family="alerts",
        debugging_family="alerts",
        include_keys=("alerts",),
        fact_family="alert",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=2 * H,
        text_patterns=(
            r"\b("
            r"pagerduty|paged\s+on[- ]?call|pager\s+duty|"
            r"alert\s+fired|alerting\s+rule|threshold\s+alert"
            r")",
        ),
    ),
    "BugPattern": _e(
        "BugPattern",
        "debugging_reliability",
        "Repeated failure mode or symptom cluster.",
        identity="scope plus symptom signature",
        required=("summary",),
        project_map_family="bug_patterns",
        debugging_family="prior_fixes",
        include_keys=("prior_fixes", "bug_patterns"),
        fact_family="bug_pattern",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=90 * DAY,
        text_patterns=(
            r"\b("
            r"flaky\s+(test|suite|spec)|bug\s+pattern|anti[- ]pattern|"
            r"recurring\s+(failure|bug|regression)|"
            r"known\s+(issue|failure\s+mode|bad\s+pattern)"
            r")",
        ),
    ),
    "Investigation": _e(
        "Investigation",
        "debugging_reliability",
        "Debugging session, diagnostic path, or incident investigation.",
        identity="source session/incident id or generated id",
        required=("summary", "status"),
        project_map_family="investigations",
        debugging_family="investigations",
        include_keys=("investigations", "prior_fixes"),
        fact_family="investigation",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=90 * DAY,
    ),
    "Fix": _e(
        "Fix",
        "debugging_reliability",
        "Resolution, mitigation, workaround, or permanent code/config change.",
        identity="source ref or generated fix id",
        required=("summary", "fix_type"),
        project_map_family="fixes",
        debugging_family="prior_fixes",
        include_keys=("prior_fixes", "fixes"),
        fact_family="fix",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=90 * DAY,
        property_signatures=("fix_type",),
        text_patterns=(
            r"\b("
            r"hotfix|bug\s*fix|bugfix|"
            r"patch(ed|ing)?\s+(the\s+)?(bug|issue|regression|vulnerability)|"
            r"workaround\s+for|mitigation\s+for"
            r")",
        ),
    ),
    "Observation": _e(
        "Observation",
        "knowledge_evidence",
        "Normalized evidence unit. Absorbs DiagnosticSignal — error "
        "signatures, log queries, metrics, alert fingerprints, and symptoms "
        "are Observations with ``signal_type``.",
        identity="source ref plus observation slug",
        required=("summary",),
        project_map_family="observations",
        debugging_family="diagnostic_signals",
        include_keys=("observations", "diagnostic_signals"),
        fact_family="observation",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=30 * DAY,
        property_signatures=("signal_type", "observed_at"),
    ),
    "Runbook": _e(
        "Runbook",
        "delivery_operations",
        "Human-usable remediation procedure.",
        identity="source doc id or scoped runbook slug",
        required=("title",),
        project_map_family="runbooks",
        debugging_family="runbooks",
        include_keys=("runbooks",),
        fact_family="runbook",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=30 * DAY,
        text_patterns=(
            r"\b(runbook|playbook|operational\s+procedure|recovery\s+steps|on[- ]?call\s+procedure)",
        ),
    ),
    "LocalWorkflow": _e(
        "LocalWorkflow",
        "delivery_operations",
        "How people run, test, debug, or deploy locally.",
        identity="repo/service-scoped workflow slug",
        required=("name", "workflow_type"),
        project_map_family="local_workflows",
        include_keys=("local_workflows",),
        fact_family="local_workflow",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=30 * DAY,
        property_signatures=("workflow_type",),
    ),
    "Script": _e(
        "Script",
        "delivery_operations",
        "Local, CI, debug, or deployment command used by the team.",
        identity="repo path or scoped script slug",
        required=("name", "command"),
        project_map_family="scripts",
        include_keys=("scripts",),
        fact_family="script",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=30 * DAY,
    ),
    "Metric": _e(
        "Metric",
        "delivery_operations",
        "Named health indicator.",
        identity="service/environment metric name",
        required=("name", "metric_type"),
        project_map_family="metrics",
        debugging_family="metrics",
        include_keys=("metrics",),
        fact_family="metric",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=1 * DAY,
        property_signatures=("metric_type",),
    ),
    # ============================================================
    # Layer 7 — People
    # ============================================================
    "Person": _e(
        "Person",
        "team_ownership",
        "Contributor or stakeholder.",
        identity="provider user id, email, or pot-scoped person slug",
        required=("name",),
        project_map_family="owners",
        include_keys=("owners", "people"),
        fact_family="ownership",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
        property_signatures=("github_login", "display_name"),
    ),
    "Team": _e(
        "Team",
        "team_ownership",
        "Functional, product, or ownership team.",
        identity="pot-scoped team slug",
        required=("name",),
        lifecycle=COMMON_LIFECYCLE_STATES,
        project_map_family="owners",
        include_keys=("owners", "teams"),
        fact_family="ownership",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
    ),
    "Agent": _e(
        "Agent",
        "team_ownership",
        "Automated coding agent, IDE agent, or service agent.",
        identity="agent provider plus agent id or pot-scoped slug",
        required=("name", "agent_type"),
        project_map_family="owners",
        include_keys=("agents",),
        fact_family="ownership",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
        property_signatures=("agent_type",),
    ),
    "RoleAssignment": _e(
        "RoleAssignment",
        "team_ownership",
        "Time-bounded role assignment (on-call, owner, reviewer, maintainer). "
        "Has valid_from/valid_to so 'who was on-call on date X' is queryable.",
        identity="scope plus role plus person plus interval",
        required=("role_type",),
        project_map_family="owners",
        include_keys=("role_assignments", "oncall"),
        fact_family="ownership",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
        property_signatures=("role_type",),
    ),
    # ============================================================
    # Layer 8 — Evidence & provenance
    # ============================================================
    "SourceSystem": _e(
        "SourceSystem",
        "knowledge_evidence",
        "GitHub, Linear, Slack, Docs, Sentry, GCP, AWS, or another provider.",
        identity="provider slug",
        required=("name", "source_type"),
        fact_family="source_system",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=7 * DAY,
    ),
    "SourceReference": _e(
        "SourceReference",
        "knowledge_evidence",
        "Stable pointer to an external artifact with resolver hints and freshness metadata.",
        identity="source system plus external id",
        required=("source_system", "external_id", "ref_type"),
        fact_family="source_reference",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=14 * DAY,
        property_signatures=("ref_type",),
    ),
    "Document": _e(
        "Document",
        "knowledge_evidence",
        "ADR, product doc, design doc, runbook doc, or wiki page.",
        identity="source document id or URL",
        required=("title", "source_uri"),
        project_map_family="docs",
        include_keys=("docs", "documents"),
        fact_family="document",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=30 * DAY,
    ),
    "Conversation": _e(
        "Conversation",
        "knowledge_evidence",
        "Slack thread, review discussion, planning thread, or incident thread.",
        identity="source conversation id",
        required=("title", "source_uri"),
        project_map_family="discussions",
        include_keys=("discussions",),
        fact_family="discussion",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=90 * DAY,
    ),
    # ============================================================
    # Layer 9 — Timeline (unified view across change events)
    # ============================================================
    "Activity": _e(
        "Activity",
        "timeline",
        "Atomic happening: verb + actor + subjects + time. Rich change "
        "events (PR, Commit, Deployment, Incident, ...) carry this label too "
        "so the timeline is one query. Pure Activities exist for events with "
        "no entity backing (chat, standup, agent micro-action).",
        identity="source system plus deterministic activity slug",
        required=("verb", "occurred_at"),
        lifecycle=frozenset({"in_progress", "completed", "unknown"}),
        project_map_family="timeline",
        include_keys=("timeline", "activities"),
        fact_family="activity",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=7 * DAY,
        property_signatures=("verb",),
    ),
    "Period": _e(
        "Period",
        "timeline",
        "Named time-window (Q2-2026, pre-launch-freeze). Auto-buckets "
        "(daily/weekly) are query primitives, not entities.",
        identity="pot-scoped label",
        required=("label", "period_kind", "opened_at"),
        lifecycle=frozenset({"open", "closed", "unknown"}),
        project_map_family="timeline",
        include_keys=("periods",),
        fact_family="period",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=30 * DAY,
    ),
    # ============================================================
    # Layer 10 — Graph hygiene (internal; not surfaced to agents)
    # ============================================================
    "QualityIssue": _e(
        "QualityIssue",
        "quality_drift",
        "Detected graph quality, freshness, source-sync, alias, orphan, or bridge issue.",
        identity="pot/scope plus issue code and affected entity",
        required=("code", "severity", "status"),
        lifecycle=COMMON_LIFECYCLE_STATES,
        public=False,
        fact_family="quality",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=7 * DAY,
    ),
    "MaintenanceJob": _e(
        "MaintenanceJob",
        "quality_drift",
        "Recurring verification, repair, cleanup, or materialization job.",
        identity="job family plus scope id",
        required=("job_type", "status"),
        lifecycle=COMMON_LIFECYCLE_STATES,
        public=False,
        fact_family="quality",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=7 * DAY,
        property_signatures=("job_type",),
    ),
    "MaterializedAccessPath": _e(
        "MaterializedAccessPath",
        "quality_drift",
        "Compact derived path maintained for repeated agent queries.",
        identity="pattern type plus scope id",
        required=("name", "pattern_type"),
        lifecycle=COMMON_LIFECYCLE_STATES,
        public=False,
        fact_family="quality",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=7 * DAY,
        property_signatures=("pattern_type",),
    ),
}


# --- Edge catalog (v2) ------------------------------------------------------


def _x(
    edge_type: str,
    description: str,
    pairs: Iterable[tuple[str, str]],
    *,
    required: Iterable[str] = (),
    category: str = "structural",
    public: bool = True,
    lifecycle_carrier: bool = False,
    predicate_family: str | None = None,
    singleton: bool = False,
    source_inferred: Iterable[str] = (),
    target_inferred: Iterable[str] = (),
) -> EdgeTypeSpec:
    return EdgeTypeSpec(
        edge_type=edge_type,
        description=description,
        allowed_pairs=tuple(pairs),
        required_properties=frozenset(required),
        public=public,
        category=category,
        lifecycle_carrier=lifecycle_carrier,
        predicate_family=predicate_family,
        singleton=singleton,
        source_inferred_labels=tuple(source_inferred),
        target_inferred_labels=tuple(target_inferred),
    )


EDGE_TYPES: dict[str, EdgeTypeSpec] = {
    # ===== Structural ========================================================
    "SCOPES": _x(
        "SCOPES",
        "Pot scopes project context entities.",
        [("Pot", SCOPE_ENDPOINT), ("Pot", "Document"), ("Pot", "Person"), ("Pot", "Team")],
        category="structural",
    ),
    "CONTAINS": _x(
        "CONTAINS",
        "Scope-to-scope or feature-to-feature containment.",
        [
            ("System", "Service"),
            ("System", "Component"),
            ("System", "Capability"),
            ("Repository", "Component"),
            ("Service", "Component"),
            ("Component", "Component"),
            ("Component", "CodeAsset"),
            ("Feature", "Feature"),
            ("Capability", "Feature"),
        ],
        category="structural",
    ),
    "BACKED_BY": _x(
        "BACKED_BY",
        "Service or component is backed by a repository.",
        [("Service", "Repository"), ("Component", "Repository")],
        category="structural",
    ),
    "EXPOSES": _x(
        "EXPOSES",
        "Component or service exposes an API or data contract.",
        [
            ("Component", "APIContract"),
            ("Service", "APIContract"),
            ("Component", "DataContract"),
            ("Service", "DataContract"),
        ],
        category="structural",
    ),
    "USES": _x(
        "USES",
        "Component or service uses a dependency or data store.",
        [
            ("Component", "Dependency"),
            ("Service", "Dependency"),
            ("Component", "DataStore"),
            ("Service", "DataStore"),
        ],
        category="structural",
        predicate_family="datastore_binding",
    ),
    "DEPENDS_ON": _x(
        "DEPENDS_ON",
        "Service-to-service or component-to-service runtime dependency.",
        [
            ("Service", "Service"),
            ("Component", "Service"),
        ],
        category="structural",
    ),
    "DEPLOYED_TO": _x(
        "DEPLOYED_TO",
        "Service or deployment runs in an environment.",
        [("Service", "Environment"), ("Deployment", "Environment")],
        category="structural",
        predicate_family="deployment_target",
        singleton=True,
    ),
    "OF_SERVICE": _x(
        "OF_SERVICE",
        # Rebuild plan P3 (F1): Service ↔ Deployment join.
        #
        # The proper POC found that LLM extractors over k8s manifests
        # commonly classify the manifest entity as ``Deployment`` (key
        # ``deploy:auth-svc``) and emit topology claims (USES,
        # DEPLOYED_TO) from the Deployment, not the Service. Without an
        # OF_SERVICE edge back to the named Service, the InfraTopology
        # reader's "what does auth-svc depend on?" query returns 0%.
        # Marking OF_SERVICE singleton lets the canonical writer keep
        # exactly one live binding per Deployment.
        "Deployment belongs to a named Service (per-Service deployment "
        "manifest, k8s namespace annotation, etc.).",
        [("Deployment", "Service")],
        category="structural",
        singleton=True,
    ),
    "OWNED_BY": _x(
        "OWNED_BY",
        # Rebuild plan P2 (F3) + Position B: services / components have
        # exactly one *current* owner. Multi-source corroboration on the
        # same owner is fine; a new owner supersedes the old one
        # deterministically. Inverse of OWNS (kept for back-compat with
        # the LLM extraction pipeline).
        "Service / component / feature is owned by one person or team.",
        [
            ("Service", "Person"),
            ("Service", "Team"),
            ("Component", "Person"),
            ("Component", "Team"),
            ("Feature", "Person"),
            ("Feature", "Team"),
            ("Repository", "Person"),
            ("Repository", "Team"),
        ],
        category="ownership",
        predicate_family="owner_binding",
        singleton=True,
    ),
    "GATED_BY": _x(
        "GATED_BY",
        "Feature or code asset is gated behind a feature flag.",
        [("Feature", "FeatureFlag"), ("CodeAsset", "FeatureFlag"), ("Service", "FeatureFlag")],
        category="structural",
    ),
    "CONFIGURED_BY": _x(
        "CONFIGURED_BY",
        "Scope is configured by a configuration variable.",
        [(SCOPE_ENDPOINT, "ConfigVariable")],
        category="structural",
    ),
    "TOUCHES_CODE": _x(
        "TOUCHES_CODE",
        "Feature touches code asset (materialized heuristic).",
        [("Feature", "CodeAsset")],
        category="structural",
        public=True,
    ),
    # ===== Intent ============================================================
    "PURSUES": _x(
        "PURSUES",
        "Initiative pursues a capability or feature.",
        [("Initiative", "Capability"), ("Initiative", "Feature")],
        category="intent",
    ),
    "IMPLEMENTS": _x(
        "IMPLEMENTS",
        "Feature implements capability; Service/Component implements feature.",
        [
            ("Feature", "Capability"),
            ("Service", "Feature"),
            ("Component", "Feature"),
        ],
        category="intent",
    ),
    "ADDRESSES": _x(
        "ADDRESSES",
        "Change addresses an issue, risk, roadmap item, or open question.",
        [
            ("PullRequest", "Issue"),
            ("PullRequest", "Risk"),
            ("PullRequest", "OpenQuestion"),
            ("Issue", "RoadmapItem"),
            ("Initiative", "RoadmapItem"),
            ("Initiative", "Risk"),
            ("Commit", "Issue"),
        ],
        category="intent",
        target_inferred=("Issue",),
    ),
    "BLOCKED_BY": _x(
        "BLOCKED_BY",
        "Intent is blocked by a risk, issue, open question, or migration.",
        [
            ("Initiative", "Risk"),
            ("Initiative", "Issue"),
            ("Initiative", "OpenQuestion"),
            ("Feature", "Risk"),
            ("Feature", "Issue"),
            ("Feature", "OpenQuestion"),
            ("RoadmapItem", "Risk"),
            ("RoadmapItem", "OpenQuestion"),
            ("Migration", "Risk"),
        ],
        category="intent",
    ),
    "PART_OF": _x(
        "PART_OF",
        "Element belongs to a larger work group.",
        [
            ("PullRequest", "Initiative"),
            ("Commit", "PullRequest"),
            ("Issue", "Initiative"),
            ("Decision", "Initiative"),
            ("Feature", "Initiative"),
            ("Migration", "Initiative"),
        ],
        category="intent",
    ),
    # ===== Norms / decisions =================================================
    "APPLIES_TO": _x(
        "APPLIES_TO",
        "Policy applies to a scope.",
        [("Policy", SCOPE_ENDPOINT)],
        category="norm",
        source_inferred=("Policy",),
    ),
    "AFFECTS": _x(
        "AFFECTS",
        "Decision affects a scope, feature, migration, or code asset.",
        [
            ("Decision", "Feature"),
            ("Decision", "Component"),
            ("Decision", "Service"),
            ("Decision", "CodeAsset"),
            ("Decision", "Migration"),
            ("Decision", "APIContract"),
            ("Decision", "DataContract"),
            ("Decision", "Initiative"),
        ],
        category="norm",
        source_inferred=("Decision",),
    ),
    "SUPERSEDES": _x(
        "SUPERSEDES",
        "Newer fact or entity supersedes a prior version; the target is soft-invalidated with valid_to.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="lifecycle",
        public=False,
    ),
    "MADE_IN": _x(
        "MADE_IN",
        "Decision was made in a source context.",
        [
            ("Decision", "PullRequest"),
            ("Decision", "Incident"),
            ("Decision", "Document"),
            ("Decision", "Conversation"),
        ],
        category="norm",
        source_inferred=("Decision",),
    ),
    "INFORMS": _x(
        "INFORMS",
        "Policy or agent-facing guidance informs an audience (humans/agents).",
        [
            ("Policy", "Agent"),
            ("Policy", "Person"),
            ("Policy", "Team"),
        ],
        category="norm",
        source_inferred=("Policy",),
    ),
    # ===== Ownership =========================================================
    "OWNS": _x(
        "OWNS",
        "Person or team owns scope or artifact.",
        [
            ("Person", "Service"),
            ("Person", "Component"),
            ("Person", "Feature"),
            ("Person", "Repository"),
            ("Person", "CodeAsset"),
            ("Team", "Service"),
            ("Team", "Component"),
            ("Team", "Capability"),
            ("Team", "Feature"),
            ("Team", "Runbook"),
            ("Team", "Repository"),
            ("Team", "Initiative"),
        ],
        category="ownership",
        predicate_family="owner_binding",
    ),
    "MEMBER_OF": _x(
        "MEMBER_OF",
        "Person belongs to a team.",
        [("Person", "Team")],
        category="ownership",
        target_inferred=("Team",),
    ),
    "ASSIGNED": _x(
        "ASSIGNED",
        "RoleAssignment binds a person to a scope for an interval.",
        [
            ("RoleAssignment", "Person"),
            ("RoleAssignment", SCOPE_ENDPOINT),
        ],
        category="ownership",
        source_inferred=("RoleAssignment",),
    ),
    "REVIEWS": _x(
        "REVIEWS",
        "Person reviews a change.",
        [("Person", "PullRequest"), ("Person", "Commit")],
        category="ownership",
    ),
    # ===== Change / pipeline =================================================
    "HAS_COMMIT": _x(
        "HAS_COMMIT",
        "Pull request contains a commit.",
        [("PullRequest", "Commit")],
        category="change",
        target_inferred=("Commit",),
    ),
    "HAS_REVIEW_DECISION": _x(
        "HAS_REVIEW_DECISION",
        "Pull request review discussion produced a decision.",
        [("PullRequest", "Decision")],
        category="change",
        target_inferred=("Decision",),
    ),
    "MODIFIED": _x(
        "MODIFIED",
        "Pull request modified a code asset.",
        [("PullRequest", "CodeAsset")],
        category="change",
        source_inferred=("PullRequest",),
    ),
    "TARGETS": _x(
        "TARGETS",
        "Deployment targets environment.",
        [("Deployment", "Environment")],
        category="change",
        source_inferred=("Deployment",),
    ),
    "DEPLOYED_AS": _x(
        "DEPLOYED_AS",
        "Branch is deployed as a deployment; deployment ships a release.",
        [("Branch", "Deployment"), ("Release", "Deployment")],
        category="change",
        source_inferred=("Branch",),
        target_inferred=("Deployment",),
    ),
    # ===== Reliability =======================================================
    "IMPACTS": _x(
        "IMPACTS",
        "Incident or alert impacts a service, environment, or feature.",
        [
            ("Incident", "Service"),
            ("Incident", "Environment"),
            ("Incident", "Feature"),
            ("Alert", "Service"),
            ("Alert", "Environment"),
        ],
        category="reliability",
    ),
    "FIRED_IN": _x(
        "FIRED_IN",
        "Alert fired in environment.",
        [("Alert", "Environment")],
        category="reliability",
        source_inferred=("Alert",),
        target_inferred=("Environment",),
    ),
    "INDICATES": _x(
        "INDICATES",
        "Alert indicates incident.",
        [("Alert", "Incident")],
        category="reliability",
        source_inferred=("Alert",),
        target_inferred=("Incident",),
    ),
    "MATCHES_PATTERN": _x(
        "MATCHES_PATTERN",
        "Incident matches a bug pattern.",
        [("Incident", "BugPattern")],
        category="reliability",
        target_inferred=("BugPattern",),
    ),
    "DEBUGGED": _x(
        "DEBUGGED",
        "Investigation debugged incident or pattern.",
        [("Investigation", "Incident"), ("Investigation", "BugPattern")],
        category="reliability",
        source_inferred=("Investigation",),
    ),
    "OBSERVED_IN": _x(
        "OBSERVED_IN",
        "Observation was seen in an investigation or incident.",
        [("Observation", "Investigation"), ("Observation", "Incident")],
        category="reliability",
        source_inferred=("Observation",),
    ),
    "HAS_SIGNAL": _x(
        "HAS_SIGNAL",
        "Bug pattern, incident, or investigation has an observation/signal.",
        [
            ("BugPattern", "Observation"),
            ("Incident", "Observation"),
            ("Investigation", "Observation"),
        ],
        category="reliability",
        target_inferred=("Observation",),
    ),
    "HAS_ROOT_CAUSE": _x(
        "HAS_ROOT_CAUSE",
        "Investigation or incident identified a root-cause observation.",
        [("Investigation", "Observation"), ("Incident", "Observation")],
        category="reliability",
        target_inferred=("Observation",),
    ),
    "RESOLVED": _x(
        "RESOLVED",
        "Fix resolved incident or bug pattern.",
        [("Fix", "Incident"), ("Fix", "BugPattern")],
        category="reliability",
        source_inferred=("Fix",),
    ),
    "CHANGED_BY": _x(
        "CHANGED_BY",
        "Fix was changed by PR or commit.",
        [("Fix", "PullRequest"), ("Fix", "Commit")],
        category="reliability",
        source_inferred=("Fix",),
    ),
    "MITIGATES": _x(
        "MITIGATES",
        "Runbook or fix mitigates incident or bug pattern.",
        [
            ("Runbook", "Incident"),
            ("Runbook", "BugPattern"),
            ("Fix", "Incident"),
            ("Fix", "BugPattern"),
        ],
        category="reliability",
    ),
    "SEEN_IN": _x(
        "SEEN_IN",
        "Bug pattern seen in scope.",
        [
            ("BugPattern", "Service"),
            ("BugPattern", "Environment"),
            ("BugPattern", "Component"),
        ],
        category="reliability",
        source_inferred=("BugPattern",),
    ),
    "INVOLVES_CODE": _x(
        "INVOLVES_CODE",
        "Incident involves code asset.",
        [("Incident", "CodeAsset")],
        category="reliability",
    ),
    "REFERENCES_CODE": _x(
        "REFERENCES_CODE",
        "Runbook references code asset.",
        [("Runbook", "CodeAsset")],
        category="reliability",
    ),
    # ===== Evidence / provenance ============================================
    "EVIDENCED_BY": _x(
        "EVIDENCED_BY",
        "Canonical fact is evidenced by source reference.",
        [(WILDCARD_ENDPOINT, "SourceReference")],
        category="evidence",
        target_inferred=("SourceReference",),
    ),
    "FROM_SOURCE": _x(
        "FROM_SOURCE",
        "Source reference came from source system.",
        [("SourceReference", "SourceSystem")],
        category="evidence",
        source_inferred=("SourceReference",),
        target_inferred=("SourceSystem",),
    ),
    "DESCRIBES": _x(
        "DESCRIBES",
        "Document or conversation describes a context entity.",
        [
            ("Document", "Feature"),
            ("Document", "Component"),
            ("Document", "Policy"),
            ("Document", "Initiative"),
            ("Document", "Migration"),
            ("Conversation", "Decision"),
            ("Conversation", "Incident"),
        ],
        category="evidence",
    ),
    "RESULTED_IN": _x(
        "RESULTED_IN",
        "Conversation resulted in a decision.",
        [("Conversation", "Decision")],
        category="evidence",
        target_inferred=("Decision",),
    ),
    # ===== Identity =========================================================
    "ALIASES": _x(
        "ALIASES",
        "Entity has an alias.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="identity",
    ),
    "RENAMED_FROM": _x(
        "RENAMED_FROM",
        "Entity was renamed from prior identity.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="identity",
    ),
    "MERGED_FROM": _x(
        "MERGED_FROM",
        "Entity was merged from another identity.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="identity",
    ),
    "SPLIT_FROM": _x(
        "SPLIT_FROM",
        "Entity was split from another identity.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="identity",
    ),
    # ===== Hygiene (internal) ===============================================
    "FLAGS": _x(
        "FLAGS",
        "Quality issue flags an affected entity or source reference.",
        [("QualityIssue", WILDCARD_ENDPOINT)],
        category="hygiene",
        public=False,
        source_inferred=("QualityIssue",),
    ),
    "REPAIRS": _x(
        "REPAIRS",
        "Maintenance job repairs or verifies an affected entity.",
        [("MaintenanceJob", WILDCARD_ENDPOINT)],
        category="hygiene",
        public=False,
        source_inferred=("MaintenanceJob",),
    ),
    "MATERIALIZES": _x(
        "MATERIALIZES",
        "Materialized access path precomputes a query path for an entity.",
        [("MaterializedAccessPath", WILDCARD_ENDPOINT)],
        category="hygiene",
        public=False,
        source_inferred=("MaterializedAccessPath",),
    ),
    # ===== Lifecycle / causal (controlled wildcards) ========================
    "LIFECYCLE_TRANSITION": _x(
        "LIFECYCLE_TRANSITION",
        "Any entity transitions to a new lifecycle state. Carries "
        "``from_state``, ``to_state``, ``verb`` properties. Replaces "
        "PLANNED / DELIVERED / DEPRECATED / DECOMMISSIONED / MIGRATED_TO / "
        "ADDED_TO / REMOVED_FROM / GENERIC_ACTION episodic verbs.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="lifecycle",
        lifecycle_carrier=True,
        predicate_family="lifecycle_status",
    ),
    "CAUSED": _x(
        "CAUSED",
        "Causal link between events or changes.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="causal",
    ),
    "DECIDES_FOR": _x(
        "DECIDES_FOR",
        "Decision or policy governs a scope or component. Common LLM "
        "episodic verb; accepted as a polymorphic edge so ingestion does "
        "not reject upstream extractor output.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="norm",
        target_inferred=("Decision",),
    ),
    "REPLACES": _x(
        "REPLACES",
        "New component, system, or contract replaces an older one.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="lifecycle",
    ),
    "STORED_IN": _x(
        "STORED_IN",
        "Data is persisted in or primarily associated with a data store.",
        [(WILDCARD_ENDPOINT, "DataStore"), ("Component", "DataStore"), ("Service", "DataStore")],
        category="structural",
        predicate_family="datastore_binding",
        target_inferred=("DataStore",),
    ),
    "RELATED_TO": _x(
        "RELATED_TO",
        "Catch-all when the extractor emits a non-catalog edge type; "
        "preserve ``original_edge_type`` on edge properties.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="lifecycle",
        public=False,
    ),
    # ===== Timeline =========================================================
    "PERFORMED": _x(
        "PERFORMED",
        "Actor (Person, Agent, or Team) performed a timeline Activity.",
        [
            ("Person", "Activity"),
            ("Agent", "Activity"),
            ("Team", "Activity"),
        ],
        required=("valid_from",),
        category="timeline",
    ),
    "TOUCHED": _x(
        "TOUCHED",
        "Timeline Activity touched a subject. Wildcard target with mandatory "
        "``verb_class`` property (code_change / deployment / discussion / "
        "decision / alert) so agents can filter without loading the Activity node.",
        [("Activity", WILDCARD_ENDPOINT)],
        required=("valid_from",),
        category="timeline",
    ),
    "IN_PERIOD": _x(
        "IN_PERIOD",
        "Timeline Activity falls within a Period rollup bucket.",
        [("Activity", "Period")],
        category="timeline",
    ),
    "MENTIONS": _x(
        "MENTIONS",
        # Rebuild plan P5 (F4): episode → entity provenance.
        #
        # The proper POC found that PR-merged Activity events emit
        # ``(person:alice) -[MERGED_BY]-> (pr:1042)`` but never link to
        # the affected service. The TIME reader's "activities about
        # service X" query missed PR events entirely. MENTIONS closes
        # this: every entity surfaced in an enriched episode body gets
        # a MENTIONS claim back from the source Activity, so readers
        # can traverse ``(Activity)-[:MENTIONS]->(Entity)`` to find
        # scope-relevant activities.
        "Activity / episode body references an entity (provenance link).",
        [("Activity", WILDCARD_ENDPOINT)],
        category="timeline",
    ),
    "ALIAS_OF": _x(
        "ALIAS_OF",
        # Rebuild plan P1: alias-as-claim. A surface form like
        # "Auth Service" resolves to ``service:auth-svc`` by writing
        # an ALIAS_OF claim from the variant entity to the canonical;
        # the canonicalization pass + identity resolver follow the
        # chain to find the canonical key.
        "Surface-form entity is the same logical thing as the canonical entity.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="identity",
    ),
}


# --- Derived tables (single source of truth: ENTITY_TYPES / EDGE_TYPES) -----

CANONICAL_LABELS: frozenset[str] = frozenset(ENTITY_TYPES.keys())
CANONICAL_EDGE_TYPES: frozenset[str] = frozenset(EDGE_TYPES.keys())
SCOPE_LABELS: frozenset[str] = frozenset(
    label for label, spec in ENTITY_TYPES.items() if spec.scope
)
ACTIVITY_LABELS: frozenset[str] = frozenset(
    label for label, spec in ENTITY_TYPES.items() if spec.is_activity
) | {"Activity"}
# Rebuild plan P2 (F3) + P3: ontology is the single source of truth
# for which predicates are singleton; the registry rebuilds itself from
# this set so any new ``singleton=True`` flag on an edge spec auto-
# propagates to the canonical writer's supersession path.
SINGLETON_EDGE_TYPES: frozenset[str] = frozenset(
    edge_type for edge_type, spec in EDGE_TYPES.items() if spec.singleton
)


def _sync_singleton_registry() -> None:
    """Reset the singleton-predicate registry to match the ontology.

    Called at module import so adding a ``singleton=True`` flag in
    ``EDGE_TYPES`` automatically applies to the canonical writer's
    supersession path. Tests that need a custom set should use
    :func:`domain.singleton_predicates.replace_singletons` then restore.
    """
    from domain.singleton_predicates import replace_singletons

    replace_singletons(SINGLETON_EDGE_TYPES)


_sync_singleton_registry()


def _build_entity_family_map(attr: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for label, spec in ENTITY_TYPES.items():
        value = getattr(spec, attr)
        if value:
            out[label] = value
    return out


# Internal label → public family name (for project map / structural reader).
ENTITY_PROJECT_MAP_FAMILY: dict[str, str] = _build_entity_family_map("project_map_family")
ENTITY_DEBUGGING_FAMILY: dict[str, str] = _build_entity_family_map("debugging_family")
ENTITY_FACT_FAMILY: dict[str, str] = _build_entity_family_map("fact_family")


# Aggregate include keys: bundles of family names. Used when an agent asks
# for a coarse cut like ``operations``. Data, not control flow — extending an
# aggregate is one row here.
INCLUDE_KEY_AGGREGATES: dict[str, tuple[str, ...]] = {
    "operations": (
        "deployments",
        "runbooks",
        "scripts",
        "config",
        "local_workflows",
    ),
    "debugging_memory": (
        "prior_fixes",
        "diagnostic_signals",
        "incidents",
        "alerts",
        "investigations",
    ),
    "agent_context": (
        "feature_map",
        "service_map",
        "decisions",
        "policies",
        "changes",
    ),
}


# Agent-contract include key → set of canonical labels that answer it.
# Resolution union: explicit ``include_keys`` on each entity spec PLUS the
# ``project_map_family`` and ``debugging_family`` attributes act as implicit
# include keys. To wire a new label into the agent contract, set one of those
# three fields on its spec — no edit needed downstream.
def _build_include_key_index() -> dict[str, tuple[str, ...]]:
    out: dict[str, set[str]] = {}
    for label, spec in ENTITY_TYPES.items():
        keys: set[str] = set(spec.include_keys)
        if spec.project_map_family:
            keys.add(spec.project_map_family)
        if spec.debugging_family:
            keys.add(spec.debugging_family)
        for key in keys:
            out.setdefault(key, set()).add(label)
    # Expand aggregates: a query for ``operations`` resolves to the union of
    # all labels under each child family.
    for aggregate, children in INCLUDE_KEY_AGGREGATES.items():
        bag = out.setdefault(aggregate, set())
        for child in children:
            bag.update(out.get(child, set()))
    return {key: tuple(sorted(labels)) for key, labels in out.items()}


INCLUDE_KEY_LABELS: dict[str, tuple[str, ...]] = _build_include_key_index()


# Fact-family → freshness / source-of-truth policy (used by graph_quality).
# Built by walking specs; first spec to declare a family wins on tie.
def _build_freshness_table() -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in ENTITY_TYPES.values():
        fam = spec.fact_family or "unknown"
        out.setdefault(fam, spec.freshness_ttl_hours)
    return out


def _build_sot_table() -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in ENTITY_TYPES.values():
        fam = spec.fact_family or "unknown"
        out.setdefault(fam, spec.source_of_truth)
    return out


FACT_FAMILY_FRESHNESS_TTL_HOURS: dict[str, int] = _build_freshness_table()
SOURCE_OF_TRUTH_POLICIES: dict[str, str] = _build_sot_table()
# Ensure the "unknown" fallback is always present.
FACT_FAMILY_FRESHNESS_TTL_HOURS.setdefault("unknown", 30 * DAY)
SOURCE_OF_TRUTH_POLICIES.setdefault("unknown", SOT_MEMORY)


# Classifier tables — built from spec text_patterns / property_signatures.
def _build_text_classifiers() -> tuple[tuple[str, re.Pattern[str]], ...]:
    out: list[tuple[str, re.Pattern[str]]] = []
    for label, spec in ENTITY_TYPES.items():
        for pattern in spec.text_patterns:
            out.append((label, re.compile(pattern, re.IGNORECASE)))
    return tuple(out)


ENTITY_TEXT_CLASSIFIERS: tuple[tuple[str, re.Pattern[str]], ...] = _build_text_classifiers()


def _build_property_signatures() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for label, spec in ENTITY_TYPES.items():
        for prop in spec.property_signatures:
            out.setdefault(prop, []).append(label)
    return {prop: tuple(sorted(labels)) for prop, labels in out.items()}


ENTITY_PROPERTY_SIGNATURES: dict[str, tuple[str, ...]] = _build_property_signatures()


# Legacy episodic-verb endpoint inference rules. Graphiti's LLM extractor
# emits these verb names even though they are not canonical edges in our
# catalog — without this table the classifier would lose label inference for
# real-world output. Each rule maps a normalized verb name + endpoint role to
# the canonical label that should be added to that endpoint.
#
# To register a new episodic-verb inference rule for a non-canonical verb,
# add a row here. Canonical edges (those in :data:`EDGE_TYPES`) declare
# inference via the ``source_inferred_labels`` / ``target_inferred_labels``
# fields on their spec instead.
EPISODIC_VERB_INFERRED_LABELS: dict[tuple[str, str], tuple[str, ...]] = {
    ("DECIDES_FOR", "target"): ("Decision",),
    ("AUTHORED_BY", "target"): ("Person",),
    ("PART_OF_FEATURE", "target"): ("Feature",),
    # Legacy lifecycle verbs that the LLM still emits; pin them to the
    # relevant Lifecycle-state-bearing entity types when unambiguous.
    ("DEPLOYED_TO", "target"): ("Deployment",),
}


# Edge endpoint inference table — built from spec source/target_inferred PLUS
# the legacy episodic-verb rules above.
def _build_edge_endpoint_inferred_labels() -> dict[tuple[str, str], tuple[str, ...]]:
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for edge_type, spec in EDGE_TYPES.items():
        if spec.source_inferred_labels:
            out[(edge_type, "source")] = spec.source_inferred_labels
        if spec.target_inferred_labels:
            out[(edge_type, "target")] = spec.target_inferred_labels
    # Legacy verbs that aren't canonical edges but still drive inference.
    for key, labels in EPISODIC_VERB_INFERRED_LABELS.items():
        out.setdefault(key, labels)
    return out


EDGE_ENDPOINT_INFERRED_LABELS: dict[tuple[str, str], tuple[str, ...]] = (
    _build_edge_endpoint_inferred_labels()
)


# Predicate families — built from spec predicate_family field.
def _build_predicate_family_edge_names() -> dict[str, frozenset[str]]:
    out: dict[str, set[str]] = {}
    for edge_type, spec in EDGE_TYPES.items():
        if spec.predicate_family:
            out.setdefault(spec.predicate_family, set()).add(edge_type)
    # Legacy episodic verbs still seen on the wire; keep them grouped so
    # auto-supersede catches them when the classifier hasn't normalized yet.
    out.setdefault("datastore_binding", set()).update(
        {"PERSISTS_TO", "MIGRATED_TO", "USES_DATA_STORE"}
    )
    out.setdefault("owner_binding", set()).update({"OWNED_BY", "MAINTAINED_BY"})
    out.setdefault("deployment_target", set()).update({"RUNS_ON", "HOSTED_ON"})
    out.setdefault("lifecycle_status", set()).update(
        {
            "PROPOSED",
            "IN_PROGRESS",
            "COMPLETED",
            "DEPRECATED",
            "DECOMMISSIONED",
            "PLANNED",
            "DELIVERED",
        }
    )
    return {fam: frozenset(names) for fam, names in out.items()}


PREDICATE_FAMILY_EDGE_NAMES: dict[str, frozenset[str]] = _build_predicate_family_edge_names()


def _union_entity_lifecycle_strings() -> frozenset[str]:
    out: set[str] = set()
    for spec in ENTITY_TYPES.values():
        out.update(spec.lifecycle_states)
    out.update(member.value for member in LifecycleStatus)
    return frozenset(out)


ALLOWED_LIFECYCLE_STATUSES: frozenset[str] = _union_entity_lifecycle_strings()


# --- Helpers ----------------------------------------------------------------


def normalize_graphiti_edge_name(name: str) -> str:
    """Normalize LLM / Graphiti relation labels for family lookup."""
    return name.strip().upper().replace(" ", "_").replace("-", "_")


def inferred_labels_for_episodic_edge_endpoint(
    edge_name: str, role: str
) -> tuple[str, ...]:
    """Canonical labels to add for a RELATES_TO endpoint, if unambiguous."""
    if role not in ("source", "target"):
        return ()
    key = (normalize_graphiti_edge_name(edge_name), role)
    return EDGE_ENDPOINT_INFERRED_LABELS.get(key, ())


# Hand-curated: targets that disambiguate ``CHOSE`` toward datastore binding.
_DATASTORE_CHOOSE_TARGET_LABEL_HINTS: frozenset[str] = frozenset({"DataStore"})


def predicate_family_for_edge_name(name: str) -> str | None:
    """Return predicate family id for a Graphiti ``RELATES_TO.name``, if any."""
    n = normalize_graphiti_edge_name(name)
    for fam, members in PREDICATE_FAMILY_EDGE_NAMES.items():
        if n in members:
            return fam
    return None


def predicate_family_for_episodic_supersede(
    edge_name: str,
    target_labels: Iterable[str] | None = None,
) -> str | None:
    """Predicate family for temporal auto-supersede / pairwise conflict bucketing.

    Graphiti may emit ``CHOSE`` for decisions not related to data stores; only
    join ``datastore_binding`` when the target has a DataStore hint.
    """
    n = normalize_graphiti_edge_name(edge_name)
    if n == "CHOSE":
        hinted = frozenset(canonical_entity_labels(target_labels or ()))
        if not hinted:
            return None
        if hinted & _DATASTORE_CHOOSE_TARGET_LABEL_HINTS:
            return "datastore_binding"
        return None
    return predicate_family_for_edge_name(edge_name)


def object_counterparty_uuid_for_edge(
    edge_name: str,
    source_uuid: str,
    target_uuid: str,
    *,
    predicate_family: str | None = None,
) -> str | None:
    """Endpoint uuid that differs when the same resource has conflicting bindings."""
    n = normalize_graphiti_edge_name(edge_name)
    fam = predicate_family if predicate_family is not None else predicate_family_for_edge_name(
        edge_name
    )
    if fam is None:
        return None
    if fam in {"datastore_binding", "deployment_target", "lifecycle_status"}:
        return target_uuid
    if fam == "owner_binding":
        if n == "OWNS":
            return source_uuid
        if n == "OWNED_BY":
            return target_uuid
        if n == "MAINTAINED_BY":
            return source_uuid
    return None


def temporal_subject_key_for_edge(
    edge_name: str,
    source_uuid: str,
    target_uuid: str,
    *,
    predicate_family: str | None = None,
) -> str | None:
    """Stable subject node uuid for contradiction grouping within a family."""
    n = normalize_graphiti_edge_name(edge_name)
    fam = predicate_family if predicate_family is not None else predicate_family_for_edge_name(
        edge_name
    )
    if fam is None:
        return None
    if fam in {"datastore_binding", "deployment_target", "lifecycle_status"}:
        return source_uuid
    if fam == "owner_binding":
        if n == "OWNS":
            return target_uuid
        if n == "OWNED_BY":
            return source_uuid
        if n == "MAINTAINED_BY":
            return target_uuid
    return None


def canonical_entity_labels(labels: Iterable[str]) -> tuple[str, ...]:
    """Return public canonical labels from a mixed Graphiti/Neo4j label set."""
    return tuple(label for label in labels if label in ENTITY_TYPES)


def is_canonical_entity_label(label: str) -> bool:
    return label in ENTITY_TYPES


def is_canonical_edge_type(edge_type: str) -> bool:
    return edge_type in EDGE_TYPES


def entity_spec(label: str) -> EntityTypeSpec | None:
    return ENTITY_TYPES.get(label)


def edge_spec(edge_type: str) -> EdgeTypeSpec | None:
    return EDGE_TYPES.get(edge_type)


def labels_for_include_key(include_key: str) -> tuple[str, ...]:
    """Canonical labels that answer an agent-facing ``include`` key.

    The agent contract is keyed by stable include strings. Internal labels
    can be renamed by editing the entity spec; the include key keeps
    answering for downstream consumers without code changes elsewhere.
    """
    return INCLUDE_KEY_LABELS.get(include_key, ())


def project_map_family_for_label(label: str) -> str | None:
    return ENTITY_PROJECT_MAP_FAMILY.get(label)


def debugging_family_for_label(label: str) -> str | None:
    return ENTITY_DEBUGGING_FAMILY.get(label)


def fact_family_for_label(label: str) -> str:
    return ENTITY_FACT_FAMILY.get(label, "unknown")


def is_scope_label(label: str) -> bool:
    return label in SCOPE_LABELS


def is_activity_label(label: str) -> bool:
    return label in ACTIVITY_LABELS


# --- Validation -------------------------------------------------------------


def validate_entity_upsert(item: EntityUpsert) -> list[str]:
    errors: list[str] = []
    if not item.entity_key or not item.entity_key.strip():
        errors.append("entity_key is required")

    labels = tuple(label for label in item.labels if label)
    if not labels:
        errors.append(
            f"{item.entity_key or '<missing>'}: at least one label is required"
        )
        return errors

    allowed_noncanonical = BASE_GRAPH_LABELS | CODE_GRAPH_LABELS
    unknown = sorted(
        label
        for label in labels
        if label not in ENTITY_TYPES and label not in allowed_noncanonical
    )
    if unknown:
        errors.append(
            f"{item.entity_key}: unknown canonical labels: {', '.join(unknown)}"
        )

    canonical = canonical_entity_labels(labels)
    if not canonical:
        errors.append(
            f"{item.entity_key}: at least one public canonical label is required"
        )

    for label in canonical:
        spec = ENTITY_TYPES[label]
        missing = sorted(
            prop for prop in spec.required_properties if prop not in item.properties
        )
        if missing:
            errors.append(
                f"{item.entity_key}:{label}: missing required properties: {', '.join(missing)}"
            )
        _validate_lifecycle_state(item.entity_key, label, spec, item.properties, errors)
        if label == "SourceReference":
            errors.extend(
                f"{item.entity_key}:{error}"
                for error in validate_source_reference_properties(item.properties)
            )

    return errors


def validate_edge_upsert(
    item: EdgeUpsert,
    entity_labels_by_key: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    return _validate_edge(
        item.edge_type,
        item.from_entity_key,
        item.to_entity_key,
        item.properties,
        entity_labels_by_key,
    )


def validate_edge_delete(
    item: EdgeDelete,
    entity_labels_by_key: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    return _validate_edge(
        item.edge_type,
        item.from_entity_key,
        item.to_entity_key,
        {},
        entity_labels_by_key,
    )


def validate_structural_mutations(
    entity_upserts: Iterable[EntityUpsert],
    edge_upserts: Iterable[EdgeUpsert],
    edge_deletes: Iterable[EdgeDelete],
) -> list[str]:
    errors: list[str] = []
    labels_by_key: dict[str, tuple[str, ...]] = {}

    for entity in entity_upserts:
        errors.extend(validate_entity_upsert(entity))
        if entity.entity_key:
            labels_by_key[entity.entity_key] = tuple(entity.labels)

    for edge in edge_upserts:
        errors.extend(validate_edge_upsert(edge, labels_by_key))

    for edge in edge_deletes:
        errors.extend(validate_edge_delete(edge, labels_by_key))

    return errors


def allowed_edge_types_between(
    from_labels: Iterable[str], to_labels: Iterable[str]
) -> tuple[str, ...]:
    return tuple(
        edge_type
        for edge_type, spec in EDGE_TYPES.items()
        if spec.allows(from_labels, to_labels)
    )


def _validate_edge(
    edge_type: str,
    from_entity_key: str,
    to_entity_key: str,
    properties: dict[str, object],
    entity_labels_by_key: dict[str, tuple[str, ...]] | None,
) -> list[str]:
    errors: list[str] = []
    if not edge_type or not edge_type.strip():
        errors.append("edge_type is required")
        return errors
    spec = EDGE_TYPES.get(edge_type)
    if spec is None:
        errors.append(f"{edge_type}: unknown canonical edge type")
        return errors

    if not from_entity_key or not from_entity_key.strip():
        errors.append(f"{edge_type}: from_entity_key is required")
    if not to_entity_key or not to_entity_key.strip():
        errors.append(f"{edge_type}: to_entity_key is required")

    missing = sorted(
        prop for prop in spec.required_properties if prop not in properties
    )
    if missing:
        errors.append(f"{edge_type}: missing required properties: {', '.join(missing)}")

    if entity_labels_by_key is None:
        return errors

    from_labels = entity_labels_by_key.get(from_entity_key)
    to_labels = entity_labels_by_key.get(to_entity_key)
    if (
        from_labels is not None
        and to_labels is not None
        and not spec.allows(from_labels, to_labels)
    ):
        allowed = ", ".join(f"{left}->{right}" for left, right in spec.allowed_pairs)
        errors.append(
            f"{edge_type}: invalid endpoint labels "
            f"{canonical_entity_labels(from_labels) or from_labels} -> "
            f"{canonical_entity_labels(to_labels) or to_labels}; allowed: {allowed}"
        )
    return errors


def _validate_lifecycle_state(
    entity_key: str,
    label: str,
    spec: EntityTypeSpec,
    properties: dict[str, object],
    errors: list[str],
) -> None:
    if not spec.lifecycle_states:
        return
    # Decision uses ``status`` (ADR-style); do not treat ``lifecycle_state``
    # from another co-located label as this label's lifecycle value.
    if label == "Decision":
        value = properties.get("status")
    else:
        value = properties.get("lifecycle_state") or properties.get("status")
    if value is None:
        return
    if str(value) not in spec.lifecycle_states:
        allowed = ", ".join(sorted(spec.lifecycle_states))
        errors.append(
            f"{entity_key}:{label}: invalid lifecycle/status {value!r}; allowed: {allowed}"
        )


def _normalized_label_set(labels: Iterable[str]) -> frozenset[str]:
    label_set = set(labels)
    if label_set & {"FILE", "FUNCTION", "CLASS", "NODE"}:
        label_set.add("CodeAsset")
    return frozenset(label_set)


def _endpoint_matches(endpoint: str, labels: frozenset[str]) -> bool:
    if endpoint == WILDCARD_ENDPOINT:
        return bool(labels)
    if endpoint == SCOPE_ENDPOINT:
        return bool(labels & SCOPE_LABELS)
    if endpoint == ACTIVITY_ENDPOINT:
        return bool(labels & ACTIVITY_LABELS)
    if endpoint == "CodeAsset":
        return bool(labels & CODE_GRAPH_LABELS)
    return endpoint in labels
