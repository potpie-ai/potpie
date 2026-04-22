"""Canonical Potpie context graph ontology.

This module is the governed vocabulary that sits above Graphiti extraction.
Graphiti can still ingest episodes flexibly, but deterministic structural
mutations should use the labels, relationship types, identity rules, and
validation helpers defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from domain.graph_mutations import EdgeDelete, EdgeUpsert, EntityUpsert
from domain.source_references import validate_source_reference_properties

ONTOLOGY_VERSION = "2026-04-phase-7"


class LifecycleStatus(StrEnum):
    """Edge-level lifecycle for episodic facts (stored as ``lifecycle_status`` on the edge)."""

    proposed = "proposed"
    planned = "planned"
    in_progress = "in_progress"
    completed = "completed"
    deprecated = "deprecated"
    decommissioned = "decommissioned"
    unknown = "unknown"


BASE_GRAPH_LABELS = frozenset({"Entity"})
CODE_GRAPH_LABELS = frozenset({"CodeAsset", "FILE", "FUNCTION", "CLASS", "NODE"})
WILDCARD_ENDPOINT = "*"


@dataclass(frozen=True, slots=True)
class EntityTypeSpec:
    label: str
    category: str
    description: str
    identity_policy: str
    required_properties: frozenset[str] = frozenset()
    lifecycle_states: frozenset[str] = frozenset()
    public: bool = True


@dataclass(frozen=True, slots=True)
class EdgeTypeSpec:
    edge_type: str
    description: str
    allowed_pairs: tuple[tuple[str, str], ...]
    required_properties: frozenset[str] = frozenset()
    public: bool = True

    def allows(self, from_labels: Iterable[str], to_labels: Iterable[str]) -> bool:
        from_set = _normalized_label_set(from_labels)
        to_set = _normalized_label_set(to_labels)
        return any(
            _endpoint_matches(left, from_set) and _endpoint_matches(right, to_set)
            for left, right in self.allowed_pairs
        )


def _spec(
    label: str,
    category: str,
    description: str,
    identity_policy: str,
    required: Iterable[str] = (),
    lifecycle: Iterable[str] = (),
) -> EntityTypeSpec:
    return EntityTypeSpec(
        label=label,
        category=category,
        description=description,
        identity_policy=identity_policy,
        required_properties=frozenset(required),
        lifecycle_states=frozenset(lifecycle),
    )


def _edge(
    edge_type: str,
    description: str,
    pairs: Iterable[tuple[str, str]],
    required: Iterable[str] = (),
) -> EdgeTypeSpec:
    return EdgeTypeSpec(
        edge_type=edge_type,
        description=description,
        allowed_pairs=tuple(pairs),
        required_properties=frozenset(required),
    )


COMMON_LIFECYCLE_STATES = frozenset(
    {"proposed", "active", "deprecated", "retired", "unknown"}
)
DECISION_STATES = frozenset(
    {"proposed", "accepted", "superseded", "rejected", "unknown"}
)
ISSUE_STATES = frozenset({"open", "closed", "triaged", "blocked", "unknown"})
INCIDENT_STATES = frozenset({"open", "mitigated", "resolved", "postmortem", "unknown"})


ENTITY_TYPES: dict[str, EntityTypeSpec] = {
    "Pot": _spec(
        "Pot",
        "scope_identity",
        "Isolation boundary for project context.",
        "pot slug",
        ("name",),
    ),
    "Repository": _spec(
        "Repository",
        "scope_identity",
        "Source repository mapped to a pot.",
        "provider host plus owner/name",
        ("name", "provider"),
        COMMON_LIFECYCLE_STATES,
    ),
    "System": _spec(
        "System",
        "scope_identity",
        "Product or platform boundary.",
        "pot-scoped slug",
        ("name",),
        COMMON_LIFECYCLE_STATES,
    ),
    "Service": _spec(
        "Service",
        "scope_identity",
        "Deployable runtime unit.",
        "pot-scoped service slug",
        ("name", "criticality", "lifecycle_state"),
        COMMON_LIFECYCLE_STATES,
    ),
    "Environment": _spec(
        "Environment",
        "scope_identity",
        "Local, staging, production, preview, or regional runtime environment.",
        "pot-scoped environment slug",
        ("name", "environment_type"),
        COMMON_LIFECYCLE_STATES,
    ),
    "DeploymentTarget": _spec(
        "DeploymentTarget",
        "delivery_operations",
        "Cloud, cluster, or hosting target.",
        "provider target id or scoped slug",
        ("name",),
    ),
    "DeploymentStrategy": _spec(
        "DeploymentStrategy",
        "delivery_operations",
        "Rollout strategy such as rolling, canary, or manual.",
        "scoped strategy slug",
        ("name", "strategy_type"),
    ),
    "Component": _spec(
        "Component",
        "product_architecture",
        "Logical subsystem, module, package, or bounded context.",
        "repo/service-scoped semantic key",
        ("name", "component_type"),
        COMMON_LIFECYCLE_STATES,
    ),
    "LegacyArtifact": _spec(
        "LegacyArtifact",
        "product_architecture",
        "Deprecated, superseded, or decommissioned resource still referenced in history.",
        "scoped legacy slug or name",
        ("name", "status"),
        COMMON_LIFECYCLE_STATES,
    ),
    "Capability": _spec(
        "Capability",
        "product_architecture",
        "External product behavior or capability.",
        "system-scoped capability slug",
        ("name",),
        COMMON_LIFECYCLE_STATES,
    ),
    "Feature": _spec(
        "Feature",
        "product_architecture",
        "Concrete deliverable area within a capability.",
        "capability/system-scoped feature slug",
        ("name",),
        COMMON_LIFECYCLE_STATES,
    ),
    "Functionality": _spec(
        "Functionality",
        "product_architecture",
        "Granular behavior under a feature.",
        "feature-scoped functionality slug",
        ("name",),
        COMMON_LIFECYCLE_STATES,
    ),
    "Requirement": _spec(
        "Requirement",
        "product_architecture",
        "Expected behavior or acceptance criterion.",
        "source ref or scoped requirement slug",
        ("statement", "status"),
    ),
    "RoadmapItem": _spec(
        "RoadmapItem",
        "product_architecture",
        "Planned evolution or future direction.",
        "planning-source id or scoped slug",
        ("title", "status"),
    ),
    "Interface": _spec(
        "Interface",
        "product_architecture",
        "API, event contract, queue, webhook, or database contract.",
        "component/service-scoped interface slug",
        ("name", "interface_type"),
    ),
    "DataStore": _spec(
        "DataStore",
        "product_architecture",
        "Database, cache, object store, or SaaS storage.",
        "service/system-scoped datastore slug",
        ("name", "store_type"),
    ),
    "Integration": _spec(
        "Integration",
        "product_architecture",
        "External API, SDK, webhook, MCP, queue, or cloud service.",
        "provider/integration slug",
        ("name", "integration_type"),
    ),
    "Dependency": _spec(
        "Dependency",
        "product_architecture",
        "External system, service, or library with operational significance.",
        "package/system scoped dependency id",
        ("name", "dependency_type"),
    ),
    "Person": _spec(
        "Person",
        "team_ownership",
        "Contributor or stakeholder.",
        "provider user id, email, or pot-scoped person slug",
        ("name",),
    ),
    "Agent": _spec(
        "Agent",
        "team_ownership",
        "Automated coding agent, IDE agent, or service agent working in a pot.",
        "agent provider plus agent id or pot-scoped slug",
        ("name", "agent_type"),
    ),
    "Team": _spec(
        "Team",
        "team_ownership",
        "Functional, product, or ownership team.",
        "pot-scoped team slug",
        ("name",),
        COMMON_LIFECYCLE_STATES,
    ),
    "Role": _spec(
        "Role",
        "team_ownership",
        "On-call, tech lead, owner, reviewer, or maintainer role.",
        "scope plus role slug",
        ("name", "role_type"),
    ),
    "Change": _spec(
        "Change",
        "change_decision",
        "Generic parent concept for important change events.",
        "source id or generated change id",
        ("title", "change_type"),
    ),
    "PullRequest": _spec(
        "PullRequest",
        "change_decision",
        "Source-control pull request.",
        "provider/repo/pr number",
        ("pr_number", "title"),
    ),
    "Commit": _spec(
        "Commit",
        "change_decision",
        "Source-control commit.",
        "provider/repo/sha",
        ("sha",),
    ),
    "Issue": _spec(
        "Issue",
        "change_decision",
        "Ticket, issue, bug, or planning item.",
        "provider issue key/id",
        ("title", "status"),
        ISSUE_STATES,
    ),
    "Decision": _spec(
        "Decision",
        "change_decision",
        "Canonical engineering or product decision.",
        "source ref or scoped decision slug",
        ("title", "summary", "status"),
        DECISION_STATES,
    ),
    "Constraint": _spec(
        "Constraint",
        "change_decision",
        "Rule, architecture constraint, compliance rule, or do-not-do guidance.",
        "scoped constraint slug",
        ("statement", "constraint_type", "status"),
    ),
    "Preference": _spec(
        "Preference",
        "change_decision",
        "Team or project style and workflow preference.",
        "scope plus preference slug",
        ("statement", "preference_type", "scope_kind"),
    ),
    "AgentInstruction": _spec(
        "AgentInstruction",
        "change_decision",
        "AGENTS.md, skill, prompt, MCP guidance, or agent-facing instruction.",
        "source ref plus section id",
        ("title", "instruction_type"),
    ),
    "LocalWorkflow": _spec(
        "LocalWorkflow",
        "change_decision",
        "How people run, test, debug, or deploy locally.",
        "repo/service-scoped workflow slug",
        ("name", "workflow_type"),
    ),
    "CodeAsset": _spec(
        "CodeAsset",
        "code_topology",
        "File, function, class, or other code asset referenced by context facts.",
        "provider/repo/path/symbol identity",
        ("name", "asset_type"),
    ),
    "Document": _spec(
        "Document",
        "knowledge_evidence",
        "ADR, product doc, design doc, runbook doc, or wiki page.",
        "source document id or URL",
        ("title", "source_uri"),
    ),
    "Conversation": _spec(
        "Conversation",
        "knowledge_evidence",
        "Slack thread, review discussion, planning thread, or incident thread.",
        "source conversation id",
        ("title", "source_uri"),
    ),
    "Episode": _spec(
        "Episode",
        "knowledge_evidence",
        "Graphiti ingested episode; narrative source.",
        "Graphiti episode uuid",
        ("name",),
    ),
    "Observation": _spec(
        "Observation",
        "knowledge_evidence",
        "Normalized evidence unit.",
        "source ref plus observation slug",
        ("summary", "observed_at"),
    ),
    "SourceSystem": _spec(
        "SourceSystem",
        "knowledge_evidence",
        "GitHub, Linear, Slack, Docs, Sentry, GCP, AWS, or another provider.",
        "provider slug",
        ("name", "source_type"),
    ),
    "SourceReference": _spec(
        "SourceReference",
        "knowledge_evidence",
        "Stable pointer to an external artifact with resolver hints and freshness metadata.",
        "source system plus external id",
        ("source_system", "external_id", "ref_type"),
    ),
    "Deployment": _spec(
        "Deployment",
        "delivery_operations",
        "Version or branch promoted into an environment.",
        "target/environment/source deployment id",
        ("version", "deployed_at"),
    ),
    "Branch": _spec(
        "Branch",
        "delivery_operations",
        "Git branch with operational meaning.",
        "provider/repo/branch name",
        ("name",),
    ),
    "Alert": _spec(
        "Alert",
        "delivery_operations",
        "Monitoring or incident signal.",
        "source alert id or fingerprint",
        ("title", "severity", "status"),
    ),
    "Incident": _spec(
        "Incident",
        "delivery_operations",
        "Operational issue with timeline and severity.",
        "source incident id or scoped slug",
        ("title", "severity", "status"),
        INCIDENT_STATES,
    ),
    "Metric": _spec(
        "Metric",
        "delivery_operations",
        "Named health indicator.",
        "service/environment metric name",
        ("name", "metric_type"),
    ),
    "Runbook": _spec(
        "Runbook",
        "delivery_operations",
        "Human-usable remediation procedure.",
        "source doc id or scoped runbook slug",
        ("title",),
    ),
    "Script": _spec(
        "Script",
        "delivery_operations",
        "Local, CI, debug, or deployment command used by the team.",
        "repo path or scoped script slug",
        ("name", "command"),
    ),
    "ConfigVariable": _spec(
        "ConfigVariable",
        "delivery_operations",
        "Important config variable or secret reference, never secret value.",
        "scope plus variable name",
        ("name", "scope_kind"),
    ),
    "BugPattern": _spec(
        "BugPattern",
        "debugging_reliability",
        "Repeated failure mode or symptom cluster.",
        "scope plus symptom signature",
        ("summary",),
    ),
    "Investigation": _spec(
        "Investigation",
        "debugging_reliability",
        "Debugging session, diagnostic path, or incident investigation.",
        "source session/incident id or generated id",
        ("summary", "status"),
    ),
    "Fix": _spec(
        "Fix",
        "debugging_reliability",
        "Resolution, mitigation, workaround, or permanent code/config change.",
        "source ref or generated fix id",
        ("summary", "fix_type"),
    ),
    "DiagnosticSignal": _spec(
        "DiagnosticSignal",
        "debugging_reliability",
        "Error signature, log query, metric, alert fingerprint, or symptom.",
        "scope plus signal fingerprint",
        ("signal_type", "summary"),
    ),
    "QualityIssue": _spec(
        "QualityIssue",
        "quality_drift",
        "Detected graph quality, freshness, source sync, alias, orphan, or bridge issue.",
        "pot/scope plus issue code and affected entity",
        ("code", "severity", "status"),
        COMMON_LIFECYCLE_STATES,
    ),
    "MaintenanceJob": _spec(
        "MaintenanceJob",
        "quality_drift",
        "Recurring verification, repair, cleanup, or materialization job.",
        "job family plus scope id",
        ("job_type", "status"),
        COMMON_LIFECYCLE_STATES,
    ),
    "MaterializedAccessPath": _spec(
        "MaterializedAccessPath",
        "quality_drift",
        "Compact derived path maintained for repeated agent queries.",
        "pattern type plus scope id",
        ("name", "pattern_type"),
        COMMON_LIFECYCLE_STATES,
    ),
}


EDGE_TYPES: dict[str, EdgeTypeSpec] = {
    "SCOPES": _edge(
        "SCOPES",
        "Pot scopes project context entities.",
        [
            ("Pot", "Repository"),
            ("Pot", "System"),
            ("Pot", "Service"),
            ("Pot", "Environment"),
            ("Pot", "Feature"),
            ("Pot", "Component"),
            ("Pot", "Document"),
            ("Pot", "Person"),
            ("Pot", "Team"),
            ("Pot", "DeploymentTarget"),
            ("Pot", "Integration"),
        ],
    ),
    "CONTAINS": _edge(
        "CONTAINS",
        "System, repository, service, or component contains lower-level project context.",
        [
            ("System", "Service"),
            ("System", "Component"),
            ("System", "Capability"),
            ("Repository", "Component"),
            ("Service", "Component"),
            ("Component", "Component"),
        ],
    ),
    "BACKED_BY": _edge(
        "BACKED_BY",
        "Service or component is backed by repository.",
        [("Service", "Repository"), ("Component", "Repository")],
    ),
    "DEPLOYED_TO": _edge(
        "DEPLOYED_TO",
        "Service or deployment runs in environment.",
        [("Service", "Environment")],
    ),
    "HOSTS": _edge("HOSTS", "Environment hosts service.", [("Environment", "Service")]),
    "HOSTED_ON": _edge(
        "HOSTED_ON",
        "Environment is hosted on deployment target.",
        [("Environment", "DeploymentTarget")],
    ),
    "IMPLEMENTS": _edge(
        "IMPLEMENTS", "Feature implements capability.", [("Feature", "Capability")]
    ),
    "HAS_FUNCTIONALITY": _edge(
        "HAS_FUNCTIONALITY",
        "Feature has granular functionality.",
        [("Feature", "Functionality")],
    ),
    "DEFINES": _edge(
        "DEFINES",
        "Requirement defines feature or functionality.",
        [("Requirement", "Feature"), ("Requirement", "Functionality")],
    ),
    "EVOLVES": _edge(
        "EVOLVES",
        "Roadmap item evolves feature or capability.",
        [("RoadmapItem", "Feature"), ("RoadmapItem", "Capability")],
    ),
    "SUPPORTS": _edge(
        "SUPPORTS",
        "Component supports feature or observation supports fact.",
        [
            ("Component", "Feature"),
            ("Observation", "Decision"),
            ("Observation", "Incident"),
            ("Observation", "Constraint"),
        ],
    ),
    "EXPOSES": _edge(
        "EXPOSES", "Component exposes interface.", [("Component", "Interface")]
    ),
    "USES": _edge(
        "USES",
        "Component or service uses integration/dependency.",
        [
            ("Component", "Integration"),
            ("Component", "Dependency"),
            ("Service", "Integration"),
            ("Service", "Dependency"),
        ],
    ),
    "DEPENDS_ON": _edge(
        "DEPENDS_ON",
        "Component or service depends on another dependency/service.",
        [
            ("Component", "Dependency"),
            ("Component", "Service"),
            ("Service", "Dependency"),
            ("Service", "Service"),
        ],
    ),
    "USES_DATA_STORE": _edge(
        "USES_DATA_STORE", "Service uses data store.", [("Service", "DataStore")]
    ),
    "USES_DEPLOYMENT_STRATEGY": _edge(
        "USES_DEPLOYMENT_STRATEGY",
        "Service uses deployment strategy.",
        [("Service", "DeploymentStrategy")],
    ),
    "CALLS": _edge("CALLS", "Service calls another service.", [("Service", "Service")]),
    "OWNS": _edge(
        "OWNS",
        "Person or team owns project context.",
        [
            ("Person", "Service"),
            ("Person", "Component"),
            ("Person", "Feature"),
            ("Person", "Repository"),
            ("Team", "Service"),
            ("Team", "Component"),
            ("Team", "Capability"),
            ("Team", "Feature"),
            ("Team", "Runbook"),
            ("Team", "Repository"),
        ],
    ),
    "MEMBER_OF": _edge("MEMBER_OF", "Person belongs to team.", [("Person", "Team")]),
    "REVIEWS": _edge("REVIEWS", "Person reviews change.", [("Person", "Change")]),
    "ONCALL_FOR": _edge(
        "ONCALL_FOR",
        "Person is on call for service or environment.",
        [("Person", "Service"), ("Person", "Environment")],
    ),
    "OWNS_FILE": _edge(
        "OWNS_FILE", "Component owns code file.", [("Component", "CodeAsset")]
    ),
    "TOUCHES_CODE": _edge(
        "TOUCHES_CODE", "Feature touches code asset.", [("Feature", "CodeAsset")]
    ),
    "AFFECTS": _edge(
        "AFFECTS",
        "Decision affects product or architecture context.",
        [
            ("Decision", "Feature"),
            ("Decision", "Component"),
            ("Decision", "Service"),
            ("Decision", "CodeAsset"),
        ],
    ),
    "AFFECTS_CODE": _edge(
        "AFFECTS_CODE", "Decision affects code asset.", [("Decision", "CodeAsset")]
    ),
    "ADDRESSES": _edge(
        "ADDRESSES",
        "Pull request addresses issue.",
        [("PullRequest", "Issue"), ("Change", "Issue")],
    ),
    "HAS_COMMIT": _edge(
        "HAS_COMMIT",
        "Pull request contains a commit.",
        [("PullRequest", "Commit")],
    ),
    "HAS_REVIEW_DECISION": _edge(
        "HAS_REVIEW_DECISION",
        "Pull request review discussion produced a decision.",
        [("PullRequest", "Decision")],
    ),
    "PART_OF": _edge(
        "PART_OF",
        "Change hierarchy membership.",
        [("PullRequest", "Change"), ("Commit", "PullRequest")],
    ),
    "MADE_IN": _edge(
        "MADE_IN",
        "Decision was made in source context.",
        [
            ("Decision", "PullRequest"),
            ("Decision", "Incident"),
            ("Decision", "Document"),
        ],
    ),
    "APPLIES_TO": _edge(
        "APPLIES_TO",
        "Constraint applies to scope.",
        [
            ("Constraint", "Service"),
            ("Constraint", "Component"),
            ("Constraint", "Feature"),
            ("Constraint", "Repository"),
        ],
    ),
    "PREFERRED_FOR": _edge(
        "PREFERRED_FOR",
        "Preference applies to scope.",
        [
            ("Preference", "Repository"),
            ("Preference", "Component"),
            ("Preference", "Team"),
        ],
    ),
    "INFORMS": _edge(
        "INFORMS",
        "Agent instruction informs a working scope.",
        [
            ("AgentInstruction", "Repository"),
            ("AgentInstruction", "Service"),
            ("AgentInstruction", "Feature"),
            ("AgentInstruction", "Agent"),
        ],
    ),
    "RUNS": _edge(
        "RUNS",
        "Workflow or script runs against project scope.",
        [
            ("LocalWorkflow", "Service"),
            ("LocalWorkflow", "Component"),
            ("LocalWorkflow", "Repository"),
            ("Script", "Service"),
            ("Script", "Component"),
        ],
    ),
    "TARGETS": _edge(
        "TARGETS", "Deployment targets environment.", [("Deployment", "Environment")]
    ),
    "DEPLOYED_AS": _edge(
        "DEPLOYED_AS", "Branch deployed as deployment.", [("Branch", "Deployment")]
    ),
    "CONFIGURES": _edge(
        "CONFIGURES",
        "Config variable configures service or environment.",
        [("ConfigVariable", "Service"), ("ConfigVariable", "Environment")],
    ),
    "MITIGATES": _edge(
        "MITIGATES",
        "Runbook or fix mitigates incident.",
        [
            ("Runbook", "Incident"),
            ("Runbook", "BugPattern"),
            ("Fix", "Incident"),
            ("Fix", "BugPattern"),
        ],
    ),
    "IMPACTS": _edge(
        "IMPACTS",
        "Incident or alert impacts service or environment.",
        [("Incident", "Service"), ("Alert", "Service"), ("Alert", "Environment")],
    ),
    "FIRED_IN": _edge(
        "FIRED_IN", "Alert fired in environment.", [("Alert", "Environment")]
    ),
    "INDICATES": _edge(
        "INDICATES", "Alert indicates incident.", [("Alert", "Incident")]
    ),
    "MATCHES_PATTERN": _edge(
        "MATCHES_PATTERN", "Incident matches bug pattern.", [("Incident", "BugPattern")]
    ),
    "DEBUGGED": _edge(
        "DEBUGGED",
        "Investigation debugged incident or pattern.",
        [("Investigation", "Incident"), ("Investigation", "BugPattern")],
    ),
    "OBSERVED_IN": _edge(
        "OBSERVED_IN",
        "Diagnostic signal was observed in investigation or incident.",
        [("DiagnosticSignal", "Investigation"), ("DiagnosticSignal", "Incident")],
    ),
    "RESOLVED": _edge(
        "RESOLVED",
        "Fix resolved incident or bug pattern.",
        [("Fix", "Incident"), ("Fix", "BugPattern")],
    ),
    "CHANGED_BY": _edge(
        "CHANGED_BY",
        "Fix was changed by PR or commit.",
        [("Fix", "PullRequest"), ("Fix", "Commit")],
    ),
    "HAS_SIGNAL": _edge(
        "HAS_SIGNAL",
        "Bug pattern, incident, or investigation has a diagnostic signal.",
        [
            ("BugPattern", "DiagnosticSignal"),
            ("Incident", "DiagnosticSignal"),
            ("Investigation", "DiagnosticSignal"),
        ],
    ),
    "HAS_ROOT_CAUSE": _edge(
        "HAS_ROOT_CAUSE",
        "Investigation or incident identified a root-cause observation.",
        [("Investigation", "Observation"), ("Incident", "Observation")],
    ),
    "SEEN_IN": _edge(
        "SEEN_IN",
        "Bug pattern seen in scope.",
        [
            ("BugPattern", "Service"),
            ("BugPattern", "Environment"),
            ("BugPattern", "Component"),
        ],
    ),
    "DESCRIBES": _edge(
        "DESCRIBES",
        "Episode or document describes context.",
        [
            ("Episode", "Change"),
            ("Episode", "Incident"),
            ("Episode", "Decision"),
            ("Episode", "Document"),
            ("Document", "Feature"),
            ("Document", "Component"),
            ("Document", "Constraint"),
        ],
    ),
    "RESULTED_IN": _edge(
        "RESULTED_IN",
        "Conversation resulted in decision.",
        [("Conversation", "Decision")],
    ),
    "EVIDENCED_BY": _edge(
        "EVIDENCED_BY",
        "Canonical fact is evidenced by source reference.",
        [(WILDCARD_ENDPOINT, "SourceReference")],
    ),
    "FROM_SOURCE": _edge(
        "FROM_SOURCE",
        "Source reference came from source system.",
        [("SourceReference", "SourceSystem")],
    ),
    "MODIFIED": _edge(
        "MODIFIED", "Pull request modified code asset.", [("PullRequest", "CodeAsset")]
    ),
    "INVOLVES_CODE": _edge(
        "INVOLVES_CODE", "Incident involves code asset.", [("Incident", "CodeAsset")]
    ),
    "REFERENCES_CODE": _edge(
        "REFERENCES_CODE", "Runbook references code asset.", [("Runbook", "CodeAsset")]
    ),
    "ALIASES": _edge(
        "ALIASES", "Entity has an alias.", [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)]
    ),
    "RENAMED_FROM": _edge(
        "RENAMED_FROM",
        "Entity was renamed from prior identity.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "MERGED_FROM": _edge(
        "MERGED_FROM",
        "Entity was merged from another identity.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "SPLIT_FROM": _edge(
        "SPLIT_FROM",
        "Entity was split from another identity.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "FLAGS": _edge(
        "FLAGS",
        "Quality issue flags an affected entity or source reference.",
        [("QualityIssue", WILDCARD_ENDPOINT)],
    ),
    "REPAIRS": _edge(
        "REPAIRS",
        "Maintenance job repairs or verifies an affected entity.",
        [("MaintenanceJob", WILDCARD_ENDPOINT)],
    ),
    "MATERIALIZES": _edge(
        "MATERIALIZES",
        "Materialized access path precomputes a query path for an entity.",
        [("MaterializedAccessPath", WILDCARD_ENDPOINT)],
    ),
    "SUPERSEDES": _edge(
        "SUPERSEDES",
        "A fact or entity supersedes a prior version; the target is soft-invalidated with valid_to.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    # Episodic extraction verbs (Graphiti); wildcard endpoints — see docs/context-graph-improvements/02.
    "GENERIC_ACTION": _edge(
        "GENERIC_ACTION",
        "Fallback when the extractor cannot name a specific predicate; carries lifecycle_status.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "MIGRATED_TO": _edge(
        "MIGRATED_TO",
        "Workload or datastore migration to a new system or store.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "DECOMMISSIONED": _edge(
        "DECOMMISSIONED",
        "Resource, cluster, or path was shut down or removed from service.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "PLANNED": _edge(
        "PLANNED",
        "Future or scheduled work not yet delivered.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "DELIVERED": _edge(
        "DELIVERED",
        "Delivered or finished change (past tense, shipped).",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "DEPRECATED": _edge(
        "DEPRECATED",
        "API, component, or path is deprecated or slated for removal.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "REPLACES": _edge(
        "REPLACES",
        "New component or system replaces an older one.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "CAUSED": _edge(
        "CAUSED",
        "Causal link between events or changes.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "FIXES": _edge(
        "FIXES",
        "Change or release fixes a defect or incident.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "ADDED_TO": _edge(
        "ADDED_TO",
        "Capability, instrumentation, or dependency was added to a scope.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "REMOVED_FROM": _edge(
        "REMOVED_FROM",
        "Something was removed from a system, path, or dependency set.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "RELATED_TO": _edge(
        "RELATED_TO",
        "Catch-all when the extractor emits a non-catalog edge type; preserve "
        "``original_edge_type`` on edge properties.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "STORED_IN": _edge(
        "STORED_IN",
        "Data is persisted in or primarily associated with a store or database.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
    "DECIDES_FOR": _edge(
        "DECIDES_FOR",
        "Decision or policy governs a scope or component.",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
    ),
}


CANONICAL_LABELS: frozenset[str] = frozenset(ENTITY_TYPES.keys())
CANONICAL_EDGE_TYPES: frozenset[str] = frozenset(EDGE_TYPES.keys())


def _union_entity_lifecycle_strings() -> frozenset[str]:
    out: set[str] = set()
    for spec in ENTITY_TYPES.values():
        out.update(spec.lifecycle_states)
    out.update(member.value for member in LifecycleStatus)
    return frozenset(out)


ALLOWED_LIFECYCLE_STATUSES: frozenset[str] = _union_entity_lifecycle_strings()


# --- Temporal predicate families (auto-supersession + search ranking) ------------
# See docs/context-graph-improvements/01-temporal-resolution-in-search.md


def normalize_graphiti_edge_name(name: str) -> str:
    """Normalize LLM / Graphiti relation labels for family lookup."""
    return name.strip().upper().replace(" ", "_").replace("-", "_")


# Episodic edge endpoint → canonical node labels (see docs/context-graph-improvements/03).
# Key: (normalized RELATES_TO.name, "source" | "target") → labels to add on that endpoint.
# Omit ambiguous rows (multiple equally valid ontology labels → no inference).
EDGE_ENDPOINT_INFERRED_LABELS: dict[tuple[str, str], tuple[str, ...]] = {
    ("DECIDES_FOR", "target"): ("Decision",),
    ("CAUSED", "target"): ("Incident",),
    ("DEPLOYED_TO", "target"): ("Deployment",),
    ("DEPRECATED", "target"): ("LegacyArtifact",),
    ("DECOMMISSIONED", "target"): ("LegacyArtifact",),
}


def inferred_labels_for_episodic_edge_endpoint(
    edge_name: str, role: str
) -> tuple[str, ...]:
    """Return canonical ontology labels to add for a RELATES_TO endpoint, if unambiguous."""
    if role not in ("source", "target"):
        return ()
    key = (normalize_graphiti_edge_name(edge_name), role)
    return EDGE_ENDPOINT_INFERRED_LABELS.get(key, ())


# Hand-curated: same family + same logical subject + different object ⇒ contradiction.
PREDICATE_FAMILY_EDGE_NAMES: dict[str, frozenset[str]] = {
    "datastore_binding": frozenset(
        {
            "USES_DATA_STORE",
            "STORED_IN",
            "PERSISTS_TO",
            "MIGRATED_TO",
        }
    ),
    "owner_binding": frozenset({"OWNS", "OWNED_BY", "MAINTAINED_BY"}),
    "deployment_target": frozenset({"DEPLOYED_TO", "RUNS_ON", "HOSTED_ON"}),
    "lifecycle_status": frozenset(
        {
            "PROPOSED",
            "IN_PROGRESS",
            "COMPLETED",
            "DEPRECATED",
            "DECOMMISSIONED",
        }
    ),
}


def predicate_family_for_edge_name(name: str) -> str | None:
    """Return predicate family id for a Graphiti ``RELATES_TO.name``, if any."""
    n = normalize_graphiti_edge_name(name)
    for fam, members in PREDICATE_FAMILY_EDGE_NAMES.items():
        if n in members:
            return fam
    return None


# Targets that disambiguate ``CHOSE`` toward datastore / persistence binding (see fix 02).
_DATASTORE_CHOOSE_TARGET_LABEL_HINTS: frozenset[str] = frozenset({"DataStore"})


def predicate_family_for_episodic_supersede(
    edge_name: str,
    target_labels: Iterable[str] | None = None,
) -> str | None:
    """Predicate family for temporal auto-supersede and pairwise conflict bucketing.

    Graphiti may emit ``CHOSE`` for decisions that are not datastore-related; those must not
    join ``datastore_binding``. When the target carries no canonical hints, ``CHOSE`` is
    excluded (strict fallback—matches single-type supersession only for that edge).
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
    """Endpoint uuid whose identity differs when the same resource has a conflicting binding."""

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
    """Stable subject node uuid for contradiction grouping within a family.

    * datastore / deployment / lifecycle: subject is the ``source`` entity.
    * owner edges: the *resource* whose ownership is described (OWNS: target;
      OWNED_BY / MAINTAINED_BY: source when the edge is Resource -> Actor).
    """
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
    """Validate deterministic mutations against the public ontology.

    Edge endpoint compatibility is enforced when both endpoints are present in
    the same mutation batch. If an endpoint already exists in the graph and is
    not part of the current batch, Phase 1 only validates the relationship type;
    the write adapter can perform graph-backed endpoint checks later.
    """
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
    # Decision uses ``status`` (ADR-style); do not treat ``lifecycle_state`` from another
    # co-located label (e.g. Service) as this label's lifecycle value.
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
    if endpoint == "CodeAsset":
        return bool(labels & CODE_GRAPH_LABELS)
    return endpoint in labels
