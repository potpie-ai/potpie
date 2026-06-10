"""Canonical Potpie context graph ontology — unified topology + memory.

Single declarative catalog covering everything an agent can read or write.
Three sections, all spec-driven:

1. :data:`ENTITY_TYPES` — every label that can live in the graph, with the
   identity class and key prefix that mint it, plus fact family, freshness
   TTL, classifier cues, and project-map / debugging family mappings.
2. :data:`EDGE_TYPES` — every predicate that can ride on a ``:RELATES_TO``
   claim, grouped by ``category`` (``topology`` / ``ownership`` / ``people``
   / ``timeline`` / ``memory`` / ``generic``).
3. :data:`RECORD_TYPES` — the agent-facing context_record vocabulary, each
   row joining a record type to its anchor entity, emitted predicate,
   payload schema, and reader include key. This is the table that unifies
   the agent surface (record types + include families) with the graph
   ontology (entities + predicates).

Every other module — canonical writer, structural reader, reconciliation
validation, graph-quality policy, agent context port — derives its behavior
from these three catalogs rather than carrying parallel vocabularies.

**Adding an entity:** one row in :data:`ENTITY_TYPES`.
**Adding a predicate:** one row in :data:`EDGE_TYPES`.
**Adding a record type:** one row in :data:`RECORD_TYPES`.

The identity registry (:mod:`domain.identity`) is a *view* over
:data:`ENTITY_TYPES` (populated at the bottom of this module). The agent
context port's record-type and include vocabularies are *views* over
:data:`RECORD_TYPES`. :mod:`domain.coherence` runs import-time invariants
that fail loud if any two views disagree.

Design pillars:

1. **An entity exists only if an edge needs it as an endpoint.** No
   aspirational nodes; if nothing connects to it and no query traverses it,
   it is a property, not a node.
2. **Identity-only required fields.** A node needs its ``entity_key`` and a
   label; everything else is optional enrichment. No required-property
   downgrade traps.
3. **One name per edge, declared once.** The edge vocabulary *is* the schema;
   readers and writers traverse the same names.
4. **A ``Scope`` interface, not a base class.** Entities flagged
   ``scope=True`` (``Repository``/``Service``/``Environment``/``DataStore``/
   ``Cluster``) act as scope endpoints for cross-cutting edges (memory
   claims, ownership) without each edge enumerating them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from domain.graph_contract import ONTOLOGY_VERSION
from domain.graph_mutations import EdgeDelete, EdgeUpsert, EntityUpsert
from domain.identity import IdentityClass, IdentitySpec, register_identity

# ``ONTOLOGY_VERSION`` is owned by :mod:`domain.graph_contract` (the single
# contract home) and mirrored here so existing importers of
# ``domain.ontology.ONTOLOGY_VERSION`` and the graph catalog never disagree.
__all_ontology_version__ = ONTOLOGY_VERSION


# --- Lifecycle vocabulary ---------------------------------------------------


class LifecycleStatus(StrEnum):
    """Edge-level lifecycle for facts that carry ``lifecycle_status``.

    Retained as a stable vocabulary for the claim layer added later; the
    topology core does not use lifecycle edges.
    """

    proposed = "proposed"
    planned = "planned"
    in_progress = "in_progress"
    completed = "completed"
    deprecated = "deprecated"
    decommissioned = "decommissioned"
    unknown = "unknown"


# --- Source-of-truth / evidence vocabulary ---------------------------------


SOURCE_OF_TRUTH_AUTHORITATIVE_EXTERNAL = "authoritative_external_truth"
SOURCE_OF_TRUTH_AUTHORITATIVE_CODE = "authoritative_code_truth"
SOURCE_OF_TRUTH_CANONICAL_MEMORY = "canonicalized_memory"
SOURCE_OF_TRUTH_SOFT_INFERENCE = "soft_inference"

EVIDENCE_STRENGTHS = ("deterministic", "attested", "inferred", "hypothesized")
DEFAULT_EVIDENCE_STRENGTH = "inferred"


# --- Graph labels & endpoints ----------------------------------------------


BASE_GRAPH_LABELS = frozenset({"Entity"})
# Legacy code-graph labels still in the data; ``CodeAsset`` is the canonical
# bridge. The validator treats any of these as a CodeAsset endpoint.
CODE_GRAPH_LABELS = frozenset({"CodeAsset", "FILE", "FUNCTION", "CLASS", "NODE"})
WILDCARD_ENDPOINT = "*"
SCOPE_ENDPOINT = "@Scope"  # resolves to any entity flagged ``scope=True``
ACTIVITY_ENDPOINT = (
    "@Activity"  # resolves to Activity or any entity flagged ``is_activity=True``
)


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
    # --- Identity --- (binds to domain.identity.IdentitySpec) --------------
    identity_class: IdentityClass
    """How identity is established: EXTERNAL_ID / SLUG_ALIAS / CONTENT_HASH."""

    key_prefix: str
    """Canonical key prefix (e.g. ``service`` → ``service:auth-svc``)."""

    identity_policy: str
    """Human-readable identity sketch (e.g. ``service:<name>``) for docs/errors."""

    authoritative_source: str | None = None
    """For EXTERNAL_ID identities: source system whose id wins (e.g. ``github``)."""

    required_properties: frozenset[str] = frozenset()
    lifecycle_states: frozenset[str] = frozenset()
    public: bool = True

    # --- Structural traits ---------------------------------------------------
    scope: bool = False
    """True for Repository/Service/Environment/DataStore/Cluster — scope endpoints."""

    is_activity: bool = False
    """True for rich change events — multi-labels with Activity for timeline."""

    # --- Agent-facing family mapping ----------------------------------------
    project_map_family: str | None = None
    """Which family this label answers in the project_map response."""

    debugging_family: str | None = None
    """Which family in the debugging_memory response."""

    # --- Source-of-truth / freshness ---------------------------------------
    fact_family: str = "unknown"
    """Identifier for the SoT family (topology, ownership, …)."""

    source_of_truth: str = SOURCE_OF_TRUTH_CANONICAL_MEMORY
    freshness_ttl_hours: int = 24 * 30

    # --- Classification cues (used by ontology_classifier) -----------------
    text_patterns: tuple[str, ...] = ()
    """Regex patterns over name/title/summary/fact that classify text → this label."""

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

    # --- Edge cardinality --------------------------------------------------
    singleton: bool = False
    """When True, ``(subject, predicate)`` admits one live object at a time.

    The canonical writer auto-stamps ``invalid_at`` on any prior live claim
    with the same subject + predicate but a different object when a new
    deterministic claim lands. Multi-source corroboration on the same object
    is preserved (singleton applies to objects, not sources).
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


# --- Entity catalog ---------------------------------------------------------
# Conventions:
#   * ``scope=True``  → endpoint for future cross-cutting edges (APPLIES_TO etc.)
#   * ``project_map_family`` → drives the structural reader / project map
#   * ``fact_family``, ``source_of_truth``, ``freshness_ttl_hours`` → graph_quality
#   * ``text_patterns`` → drive the ontology_classifier


def _e(
    label: str,
    category: str,
    description: str,
    *,
    identity_class: IdentityClass,
    key_prefix: str,
    identity_policy: str | None = None,
    authoritative_source: str | None = None,
    **kwargs,
) -> EntityTypeSpec:
    """Helper for declaring an entity spec with concise call sites.

    ``identity_policy`` defaults to a human-readable sketch built from the
    key prefix and identity class so docs don't repeat the same string twice.
    """
    required = kwargs.pop("required", ())
    lifecycle = kwargs.pop("lifecycle", frozenset())
    if identity_policy is None:
        suffix = {
            IdentityClass.SLUG_ALIAS: "<slug>",
            IdentityClass.EXTERNAL_ID: "<id>",
            IdentityClass.CONTENT_HASH: "<hash>",
        }[identity_class]
        identity_policy = f"{key_prefix}:{suffix}"
    return EntityTypeSpec(
        label=label,
        category=category,
        description=description,
        identity_class=identity_class,
        key_prefix=key_prefix,
        identity_policy=identity_policy,
        authoritative_source=authoritative_source,
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
    # --- Topology scope endpoints ------------------------------------------
    "Repository": _e(
        "Repository",
        "topology",
        "A source code repository.",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="repo",
        identity_policy="repo:<host>/<org>/<name>",
        scope=True,
        project_map_family="repos",
        fact_family="topology",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=WEEK,
        text_patterns=(r"\brepo(sitory)?\b", r"\bmonorepo\b"),
    ),
    "Service": _e(
        "Service",
        "topology",
        "A deployable/runnable unit (service, worker, cronjob, frontend, gateway, library).",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="service",
        scope=True,
        project_map_family="services",
        fact_family="topology",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=WEEK,
        text_patterns=(
            r"\bservice\b",
            r"\bmicroservice\b",
            r"\bworker\b",
            r"\bcronjob\b",
        ),
    ),
    "Environment": _e(
        "Environment",
        "topology",
        "A named runtime target (prod, staging, dev, preview).",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="environment",
        scope=True,
        project_map_family="environments",
        fact_family="topology",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=WEEK,
        text_patterns=(
            r"\benv(ironment)?\b",
            r"\b(prod|production|staging|stage|dev|preview)\b",
        ),
    ),
    "DataStore": _e(
        "DataStore",
        "topology",
        "A stateful backing resource (postgres, mysql, redis, kafka, s3, ...).",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="datastore",
        scope=True,
        project_map_family="datastores",
        fact_family="topology",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=WEEK,
        text_patterns=(
            r"\b(database|datastore|postgres|mysql|redis|kafka|bucket|s3|elastic)\b",
        ),
    ),
    "Cluster": _e(
        "Cluster",
        "topology",
        "The cloud account / cluster / region an environment runs on.",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="cluster",
        scope=True,
        project_map_family="clusters",
        fact_family="topology",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=WEEK,
        text_patterns=(r"\b(cluster|region|eks|gke|ecs|kubernetes|k8s)\b",),
    ),
    # --- Code-anchored topology nodes --------------------------------------
    # ``Dependency`` is a package-manager dependency (``requests==2.31``);
    # ``APIContract`` is an OpenAPI operation (path + method). Harnesses may
    # assert these when grounded in explicit repository evidence.
    "Dependency": _e(
        "Dependency",
        "topology",
        "A third-party package / library a service depends on.",
        identity_class=IdentityClass.EXTERNAL_ID,
        key_prefix="dependency",
        identity_policy="dependency:<ecosystem>:<name>",
        fact_family="topology",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=WEEK,
        property_signatures=("package_name", "version"),
        text_patterns=(r"\bdependency\b", r"\bpackage\b", r"\blibrary\b"),
    ),
    "APIContract": _e(
        "APIContract",
        "topology",
        "One operation in an OpenAPI / RPC spec — a (path, method) pair a "
        "service exposes.",
        identity_class=IdentityClass.EXTERNAL_ID,
        key_prefix="api_contract",
        identity_policy="api_contract:<service>:<method>:<path>",
        fact_family="topology",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=WEEK,
        property_signatures=("http_method", "path"),
        text_patterns=(r"\bAPI\b", r"\bendpoint\b", r"\boperation\b"),
    ),
    # --- Product functionality ----------------------------------------------
    # ``Feature`` is the first-class answer to "what does this repo/service
    # do?". Harnesses assert features from authored evidence (README, docs,
    # route specs) via PROVIDES / IMPLEMENTED_IN claims; nothing infers them
    # from file trees.
    "Feature": _e(
        "Feature",
        "product",
        "A user-facing or system-facing capability a repository or service "
        "provides (e.g. checkout, SSO login, usage metering).",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="feature",
        project_map_family="features",
        fact_family="product",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=12 * WEEK,
        text_patterns=(r"\bfeature\b", r"\bcapabilit(y|ies)\b"),
    ),
    # --- People ------------------------------------------------------------
    "Team": _e(
        "Team",
        "people",
        "An owning group / squad.",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="team",
        project_map_family="teams",
        fact_family="ownership",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=WEEK,
        text_patterns=(r"\bteam\b", r"\bsquad\b"),
    ),
    "Person": _e(
        "Person",
        "people",
        "An individual contributor / owner.",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="person",
        identity_policy="person:<handle>",
        project_map_family="people",
        fact_family="ownership",
        source_of_truth=SOT_CODE,
        freshness_ttl_hours=WEEK,
        text_patterns=(r"\bperson\b", r"\bauthor\b", r"\bowner\b"),
    ),
    # --- Activity timeline -------------------------------------------------
    # A timestamped event. ``is_activity=True`` flags it as a timeline node;
    # its claims carry ``valid_at = occurred_at`` (event time), so the
    # timeline is reconstructed by the existing bitemporal window/as-of
    # filters rather than any dedicated temporal edge. The event ``kind`` is
    # the ``verb`` property (opened_pr / merged_pr / deployed / decided …),
    # which rides on the activity's claims and so comes back in
    # ``ClaimRow.properties`` for read-time include/exclude.
    "Activity": _e(
        "Activity",
        "timeline",
        "A timestamped event — something that happened (PR merged, deploy, "
        "alert, discussion, decision). The unit of the activity timeline.",
        identity_class=IdentityClass.EXTERNAL_ID,
        key_prefix="activity",
        identity_policy="activity:<source>:<id>",
        is_activity=True,
        fact_family="timeline",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=2 * WEEK,
        property_signatures=("occurred_at", "verb_class"),
        text_patterns=(
            r"\bmerged\b",
            r"\bdeploy(ed|ment)?\b",
            r"\brolled?\s*back\b",
            r"\balert(ed)?\b",
            r"\bincident\b",
        ),
    ),
    "Period": _e(
        "Period",
        "timeline",
        "A timeline bucket (daily by default) anchoring activities for "
        "windowed timeline queries.",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="period",
        identity_policy="timeline:period:daily:<pot>:<yyyy-mm-dd>",
        fact_family="timeline",
        source_of_truth=SOT_EXTERNAL,
        freshness_ttl_hours=2 * WEEK,
        property_signatures=("period_kind",),
    ),
    # --- Memory tier (preferences / bugs / decisions) ----------------------
    # Anchors for the agent-facing context_record flow. Each carries a
    # ``fact_family`` distinct from topology so freshness / source-of-truth
    # policies can diverge. All emit ``:RELATES_TO`` claims with category
    # ``memory`` (see EDGE_TYPES below).
    "Preference": _e(
        "Preference",
        "memory",
        "A coding preference / policy with scope-qualified prescription "
        "(language / framework / repo / service).",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="preference",
        fact_family="preference",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=12 * WEEK,
        text_patterns=(r"\bprefer(ence)?\b", r"\bpolicy\b", r"\bconvention\b"),
        property_signatures=("policy_kind", "prescription"),
    ),
    "Policy": _e(
        "Policy",
        "memory",
        "A named project-wide policy (typically a strong/hard preference).",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="policy",
        fact_family="preference",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=12 * WEEK,
        lifecycle=("proposed", "active", "deprecated"),
    ),
    "BugPattern": _e(
        "BugPattern",
        "memory",
        "A reproducible failure pattern — the symptom side of a fix.",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="bug_pattern",
        fact_family="bugs",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=12 * WEEK,
        text_patterns=(r"\bbug\b", r"\bfailure\b", r"\bregression\b"),
        property_signatures=("symptom_signature",),
    ),
    "Fix": _e(
        "Fix",
        "memory",
        "A bug-fix observation retrievable later by symptom. Carries "
        "``verification_status`` and links to its BugPattern via RESOLVED / "
        "ATTEMPTED_FIX_FAILED claims.",
        identity_class=IdentityClass.CONTENT_HASH,
        key_prefix="fix",
        fact_family="bugs",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=12 * WEEK,
        text_patterns=(r"\bfix(ed)?\b", r"\bworkaround\b"),
        property_signatures=("fix_steps", "verification_status"),
    ),
    "Decision": _e(
        "Decision",
        "memory",
        "An ADR-style architectural decision with title, rationale, "
        "alternatives_rejected.",
        identity_class=IdentityClass.CONTENT_HASH,
        key_prefix="decision",
        fact_family="decisions",
        source_of_truth=SOT_MEMORY,
        freshness_ttl_hours=24 * WEEK,
        lifecycle=("proposed", "accepted", "superseded", "deprecated", "rejected"),
        text_patterns=(r"\bdecision\b", r"\b(ADR|architecture decision)\b"),
        property_signatures=("rationale", "alternatives_rejected"),
    ),
    # --- Generic fail-open fallbacks (soft-fail downgrade targets) ----------
    # The agent reconciliation path coerces unrecognized output onto these
    # rather than rejecting the batch. ``public=False`` keeps them out of the
    # agent-facing topology contract; they carry no project-map family.
    "Document": _e(
        "Document",
        "evidence",
        "Generic document / note (soft-fail fallback for unrecognized doc-like entities).",
        identity_class=IdentityClass.CONTENT_HASH,
        key_prefix="document",
        public=False,
        fact_family="evidence",
        source_of_truth=SOT_MEMORY,
    ),
    "Observation": _e(
        "Observation",
        "evidence",
        "Generic observation / signal (soft-fail fallback for unrecognized entities).",
        identity_class=IdentityClass.CONTENT_HASH,
        key_prefix="observation",
        public=False,
        fact_family="evidence",
        source_of_truth=SOT_SOFT,
    ),
    "QualityIssue": _e(
        "QualityIssue",
        "hygiene",
        "Internal marker the soft-fail path attaches to record an ontology downgrade.",
        identity_class=IdentityClass.CONTENT_HASH,
        key_prefix="quality",
        public=False,
        fact_family="evidence",
        source_of_truth=SOT_SOFT,
    ),
}


# --- Edge catalog -----------------------------------------------------------
# Direction is chosen so the common query is a single forward or backward hop.
# Cardinality drives supersession: ``singleton=True`` means a new object
# supersedes the old; everything else accumulates.


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
    "DEFINED_IN": _x(
        "DEFINED_IN",
        "A service's code lives in a repository (edge prop ``path`` carries the subtree).",
        [("Service", "Repository")],
        category="topology",
        source_inferred=("Service",),
        target_inferred=("Repository",),
    ),
    "DEPLOYED_TO": _x(
        "DEPLOYED_TO",
        "A service runs in an environment.",
        [("Service", "Environment")],
        category="topology",
        predicate_family="deployment_target",
        source_inferred=("Service",),
        target_inferred=("Environment",),
    ),
    "DEPENDS_ON": _x(
        "DEPENDS_ON",
        "A service depends on / calls another service.",
        [("Service", "Service")],
        category="topology",
        source_inferred=("Service",),
        target_inferred=("Service",),
    ),
    "USES": _x(
        "USES",
        "A service uses a datastore or third-party dependency.",
        [("Service", "DataStore"), ("Service", "Dependency")],
        category="topology",
        predicate_family="datastore_binding",
        source_inferred=("Service",),
    ),
    "EXPOSES": _x(
        "EXPOSES",
        "A service exposes an API operation (path + method) defined by an "
        "OpenAPI / RPC contract.",
        [("Service", "APIContract")],
        category="topology",
        source_inferred=("Service",),
        target_inferred=("APIContract",),
    ),
    "HOSTED_ON": _x(
        "HOSTED_ON",
        "An environment runs on a cluster / platform.",
        [("Environment", "Cluster")],
        category="topology",
        predicate_family="deployment_target",
        source_inferred=("Environment",),
        target_inferred=("Cluster",),
    ),
    "PROVIDES": _x(
        "PROVIDES",
        "A repository or service provides a feature / capability. The spine "
        "of 'what does this repo do'.",
        [("Repository", "Feature"), ("Service", "Feature")],
        category="topology",
        target_inferred=("Feature",),
    ),
    "IMPLEMENTED_IN": _x(
        "IMPLEMENTED_IN",
        "A feature's implementation lives in a repository, service, or code "
        "asset. The navigation backlink from capability to code.",
        [("Feature", "Repository"), ("Feature", "Service"), ("Feature", "CodeAsset")],
        category="topology",
        source_inferred=("Feature",),
    ),
    "OWNED_BY": _x(
        "OWNED_BY",
        "A service or repo is owned by a team or person (one live owner at a time).",
        [
            ("Service", "Team"),
            ("Service", "Person"),
            ("Repository", "Team"),
            ("Repository", "Person"),
        ],
        category="ownership",
        predicate_family="owner_binding",
        singleton=True,
    ),
    "MEMBER_OF": _x(
        "MEMBER_OF",
        "A person is a member of a team.",
        [("Person", "Team")],
        category="people",
        source_inferred=("Person",),
        target_inferred=("Team",),
    ),
    # --- Activity timeline (events) ----------------------------------------
    # Three uniform claims hang off every Activity. There are deliberately NO
    # stored temporal/causal edges (TRIGGERED_BY / PRECEDED_BY / HOTSPOT): the
    # timeline is a read-time query over ``valid_at`` (ordering, windowing,
    # proximity correlation, churn counts), not graph topology. Each edge is
    # source-inferred to ``Activity`` so any event endpoint auto-classifies.
    "TOUCHED": _x(
        "TOUCHED",
        "An activity changed / affected a scope (service, datastore, "
        "environment, repo) or code asset. The spine of 'what changed in X'.",
        [(ACTIVITY_ENDPOINT, SCOPE_ENDPOINT), (ACTIVITY_ENDPOINT, "CodeAsset")],
        category="timeline",
        source_inferred=("Activity",),
    ),
    "PERFORMED": _x(
        "PERFORMED",
        "A person or team performed an activity. Direction: actor → activity, "
        "matching ``timeline_plan.build_timeline_mutations``.",
        [("Person", ACTIVITY_ENDPOINT), ("Team", ACTIVITY_ENDPOINT)],
        category="timeline",
        target_inferred=("Activity",),
    ),
    "AUTHORED": _x(
        "AUTHORED",
        "A person/team authored content referenced by an activity (commit, "
        "PR body, comment). Looser than PERFORMED — captures originating "
        "authorship that may differ from the activity actor.",
        [("Person", ACTIVITY_ENDPOINT), ("Team", ACTIVITY_ENDPOINT)],
        category="timeline",
        target_inferred=("Activity",),
    ),
    "IN_PERIOD": _x(
        "IN_PERIOD",
        "An activity falls within a timeline period bucket (daily by default).",
        [(ACTIVITY_ENDPOINT, "Period")],
        category="timeline",
        source_inferred=("Activity",),
        target_inferred=("Period",),
    ),
    "MENTIONS": _x(
        "MENTIONS",
        "An activity's body referenced an entity (provenance / recall; looser "
        "than TOUCHED — the F4 link so timeline recall doesn't fall through).",
        [(ACTIVITY_ENDPOINT, WILDCARD_ENDPOINT)],
        category="timeline",
        source_inferred=("Activity",),
    ),
    # --- Memory tier predicates --------------------------------------------
    # Emitted by the context_record path (via LLM extraction or, in the
    # P6 deterministic emitter, directly). Each is anchored by one of the
    # memory-tier entity types (Preference / Policy / BugPattern / Fix /
    # Decision) and attaches to a topology scope or another memory anchor.
    "POLICY_APPLIES_TO": _x(
        "POLICY_APPLIES_TO",
        "A preference / policy applies to a scope or code asset. Backs the "
        "coding_preferences reader.",
        [
            ("Preference", SCOPE_ENDPOINT),
            ("Preference", "CodeAsset"),
            ("Policy", SCOPE_ENDPOINT),
            ("Policy", "CodeAsset"),
        ],
        category="memory",
        predicate_family="policy_binding",
        source_inferred=("Preference",),
    ),
    "REPRODUCES": _x(
        "REPRODUCES",
        "A bug pattern reproduces in a given scope (service / environment / "
        "code asset). Backs the prior_bugs reader's symptom-to-scope lookup.",
        [
            ("BugPattern", SCOPE_ENDPOINT),
            ("BugPattern", "CodeAsset"),
        ],
        category="memory",
        source_inferred=("BugPattern",),
    ),
    "RESOLVED": _x(
        "RESOLVED",
        "A fix resolved a bug pattern (verified or unverified — see Fix.verification_status).",
        [("Fix", "BugPattern")],
        category="memory",
        predicate_family="fix_outcome",
        source_inferred=("Fix",),
        target_inferred=("BugPattern",),
    ),
    "ATTEMPTED_FIX_FAILED": _x(
        "ATTEMPTED_FIX_FAILED",
        "A fix was attempted against a bug pattern but did not resolve it. "
        "Surfaced by prior_bugs reader so agents avoid repeating failed attempts.",
        [("Fix", "BugPattern")],
        category="memory",
        predicate_family="fix_outcome",
        source_inferred=("Fix",),
        target_inferred=("BugPattern",),
    ),
    "VERIFIED": _x(
        "VERIFIED",
        "An activity (or actor) independently confirmed a fix worked. "
        "Folded into prior_bugs ranking as verification corroboration.",
        [(ACTIVITY_ENDPOINT, "Fix"), ("Person", "Fix"), ("Team", "Fix")],
        category="memory",
        target_inferred=("Fix",),
    ),
    "DECIDED": _x(
        "DECIDED",
        "A decision was made on a scope (service / repo / environment).",
        [("Decision", SCOPE_ENDPOINT)],
        category="memory",
        source_inferred=("Decision",),
    ),
    "AFFECTS": _x(
        "AFFECTS",
        "A decision affects a scope, code asset, or downstream entity. "
        "Looser than DECIDED — captures downstream impact, not the choice itself.",
        [
            ("Decision", SCOPE_ENDPOINT),
            ("Decision", "CodeAsset"),
            ("Decision", WILDCARD_ENDPOINT),
        ],
        category="memory",
        source_inferred=("Decision",),
    ),
    # --- Generic fail-open fallback (soft-fail downgrade target) ------------
    # Unrecognized agent-emitted edge types coerce onto this wildcard edge
    # rather than failing the batch. ``public=False`` keeps it out of the
    # advertised topology vocabulary.
    "RELATED_TO": _x(
        "RELATED_TO",
        "Generic association (soft-fail fallback for unrecognized edge types).",
        [(WILDCARD_ENDPOINT, WILDCARD_ENDPOINT)],
        category="generic",
        public=False,
    ),
}


# --- Derived label sets -----------------------------------------------------


CANONICAL_LABELS: frozenset[str] = frozenset(ENTITY_TYPES.keys())
CANONICAL_EDGE_TYPES: frozenset[str] = frozenset(EDGE_TYPES.keys())
# Writer-internal bookkeeping edges. These are NOT part of the agent-facing
# vocabulary (agents must never emit them) and so are deliberately
# kept out of ``EDGE_TYPES`` / ``CANONICAL_EDGE_TYPES``. The canonical writer's
# supersession machinery emits ``SUPERSEDES`` directly (see
# ``canonical_writer._write_supersedes_claim``); declaring it here keeps the
# ontology honest about every relationship name that can exist in the store.
SYSTEM_EDGE_TYPES: frozenset[str] = frozenset({"SUPERSEDES"})
# Every relationship name the store can legitimately hold: agent-facing
# canonical edges plus writer-internal system edges.
ALL_EDGE_TYPES: frozenset[str] = CANONICAL_EDGE_TYPES | SYSTEM_EDGE_TYPES
SCOPE_LABELS: frozenset[str] = frozenset(
    label for label, spec in ENTITY_TYPES.items() if spec.scope
)
ACTIVITY_LABELS: frozenset[str] = frozenset(
    label for label, spec in ENTITY_TYPES.items() if spec.is_activity
) | {"Activity"}
# The ontology is the single source of truth for which predicates are
# singleton; the registry rebuilds itself from this set so any new
# ``singleton=True`` flag on an edge spec auto-propagates to the canonical
# writer's supersession path.
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
ENTITY_PROJECT_MAP_FAMILY: dict[str, str] = _build_entity_family_map(
    "project_map_family"
)
ENTITY_DEBUGGING_FAMILY: dict[str, str] = _build_entity_family_map("debugging_family")
ENTITY_FACT_FAMILY: dict[str, str] = _build_entity_family_map("fact_family")


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


ENTITY_TEXT_CLASSIFIERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    _build_text_classifiers()
)


def _build_property_signatures() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for label, spec in ENTITY_TYPES.items():
        for prop in spec.property_signatures:
            out.setdefault(prop, []).append(label)
    return {prop: tuple(sorted(labels)) for prop, labels in out.items()}


ENTITY_PROPERTY_SIGNATURES: dict[str, tuple[str, ...]] = _build_property_signatures()


# Non-canonical verb endpoint inference. Empty in the topology core; the claim
# layer added later can register rules for verbs that aren't canonical edges.
EPISODIC_VERB_INFERRED_LABELS: dict[tuple[str, str], tuple[str, ...]] = {}


# Edge endpoint inference table — built from spec source/target_inferred PLUS
# any non-canonical verb rules above.
def _build_edge_endpoint_inferred_labels() -> dict[tuple[str, str], tuple[str, ...]]:
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for edge_type, spec in EDGE_TYPES.items():
        if spec.source_inferred_labels:
            out[(edge_type, "source")] = spec.source_inferred_labels
        if spec.target_inferred_labels:
            out[(edge_type, "target")] = spec.target_inferred_labels
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
    return {fam: frozenset(names) for fam, names in out.items()}


PREDICATE_FAMILY_EDGE_NAMES: dict[str, frozenset[str]] = (
    _build_predicate_family_edge_names()
)


def _union_entity_lifecycle_strings() -> frozenset[str]:
    out: set[str] = set()
    for spec in ENTITY_TYPES.values():
        out.update(spec.lifecycle_states)
    out.update(member.value for member in LifecycleStatus)
    return frozenset(out)


ALLOWED_LIFECYCLE_STATUSES: frozenset[str] = _union_entity_lifecycle_strings()


# --- Helpers ----------------------------------------------------------------


def normalize_edge_name(name: str) -> str:
    """Normalize an LLM-emitted relation label for family lookup."""
    return name.strip().upper().replace(" ", "_").replace("-", "_")


# Back-compat alias — external callers may still import the legacy name.


def inferred_labels_for_episodic_edge_endpoint(
    edge_name: str, role: str
) -> tuple[str, ...]:
    """Canonical labels to add for a RELATES_TO endpoint, if unambiguous."""
    if role not in ("source", "target"):
        return ()
    key = (normalize_edge_name(edge_name), role)
    return EDGE_ENDPOINT_INFERRED_LABELS.get(key, ())


def predicate_family_for_edge_name(name: str) -> str | None:
    """Return predicate family id for a ``RELATES_TO.name``, if any."""
    n = normalize_edge_name(name)
    for fam, members in PREDICATE_FAMILY_EDGE_NAMES.items():
        if n in members:
            return fam
    return None


def predicate_family_for_episodic_supersede(
    edge_name: str,
    target_labels: Iterable[str] | None = None,
) -> str | None:
    """Predicate family for temporal auto-supersede / pairwise conflict bucketing.

    ``target_labels`` is accepted for back-compat with the legacy two-arg
    callsite but is unused — family is derived from the edge name alone now
    that all canonical predicates have known endpoint shapes.
    """
    del target_labels
    return predicate_family_for_edge_name(edge_name)


def object_counterparty_uuid_for_edge(
    edge_name: str,
    source_uuid: str,
    target_uuid: str,
    *,
    predicate_family: str | None = None,
) -> str | None:
    """Endpoint uuid that differs when the same resource has conflicting bindings.

    For every declared canonical predicate, the subject is the cardinality
    anchor and the object is the counterparty that supersedes — so we
    return the target. Legacy LLM-emitted aliases (e.g. ``OWNS``) that
    inverted this convention are no longer in the catalog; if they
    resurface, declare them explicitly in :data:`EDGE_TYPES` rather than
    teaching this helper about them.
    """
    del source_uuid
    fam = (
        predicate_family
        if predicate_family is not None
        else predicate_family_for_edge_name(edge_name)
    )
    if fam is None or fam not in PREDICATE_FAMILY_EDGE_NAMES:
        return None
    return target_uuid


def temporal_subject_key_for_edge(
    edge_name: str,
    source_uuid: str,
    target_uuid: str,
    *,
    predicate_family: str | None = None,
) -> str | None:
    """Stable subject node uuid for contradiction grouping within a family.

    Mirrors :func:`object_counterparty_uuid_for_edge`: for every declared
    canonical predicate the subject is the cardinality anchor, so we
    return ``source_uuid``. Legacy inverted aliases (``OWNS``, ``MAINTAINED_BY``)
    no longer exist in the catalog; declare new edges explicitly rather
    than special-casing names here.
    """
    del target_uuid
    fam = (
        predicate_family
        if predicate_family is not None
        else predicate_family_for_edge_name(edge_name)
    )
    if fam is None or fam not in PREDICATE_FAMILY_EDGE_NAMES:
        return None
    return source_uuid


def canonical_entity_labels(labels: Iterable[str]) -> tuple[str, ...]:
    """Return public canonical labels from a mixed label set."""
    return tuple(label for label in labels if label in ENTITY_TYPES)


def is_canonical_entity_label(label: str) -> bool:
    return label in ENTITY_TYPES


def is_canonical_edge_type(edge_type: str) -> bool:
    return edge_type in EDGE_TYPES


def entity_spec(label: str) -> EntityTypeSpec | None:
    return ENTITY_TYPES.get(label)


def edge_spec(edge_type: str) -> EdgeTypeSpec | None:
    return EDGE_TYPES.get(edge_type)


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


# ---------------------------------------------------------------------------
# Identity registry sync — populate :mod:`domain.identity` from ENTITY_TYPES
# ---------------------------------------------------------------------------
# The identity registry is a *view* over the ontology: every public entity
# type registers its (label, identity_class, key_prefix) at module import so
# the registry and the ontology can never disagree. Adding ``IdentityClass``
# / ``key_prefix`` to a new ``EntityTypeSpec`` row is the only step needed —
# the registry follows automatically.


def _sync_identity_registry() -> None:
    for label, spec in ENTITY_TYPES.items():
        register_identity(
            IdentitySpec(
                label=label,
                klass=spec.identity_class,
                key_prefix=spec.key_prefix,
                authoritative_source=spec.authoritative_source,
            )
        )


_sync_identity_registry()


# ---------------------------------------------------------------------------
# Agent-facing record-type catalog
# ---------------------------------------------------------------------------
# One declarative table joining the ``context_record`` agent surface to the
# graph ontology. Each row says: when an agent submits ``record_type=t``,
# which canonical entity it anchors, which predicate that entity emits, what
# structured payload schema validates it, and which read-side include family
# the records flow back through.
#
# This catalog is the single source of truth for:
#   * agent_context_port.CONTEXT_RECORD_TYPES (derived from .keys() of public rows)
#   * agent_context_port.READER_BACKED_INCLUDES + PLANNED_INCLUDES
#     (derived by joining ``reader_include`` against the registered reader set)
#   * context_records dispatch (``payload_schema`` names the builder)


@dataclass(frozen=True, slots=True)
class RecordTypeSpec:
    """Declarative metadata for one agent-facing ``context_record`` type."""

    record_type: str
    description: str
    anchor_label: str
    """Which entity type the record creates / attaches to. Must be in ENTITY_TYPES."""

    emits_predicate: str | None
    """Primary predicate the record emits on its anchor. Must be in EDGE_TYPES
    (or None for free-form records that don't yet have a deterministic path)."""

    payload_schema: str | None
    """Name of the structured builder in ``domain.context_records``
    (``fix``/``preference``/``bug_pattern``/``decision``/``verification``).
    ``None`` means the record falls through to ``FreeFormRecord``."""

    reader_include: str | None
    """Which read-side include family these records surface through. None for
    records that don't have a primary reader yet."""

    public: bool = True
    """``False`` hides the record type from the advertised agent vocabulary."""


# NOTE: ``emits_predicate`` records the *primary* canonical predicate. Some
# record types emit secondary predicates as well (e.g. a Fix may emit
# ATTEMPTED_FIX_FAILED on retries); the writer is responsible for these
# secondaries. The catalog's role is the surface vocabulary, not the full
# predicate graph each type produces.


RECORD_TYPES: dict[str, RecordTypeSpec] = {
    "preference": RecordTypeSpec(
        record_type="preference",
        description="Coding preference / policy with scope-qualified prescription.",
        anchor_label="Preference",
        emits_predicate="POLICY_APPLIES_TO",
        payload_schema="preference",
        reader_include="coding_preferences",
    ),
    "policy": RecordTypeSpec(
        record_type="policy",
        description="Named project-wide policy (alias of preference with Policy anchor).",
        anchor_label="Policy",
        emits_predicate="POLICY_APPLIES_TO",
        payload_schema="preference",  # same shape; different anchor label
        reader_include="coding_preferences",
    ),
    "bug_pattern": RecordTypeSpec(
        record_type="bug_pattern",
        description="Reproducible failure pattern — the symptom side of a fix.",
        anchor_label="BugPattern",
        emits_predicate="REPRODUCES",
        payload_schema="bug_pattern",
        reader_include="prior_bugs",
    ),
    "fix": RecordTypeSpec(
        record_type="fix",
        description="Bug-fix observation retrievable later by symptom.",
        anchor_label="Fix",
        emits_predicate="RESOLVED",
        payload_schema="fix",
        reader_include="prior_bugs",
    ),
    "verification": RecordTypeSpec(
        record_type="verification",
        description="Confirm / refute attached to an existing Fix.",
        anchor_label="Activity",
        emits_predicate="VERIFIED",
        payload_schema="verification",
        reader_include="prior_bugs",
    ),
    "decision": RecordTypeSpec(
        record_type="decision",
        description="ADR-style architectural decision with rationale + alternatives.",
        anchor_label="Decision",
        emits_predicate="DECIDED",
        payload_schema="decision",
        reader_include="decisions",  # planned reader
    ),
    # --- Free-form record types (no structured schema yet, FreeFormRecord) -
    # Kept as advertised types so the agent surface accepts them; readers
    # surface them as ``unsupported_include`` until each gets a backing
    # entity + reader pair. The advertised contract stays honest because
    # the coherence check (domain.coherence) flags any free-form type whose
    # reader_include has neither a reader nor a planned-reader marker.
    "investigation": RecordTypeSpec(
        record_type="investigation",
        description="Ongoing or completed investigation context.",
        anchor_label="Document",
        emits_predicate=None,
        payload_schema=None,
        reader_include=None,
    ),
    "diagnostic_signal": RecordTypeSpec(
        record_type="diagnostic_signal",
        description="A diagnostic signal / observation captured during debugging.",
        anchor_label="Observation",
        emits_predicate=None,
        payload_schema=None,
        reader_include=None,
    ),
    "workflow": RecordTypeSpec(
        record_type="workflow",
        description="A repeatable workflow / runbook entry.",
        anchor_label="Document",
        emits_predicate=None,
        payload_schema=None,
        reader_include=None,
    ),
    "feature_note": RecordTypeSpec(
        record_type="feature_note",
        description="A note about a feature, its rationale, or its current state.",
        anchor_label="Document",
        emits_predicate=None,
        payload_schema=None,
        reader_include=None,
    ),
    "service_note": RecordTypeSpec(
        record_type="service_note",
        description="A note attached to a service — operational quirks, gotchas.",
        anchor_label="Document",
        emits_predicate=None,
        payload_schema=None,
        reader_include=None,
    ),
    "runbook_note": RecordTypeSpec(
        record_type="runbook_note",
        description="A runbook fragment — what to do when X happens.",
        anchor_label="Document",
        emits_predicate=None,
        payload_schema=None,
        reader_include=None,
    ),
    "integration_note": RecordTypeSpec(
        record_type="integration_note",
        description="Notes on integrating two systems / services / APIs.",
        anchor_label="Document",
        emits_predicate=None,
        payload_schema=None,
        reader_include=None,
    ),
    "incident_summary": RecordTypeSpec(
        record_type="incident_summary",
        description="Post-incident summary (root cause, contributing factors, follow-ups).",
        anchor_label="Document",
        emits_predicate=None,
        payload_schema=None,
        reader_include=None,
    ),
    "doc_reference": RecordTypeSpec(
        record_type="doc_reference",
        description="Pointer to canonical project documentation.",
        anchor_label="Document",
        emits_predicate=None,
        payload_schema=None,
        reader_include="docs",  # planned reader
    ),
}


PUBLIC_RECORD_TYPES: frozenset[str] = frozenset(
    rt for rt, spec in RECORD_TYPES.items() if spec.public
)


# Reader include families that aren't backed by RECORD_TYPES rows because
# they read pure topology data (no agent-emitted records anchor them). These
# are listed explicitly so the coherence check can distinguish "structural
# include" from "missing reader_include on a record type".
STRUCTURAL_INCLUDES: frozenset[str] = frozenset(
    {
        "infra_topology",
        "timeline",
        "owners",
        "raw_graph",
    }
)


def record_type_spec(record_type: str) -> RecordTypeSpec | None:
    return RECORD_TYPES.get(record_type)


def record_types_for_include(include_key: str) -> tuple[str, ...]:
    """Reverse lookup: which record types surface through ``include_key``."""
    return tuple(
        rt for rt, spec in RECORD_TYPES.items() if spec.reader_include == include_key
    )


def advertised_include_families() -> frozenset[str]:
    """Every include family the agent surface advertises — structural + memory."""
    from_records = frozenset(
        spec.reader_include
        for spec in RECORD_TYPES.values()
        if spec.public and spec.reader_include
    )
    return STRUCTURAL_INCLUDES | from_records
