"""Typed entity/edge schemas used for Graphiti extraction.

Edge keys use SCREAMING_SNAKE_CASE so ``relation_type`` matches Neo4j ``RELATES_TO.name``
and the governed ontology (``domain/ontology.py``).

See docs/context-graph-improvements/02-edge-type-collapse.md.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# --- Entity models (unchanged surface for Graphiti) ---------------------------------


class PullRequest(BaseModel):
    canonical_type: Optional[str] = Field(
        default=None,
        description="Optional governed ontology label when type should be stamped (e.g. PullRequest).",
    )
    pr_number: Optional[int] = None
    title: Optional[str] = None
    why_summary: Optional[str] = None
    change_type: Optional[str] = None
    feature_area: Optional[str] = None
    author: Optional[str] = None
    merged_at: Optional[str] = None
    files_changed: Optional[int] = None


class Commit(BaseModel):
    canonical_type: Optional[str] = Field(
        default=None,
        description="Optional governed ontology label when type should be stamped (e.g. Commit).",
    )
    sha: Optional[str] = None
    message: Optional[str] = None
    author: Optional[str] = None
    branch: Optional[str] = None


class Issue(BaseModel):
    canonical_type: Optional[str] = Field(
        default=None,
        description="Optional governed ontology label when type should be stamped (e.g. Issue).",
    )
    issue_number: Optional[int] = None
    title: Optional[str] = None
    problem_statement: Optional[str] = None


class Feature(BaseModel):
    canonical_type: Optional[str] = Field(
        default=None,
        description="Optional governed ontology label when type should be stamped (e.g. Feature).",
    )
    feature_name: Optional[str] = None
    description: Optional[str] = None


class Decision(BaseModel):
    canonical_type: Optional[str] = Field(
        default=None,
        description="Optional finer ontology label (usually Decision); must match vocabulary.",
    )
    decision_made: Optional[str] = None
    alternatives_rejected: Optional[str] = None
    rationale: Optional[str] = None


class Developer(BaseModel):
    canonical_type: Optional[str] = Field(
        default=None,
        description="Optional governed ontology label when type should be stamped (e.g. Person).",
    )
    github_login: Optional[str] = None
    display_name: Optional[str] = None
    expertise_areas: Optional[str] = None


ENTITY_TYPES = {
    "PullRequest": PullRequest,
    "Commit": Commit,
    "Issue": Issue,
    "Feature": Feature,
    "Decision": Decision,
    "Developer": Developer,
}


# --- Edge models (FACT_TYPES for extract_edges) --------------------------------------


class Modified(BaseModel):
    """Narrow Git sense: this PR changed the given file or code entity."""

    file_path: Optional[str] = None


class MadeIn(BaseModel):
    """Commit belongs to PR, or change was introduced in a commit/PR context."""

    confidence: Optional[float] = None


class Fixes(BaseModel):
    """PR or change fixes a ticket, bug, or incident."""

    confidence: Optional[float] = None


class PartOfFeature(BaseModel):
    """PR implements or touches a feature area."""

    confidence: Optional[float] = None


class AuthoredBy(BaseModel):
    """Developer authored a commit or drove a PR."""

    confidence: Optional[float] = None


class Owns(BaseModel):
    """Person or team owns a feature, service, or component."""

    confidence: Optional[float] = None


class MigratedTo(BaseModel):
    """System, service, or datastore was migrated to another (e.g. Mongo → Postgres)."""


class Deprecated(BaseModel):
    """API, component, library, or path is deprecated or slated for removal."""


class Decommissioned(BaseModel):
    """Cluster, environment, datastore, or resource was shut down or removed."""


class Planned(BaseModel):
    """Future work: will be added, scheduled, roadmap, not yet done."""


class Delivered(BaseModel):
    """Past work: shipped, merged, rolled out, already migrated."""


class GenericAction(BaseModel):
    """Last resort when no other FACT_TYPE fits; pair with a precise ``fact`` sentence."""


class Replaces(BaseModel):
    """New component, API, or system replaces an older one."""


class Caused(BaseModel):
    """Causal relationship (outage caused by X, regression caused by Y)."""


class AddedTo(BaseModel):
    """Telemetry, dependency, feature, or instrumentation was added to a scope."""


class RemovedFrom(BaseModel):
    """Something was removed from dependencies, stack, or codebase."""


class StoredIn(BaseModel):
    """Data is stored in or primarily associated with a database or store."""


class DecidesFor(BaseModel):
    """Decision, ADR, or policy governs a component or scope."""


class DependsOn(BaseModel):
    """Runtime or build dependency between services, libs, or components."""


class DeployedTo(BaseModel):
    """Service, job, or artifact runs in an environment or region."""


EDGE_TYPES: dict[str, type[BaseModel]] = {
    # Git / review (legacy shapes; keep names stable for prompts)
    "MODIFIED": Modified,
    "MADE_IN": MadeIn,
    "FIXES": Fixes,
    "PART_OF_FEATURE": PartOfFeature,
    "AUTHORED_BY": AuthoredBy,
    "OWNS": Owns,
    # Ontology-aligned verbs (reduce MODIFIED collapse)
    "MIGRATED_TO": MigratedTo,
    "DEPRECATED": Deprecated,
    "DECOMMISSIONED": Decommissioned,
    "PLANNED": Planned,
    "DELIVERED": Delivered,
    "GENERIC_ACTION": GenericAction,
    "REPLACES": Replaces,
    "CAUSED": Caused,
    "ADDED_TO": AddedTo,
    "REMOVED_FROM": RemovedFrom,
    "STORED_IN": StoredIn,
    "DECIDES_FOR": DecidesFor,
    "DEPENDS_ON": DependsOn,
    "DEPLOYED_TO": DeployedTo,
}

# Graphiti labels every entity node with ``Entity`` plus extracted types; allow all
# registered edge types for any entity–entity pair so FACT_TYPES fully constrains
# the LLM (see graphiti extract_edges + default edge_type_map).
EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Entity", "Entity"): sorted(EDGE_TYPES.keys()),
}

def normalized_episodic_edge_allowlist() -> frozenset[str]:
    """Uppercase normalized names allowed for episodic ``RELATES_TO.name``."""
    from domain.extraction_edges import normalize_relation_name

    return frozenset(normalize_relation_name(k) for k in EDGE_TYPES)


GRAPHITI_CUSTOM_EXTRACTION_INSTRUCTIONS = """
## Potpie extraction rules (governed)

1. Set ``relation_type`` to **exactly** one of the FACT_TYPES ``fact_type_name`` values.
2. **Never** use MODIFIED unless the source is a PullRequest and the target is a code file
   or code entity **and** the fact is about editing specific files/lines. For migrations,
   deprecations, shutdowns, observability rollouts, or roadmap items, pick a specific verb
   (MIGRATED_TO, DEPRECATED, DECOMMISSIONED, PLANNED, DELIVERED, ADDED_TO, …).
3. If nothing fits, use GENERIC_ACTION and write a precise ``fact`` sentence (no relation_type
   spam like MODIFIED).
4. Prefer tense that matches reality: future/modal → PLANNED; past shipped → DELIVERED or
   MIGRATED_TO; scheduled removal → DEPRECATED or DECOMMISSIONED.

## canonical_type is how you pin a node to the governed vocabulary

Set ``canonical_type`` on **every** entity you create. Pick the single label
that best matches what the node represents. If the node is generic project
context you cannot otherwise classify, leave ``canonical_type`` unset and the
downstream ontology classifier will infer it from edge structure, text cues,
and properties.

Prefer these labels when the signals below are present:

- ``Decision`` — the text says "we decided", "we chose", "adopted", "selected
  X over Y", "ADR", "architecture decision", "design decision". **Do not
  label an architectural choice as ``Feature`` just because it touches
  features** — the node is the decision itself, not the feature.
- ``Fix`` — hotfix, patch for a specific bug, workaround, mitigation.
- ``Incident`` — outage, downtime, postmortem, SEV/P0-P4 event.
- ``Alert`` — alerting rule, pager rule, threshold alert.
- ``Runbook`` — runbook, playbook, on-call procedure.
- ``BugPattern`` — recurring failure mode, flaky-test pattern, known bad
  pattern.
- ``Constraint`` — hard constraint, compliance requirement, "must not" rule.
- ``Preference`` — team preference or style choice.
- ``AgentInstruction`` — AGENTS.md section, skill definition, agent guidance.
- ``Service`` / ``Component`` — named microservice or code module with
  operational significance.
- ``Feature`` / ``Capability`` — a user-facing deliverable area. **Do not
  fall back to ``Feature`` for decisions, fixes, incidents, constraints, or
  preferences** — those have their own labels above.

Rule of thumb: if the entity could plausibly be a Decision, Fix, Incident, or
Constraint, prefer that over ``Feature``. The Feature label is for concrete
product deliverables, not for abstract engineering work.
""".strip()
