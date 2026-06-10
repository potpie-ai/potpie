# Graph Workbench And Ontology

Last reviewed: 2026-06-05.

This document is the detailed Graph V2 contract. `graphv2.md` explains the
architecture and direction; this file defines the workbench surface, seed
ontology, read views, semantic mutations, and the rules for evolving ontology
over time.

The design target is a local-first workbench used by agent harnesses through the
CLI. The harness supplies reasoning. Potpie supplies stable contracts,
validation, storage, and audit.

## Design Rules

- The graph product surface is `potpie graph ...`.
- Agents read through named views, not arbitrary traversal.
- Agents write through semantic mutation plans, not raw CRUD.
- Every durable fact is pot-scoped, sourced, versioned, and inspectable.
- Unknown concepts are allowed, but they start as `agent_claim` or inbox work
  until an ontology contract exists.
- The ontology is a versioned product contract. It can evolve, but changes must
  be explicit, testable, and visible in `catalog` / `describe`.

## Workbench Surface

| Command | Purpose | Writes |
|---|---|---:|
| `potpie graph status` | Report readiness, versions, supported subgraphs/views/mutations, freshness, and skill drift. | No |
| `potpie graph catalog` | Rank subgraphs/views for a task and return compact discovery guidance. | No |
| `potpie graph describe` | Return a subgraph or view contract, examples, identity rules, authority rules, and mutation policy. | No |
| `potpie graph search-entities` | Resolve names, aliases, external IDs, and duplicate candidates before writes. | No |
| `potpie graph read` | Execute a bounded named read view with scope, query, ranking, provenance, and version metadata. | No |
| `potpie graph propose` | Validate a semantic mutation payload, classify risk, compute diff, persist a plan. | Plan only |
| `potpie graph commit` | Atomically apply an unexpired server-created plan by `plan_id`. | Yes |
| `potpie graph history` | Inspect entity, claim, plan, or subgraph history. | No |
| `potpie graph inbox` | Capture, claim, list, and close pending graph work. | Inbox only |
| `potpie graph quality` | List quality findings and suggested repair operations. | No |
| `potpie graph repair/export/import/reset/admin` | Operator maintenance, snapshots, projection repair, destructive actions. | Admin only |

All commands support `--json`. Human output is a view over the JSON contract,
not a separate behavior.

## JSON Envelope

All workbench commands should use the same top-level envelope:

```json
{
  "ok": true,
  "command": "graph.read",
  "request_id": "req:01JY...",
  "pot_id": "local/default",
  "graph_contract_version": "v2",
  "ontology_version": "2026-06-graph-v2",
  "subgraph_versions": {
    "features": 12,
    "decisions": 4
  },
  "result": {},
  "warnings": [],
  "unsupported": [],
  "recommended_next_action": null
}
```

Errors use the same envelope with `ok=false`:

```json
{
  "ok": false,
  "command": "graph.propose",
  "error": {
    "code": "ontology_validation_failed",
    "message": "Operation link_entities uses relation AFFECTED_BY with invalid endpoints.",
    "detail": {
      "operation_index": 2,
      "expected": "Feature -> BugPattern"
    }
  },
  "recommended_next_action": "Run `potpie graph describe features --view feature_context --json` and rebuild the proposal."
}
```

## Status Contract

`graph status` is the first cheap call. It must be sectioned by owner.

| Section | Required Fields |
|---|---|
| `host` | host kind, liveness, version, IPC/API status, auth transport, logs path when local |
| `pot` | active pot id/name/origin, source registry count, source freshness summary, selected backend |
| `graph_service` | graph contract version, ontology version, supported subgraphs, supported commands, validator readiness |
| `backend` | profile, canonical store health, semantic index health, traversal projection health, snapshot support |
| `ledger` | binding kind, source list status, consumer cursor lag, retry/dead-letter backlog |
| `skills` | installed skill ids, recommended skill ids, drift, exact install/update command |
| `quality` | degraded projections, stale source families, duplicate/conflict counts |

Status should not do expensive graph reads. It can report counts and readiness
from cached health/projection state.

## Discovery Loop

Agents should follow this loop for non-trivial work:

1. `potpie graph status --json`
2. `potpie graph catalog --task "<task>" --json`
3. `potpie graph describe <subgraph> --view <view> --examples --json`
4. `potpie graph search-entities ... --json` when a write may create or link an entity
5. `potpie graph read --subgraph <name> --view <view> --scope ... --json`
6. `potpie graph propose --file mutation.json --json`
7. Inspect `status`, `risk`, `diff`, `warnings`, and `rejected_operations`
8. `potpie graph commit <plan_id> --json` only for validated plans that policy allows
9. Verify with `graph history` or a follow-up `graph read`

## Ontology Primitives

| Primitive | Meaning |
|---|---|
| Ontology bundle | Versioned set of all subgraph contracts. |
| Subgraph contract | Purpose, entity types, relation types, views, mutations, authority, identity, examples. |
| Entity type | Stable object type with required fields, identity policy, lifecycle fields, and aliases. |
| Relation type | Directed predicate with allowed endpoint types, validity semantics, and authority rules. |
| Claim | Canonical fact about an entity or relation with truth class, evidence, confidence, and time. |
| Event | Append-only timeline occurrence, usually source-system generated. |
| View | Bounded read contract with required/optional scope, result shape, ranking, budget, and coverage rules. |
| Semantic mutation | Agent-write operation validated against ontology and source authority. |
| Evidence ref | Pointer to source material, quote/locator, authority class, freshness, and resolver hints. |
| Identity record | Canonical key, display name, aliases, external IDs, merge/split history. |
| Quality finding | Stale, duplicate, conflict, unsupported, orphan, low-confidence, or projection-drift finding. |

## Entity Keys

Entity keys should be deterministic, readable, and pot-scoped by context.

| Type Family | Key Pattern |
|---|---|
| Repository | `repo:<provider-host>:<owner>/<name>` |
| Service | `service:<pot-or-system>:<slug>` |
| Environment | `environment:<slug>` |
| Component | `component:<repo-or-service>:<slug>` |
| Code asset | `code:<repo>:<path>#<symbol>` or `code:<repo>:<path>` |
| Feature | `feature:<system-or-repo>:<slug>` |
| Decision | `decision:<source-or-pot>:<slug>` |
| Pull request | `pr:<provider>:<owner>/<repo>:<number>` |
| Commit | `commit:<provider>:<owner>/<repo>:<sha>` |
| Issue | `issue:<provider>:<project-or-repo>:<id>` |
| Incident | `incident:<source>:<id-or-slug>` |
| Bug pattern | `bug-pattern:<scope>:<symptom-slug>` |
| Document | `doc:<source>:<id-or-url-hash>` |
| Source reference | `source-ref:<source-system>:<external-id>` |
| Activity | `activity:<source-system>:<source-event-id>` |

Rules:

- Prefer authoritative external IDs when available.
- Otherwise use scoped slugs derived from canonical names.
- Never use display names alone when provider IDs exist.
- `search-entities` must run before creating non-authoritative entities.
- Merges create `MERGED_FROM` history and alias records; they do not hard-delete.

## Truth Classes

| Truth Class | Meaning | Auto-Commit Default |
|---|---|---:|
| `authoritative_fact` | Direct source-of-truth field from an authoritative source for that field. | Yes when low risk |
| `source_observation` | Observed output or event, not necessarily durable truth. | Yes when append-only |
| `agent_claim` | Inference grounded in evidence. | Yes for low-impact links; review for state changes |
| `user_decision` | Explicit decision from user/team/source of record. | Review unless append-only |
| `preference` | Durable user/team/project preference. | Review for broad scope |
| `timeline_event` | Append-only historical activity. | Yes |
| `quality_finding` | System-generated graph quality diagnosis. | No graph repair without proposal |

## Source Authority

Authority is field-specific, not source-wide.

| Authority Class | Examples | Can Create |
|---|---|---|
| `repository_metadata` | PR title/body/status, commit metadata, CODEOWNERS, manifests | PRs, commits, code links, ownership hints |
| `product_tracker` | Linear/Jira project/issue fields | feature status, issue state, roadmap links |
| `infrastructure_inventory` | Kubernetes, Helm, Terraform, cloud inventory | services, environments, deployment topology |
| `observability_signal` | alerts, metrics, logs, incident tools | alerts, incidents, diagnostic signals |
| `documentation` | ADRs, runbooks, design docs | decisions, constraints, runbooks, doc references |
| `conversation` | Slack, review threads, user chat | user decisions, preferences, discussion observations |
| `agent_inference` | harness-generated reasoning over evidence | `agent_claim` only unless reviewed |
| `manual_user_input` | direct user command or confirmation | user decision, preference, review approval |

Authority rules live in subgraph contracts, for example:

```json
{
  "field": "Feature.status",
  "authoritative": ["product_tracker", "manual_user_input"],
  "allowed_claims": ["agent_claim"],
  "review_required": ["agent_claim"]
}
```

## Seed Subgraphs

The ontology is organized around agent use cases, not storage tables. A physical
entity can appear in multiple subgraphs through different views.

| Subgraph | Primary Use |
|---|---|
| `project_map` | Pot, repositories, systems, services, components, source registry, high-level topology. |
| `code_topology` | Code assets, ownership, modules, files, symbols, code-to-feature links. |
| `features` | Product capabilities, features, requirements, roadmap, implementation links, status. |
| `infra_topology` | Runtime topology: environments, deployment targets, dependencies, integrations, datastores. |
| `recent_changes` | Timeline of PRs, commits, issues, deployments, activities, and touched subjects. |
| `decisions` | Decisions, constraints, preferences, agent instructions, local workflows. |
| `operations` | Deployments, alerts, incidents, metrics, runbooks, config, scripts. |
| `debugging` | Bug patterns, investigations, diagnostic signals, fixes, root causes. |
| `knowledge` | Documents, conversations, observations, source references, source systems. |
| `quality` | Duplicate/stale/conflict/orphan/low-confidence findings and maintenance jobs. |

### `project_map`

| Contract Part | Values |
|---|---|
| Use when | Onboarding, repo/service discovery, source setup, high-level project map. |
| Entity types | `Pot`, `Repository`, `System`, `Service`, `Component`, `SourceSystem`, `SourceReference`. |
| Relation types | `SCOPES`, `CONTAINS`, `BACKED_BY`, `FROM_SOURCE`, `EVIDENCED_BY`. |
| Views | `overview`, `service_index`, `repo_context`, `source_coverage`. |
| Mutations | `upsert_entity`, `patch_entity`, `link_entities`, `reconcile_snapshot`, `assert_claim`. |
| Authority | repo metadata, explicit harness evidence, manual user input. |

Required first views:

| View | Required Scope | Returns |
|---|---|---|
| `overview` | none | systems, repos, services, source freshness, coverage gaps |
| `service_index` | optional `repo`, `system` | ranked services/components with owners and source refs |
| `repo_context` | `repo` | repo metadata, source systems, components, linked services |
| `source_coverage` | none | connected sources, last seen, unsupported source families |

### `code_topology`

| Contract Part | Values |
|---|---|
| Use when | Mapping code to services/features, review prep, ownership lookup. |
| Entity types | `CodeAsset`, `Component`, `Repository`, `PullRequest`, `Commit`, `Feature`. |
| Relation types | `OWNS_FILE`, `MODIFIED`, `TOUCHES_CODE`, `BACKED_BY`, `CONTAINS`. |
| Views | `code_asset_context`, `module_neighborhood`, `change_impact`, `ownership_by_path`. |
| Mutations | `upsert_entity`, `link_entities`, `assert_claim`, `reconcile_snapshot`. |
| Authority | repository metadata, CODEOWNERS, explicit harness evidence. |

### `features`

| Contract Part | Values |
|---|---|
| Use when | Feature work, planning, implementation lookup, status, linked bugs/decisions. |
| Entity types | `Capability`, `Feature`, `Functionality`, `Requirement`, `RoadmapItem`, `Component`, `CodeAsset`, `BugPattern`, `Incident`, `Issue`. |
| Relation types | `IMPLEMENTS`, `HAS_FUNCTIONALITY`, `DEFINES`, `EVOLVES`, `SUPPORTS`, `TOUCHES_CODE`, `AFFECTS`, `AFFECTED_BY`. |
| Views | `feature_context`, `implementation_map`, `requirements_map`, `feature_health`. |
| Mutations | `upsert_entity`, `patch_entity`, `transition_state`, `link_entities`, `assert_claim`, `supersede_claim`. |
| Authority | product tracker for status, repo metadata for implementation links, docs/user input for requirements. |

### `infra_topology`

| Contract Part | Values |
|---|---|
| Use when | Runtime dependency reasoning, incident triage, deployment impact, architecture mapping. |
| Entity types | `Service`, `Environment`, `DeploymentTarget`, `DeploymentStrategy`, `Interface`, `DataStore`, `Integration`, `Dependency`, `ConfigVariable`. |
| Relation types | `DEPLOYED_TO`, `HOSTS`, `HOSTED_ON`, `EXPOSES`, `USES`, `DEPENDS_ON`, `USES_DATA_STORE`, `USES_DEPLOYMENT_STRATEGY`, `CALLS`, `CONFIGURES`. |
| Views | `service_neighborhood`, `environment_map`, `dependency_path`, `config_context`, `integration_map`. |
| Mutations | `reconcile_snapshot`, `upsert_entity`, `link_entities`, `end_relation_validity`, `assert_claim`. |
| Authority | infrastructure inventory, manifests, OpenAPI, Helm/Kubernetes/Terraform evidence. |

### `recent_changes`

| Contract Part | Values |
|---|---|
| Use when | Understanding what changed, debugging regressions, review prep, timeline queries. |
| Entity types | `Activity`, `Period`, `Change`, `PullRequest`, `Commit`, `Issue`, `Deployment`, `Branch`, `Person`, `Agent`, `Team`. |
| Relation types | `PERFORMED`, `TOUCHED`, `IN_PERIOD`, `ADDRESSES`, `HAS_COMMIT`, `PART_OF`, `TARGETS`, `DEPLOYED_AS`, `CHANGED_BY`. |
| Views | `timeline`, `changes_near_scope`, `activity_pulse`, `pr_context`, `deployment_timeline`. |
| Mutations | `append_event`, `upsert_entity`, `link_entities`, `assert_claim`. |
| Authority | source-control metadata, issue tracker metadata, deployment events, ledger events. |

### `decisions`

| Contract Part | Values |
|---|---|
| Use when | Avoiding repeated decisions, checking constraints/preferences, updating team/user intent. |
| Entity types | `Decision`, `Constraint`, `Preference`, `AgentInstruction`, `LocalWorkflow`, `Conversation`, `Document`. |
| Relation types | `MADE_IN`, `AFFECTS`, `AFFECTS_CODE`, `APPLIES_TO`, `PREFERRED_FOR`, `INFORMS`, `RUNS`, `RESULTED_IN`, `SUPERSEDES`. |
| Views | `active_decisions`, `constraints_for_scope`, `preferences_for_scope`, `agent_instructions`, `workflow_context`. |
| Mutations | `append_event`, `upsert_entity`, `patch_entity`, `transition_state`, `supersede_claim`, `assert_claim`. |
| Authority | user input, ADR/docs, review threads, approved decision records. |

Policy:

- Superseding a decision is always `review_required`.
- Broad preferences that apply to a pot, repo, team, or all agents are
  `review_required`.
- Narrow workflow notes with source refs can auto-commit locally.

### `operations`

| Contract Part | Values |
|---|---|
| Use when | Deployment/debug/runbook/config/alert context. |
| Entity types | `Deployment`, `Branch`, `Alert`, `Incident`, `Metric`, `Runbook`, `Script`, `ConfigVariable`, `Service`, `Environment`. |
| Relation types | `TARGETS`, `DEPLOYED_AS`, `FIRED_IN`, `INDICATES`, `IMPACTS`, `MITIGATES`, `CONFIGURES`, `RUNS`, `REFERENCES_CODE`. |
| Views | `runbook_context`, `alert_context`, `incident_context`, `deployment_context`, `config_context`. |
| Mutations | `append_event`, `upsert_entity`, `transition_state`, `link_entities`, `assert_claim`, `reconcile_snapshot`. |
| Authority | incident tools, alerting, deployment systems, runbook docs, repository scripts. |

### `debugging`

| Contract Part | Values |
|---|---|
| Use when | Finding prior failures, root causes, fixes, diagnostic signals, recurrence patterns. |
| Entity types | `BugPattern`, `Investigation`, `Fix`, `DiagnosticSignal`, `Incident`, `Alert`, `Observation`, `PullRequest`, `Commit`. |
| Relation types | `MATCHES_PATTERN`, `DEBUGGED`, `OBSERVED_IN`, `RESOLVED`, `CHANGED_BY`, `HAS_SIGNAL`, `HAS_ROOT_CAUSE`, `SEEN_IN`, `FIXES`, `CAUSED`. |
| Views | `prior_occurrences`, `active_bug_context`, `diagnostic_signals`, `fix_history`, `causal_chain`. |
| Mutations | `upsert_entity`, `link_entities`, `assert_claim`, `append_event`, `supersede_claim`. |
| Authority | issue tracker, incidents, alerts, PR metadata, user/agent investigation evidence. |

### `knowledge`

| Contract Part | Values |
|---|---|
| Use when | Resolving documents, source refs, conversations, observations, and evidence. |
| Entity types | `Document`, `Conversation`, `Episode`, `Observation`, `SourceSystem`, `SourceReference`. |
| Relation types | `DESCRIBES`, `RESULTED_IN`, `EVIDENCED_BY`, `FROM_SOURCE`, `SUPPORTS`. |
| Views | `document_context`, `source_ref_context`, `evidence_for_entity`, `conversation_outcomes`. |
| Mutations | `upsert_entity`, `append_event`, `link_entities`, `assert_claim`. |
| Authority | docs systems, chat/review systems, source resolvers, explicit harness evidence. |

### `quality`

| Contract Part | Values |
|---|---|
| Use when | Maintaining long-lived graph quality and projection health. |
| Entity types | `QualityIssue`, `MaintenanceJob`, `MaterializedAccessPath`. |
| Relation types | `FLAGS`, `REPAIRS`, `MATERIALIZES`, `MERGED_FROM`, `SPLIT_FROM`, `RENAMED_FROM`, `SUPERSEDES`. |
| Views | `quality_summary`, `duplicate_candidates`, `stale_facts`, `conflicting_claims`, `orphan_entities`, `projection_drift`. |
| Mutations | `merge_duplicate_entities`, `supersede_claim`, `retract_claim`, `end_relation_validity`, `reconcile_snapshot`. |
| Authority | quality scans, validation reports, manual review. |

Policy:

- Quality scans produce findings and proposed repairs.
- Repairs are normal semantic mutation plans.
- Merge/split/supersede/retract operations are usually `review_required`.

## Read View Contract

Each view contract must define:

| Field | Meaning |
|---|---|
| `view` | Stable view name unique within subgraph. |
| `description` | One-sentence purpose. |
| `required_scope` | Scope keys required before execution. |
| `optional_scope` | Scope keys that improve ranking. |
| `query_support` | `none`, `keyword`, `semantic`, or `hybrid`. |
| `result_shape` | Entity/claim/event fields returned in `items[]`. |
| `ranking` | Ranking inputs and tie-breakers. |
| `source_policy_support` | Supported evidence payload modes. |
| `token_budget` | Default and max item/token limits. |
| `coverage` | What counts as complete, partial, empty, unsupported, stale. |
| `versions` | Subgraph versions included for stale-write protection. |
| `examples` | Valid CLI calls and response snippets. |

Views must return `unsupported` when scope or query mode is not implemented.
They must not silently broaden scope.

## Semantic Mutation DSL

Semantic mutations are the only ordinary write input. The validator lowers them
to backend mutations after checking schema, ontology, identity, authority, risk,
and expected versions.

| Operation | Required Inputs | Default Risk |
|---|---|---|
| `append_event` | stable event key, occurred_at, subject ids, evidence | low |
| `upsert_entity` | entity type, key or identity source, properties, evidence | low/medium |
| `patch_entity` | entity key, property patch, authority, evidence | medium |
| `transition_state` | entity key, field, expected from, to, evidence | medium/high |
| `link_entities` | relation type, endpoints, validity, evidence | low/medium |
| `end_relation_validity` | relation key or endpoints, valid_until, reason, evidence | medium/high |
| `reconcile_snapshot` | scope, completeness, entities, relations, source revision, evidence | medium/high |
| `assert_claim` | subject, predicate, object/value, truth, confidence, evidence | low/medium |
| `retract_claim` | claim key, reason, evidence | high |
| `supersede_claim` | old claim, new claim/ref, reason, evidence | high |
| `merge_duplicate_entities` | canonical entity, duplicate entities, reason, evidence | high |

Risk policy:

- `low`: append-only events, narrow links, narrow low-impact claims with evidence.
- `medium`: current-state updates, lifecycle changes, wider claims.
- `high`: merges, retractions, supersession, broad preferences, decision changes,
  large snapshot removals, destructive admin.

Local auto-commit is allowed only when:

- plan status is `validated`;
- risk is `low`;
- all operations have evidence;
- source authority is sufficient;
- expected subgraph versions match;
- mutation policy marks the operation auto-applicable.

## Plan Lifecycle

| State | Meaning |
|---|---|
| `draft` | Client-side file before proposal. |
| `validated` | Server accepted schema/policy and persisted a commit-ready plan. |
| `invalid` | Validation failed; never commit. |
| `conflict` | Expected versions are stale; reread and propose again. |
| `review_required` | Valid shape but policy requires approval. |
| `approved` | Human or configured policy approved a review-required plan. |
| `applied` | Commit succeeded and wrote mutation/audit records. |
| `expired` | Plan was not committed before expiry. |
| `abandoned` | Plan explicitly closed without commit. |

`commit` accepts only `plan_id`. It never accepts mutation payloads.

## Inbox Model

Inbox is for useful graph work that is not yet safe as a canonical fact.

Inbox items should contain:

- summary;
- pot id;
- evidence refs when available;
- suggested subgraphs;
- discovered entities or source refs;
- priority;
- created_by harness/user;
- expiry or review deadline when useful.

Inbox processing is normal graph work:

1. claim item;
2. catalog/describe/read/search-entities;
3. propose semantic mutation;
4. commit or mark review-required;
5. close inbox item with mutation id, rejected reason, or superseded item.

## Ontology Evolution

Ontology evolution is part of the product. It must be easy to add concepts, but
hard to corrupt existing graph meaning.

### Version Units

| Unit | Versioned By | When It Changes |
|---|---|---|
| Graph contract | `graph_contract_version` | Workbench command/envelope semantics change. |
| Ontology bundle | `ontology_version` | Any active subgraph contract changes. |
| Subgraph | integer or semantic version | Entity/relation/view/mutation/authority rules for that subgraph change. |
| View | subgraph version plus view revision | Result shape, ranking, scope, or source policy changes. |
| Mutation operation | operation schema version | Required fields, validation, or lowering semantics change. |
| Skill references | generated from ontology version | Skill examples or recipes need refresh. |

Reads return `ontology_version` and `subgraph_versions`. Proposals carry
`expected_subgraph_versions`.

### Change Classes

| Change Class | Examples | Policy |
|---|---|---|
| Additive | New optional field, new view, new relation with no old meaning change, new source authority entry. | Allowed with tests; bump subgraph version. |
| Tightening | New required evidence, stricter authority, narrower endpoint rule, lower auto-commit policy. | Bump subgraph version; existing facts stay readable; future writes follow new rule. |
| Shape change | View result changes, mutation schema changes, identity key format changes. | Requires new view or operation version unless still unreleased. |
| Meaning change | Predicate semantics change, entity split/merge, truth class reinterpretation. | Requires explicit ontology proposal, backfill/repair plan, and review. |
| Retirement | Deprecated entity/relation/view/mutation removed from ordinary use. | Mark deprecated first; keep reads/history; block new writes after retirement date. |

Because Graph V2 is unreleased, current draft surfaces can be replaced directly.
After release, breaking product changes require explicit versioning and a
deprecation plan.

### Evolution Workflow

1. **Capture need**
   - Use `potpie graph inbox add` or a design issue when agents repeatedly hit
     unsupported concepts, low-confidence generic claims, or awkward views.

2. **Write ontology proposal**
   - Name affected subgraphs.
   - Define problem and example agent tasks.
   - Add or update entity types, relation types, views, mutations, identity
     rules, source authority, risk policy, and examples.
   - Define whether existing facts need backfill or quality findings.

3. **Validate contract**
   - Schema lint: contracts parse and required fields exist.
   - Example lint: every `catalog`, `describe`, `read`, and `propose` example is
     executable or intentionally `not_implemented`.
   - Ontology lint: relation endpoints exist, identity rules exist, state
     transitions are legal, source authority rules reference known authority
     classes.
   - Skill lint: generated skill snippets and examples match the ontology.

4. **Add tests**
   - Unit tests for schema and validator behavior.
   - Read-view tests for result shape, coverage, ranking, unsupported behavior.
   - Mutation tests for valid, invalid, conflict, and review-required plans.
   - Quality tests if the change adds duplicate/stale/conflict detection.
   - Benchmark scenario if the concept is user-visible.

5. **Stage contract**
   - `draft`: visible only in code review/design docs.
   - `experimental`: available behind config or `describe --include-experimental`.
   - `active`: appears in normal `catalog` and skills.
   - `deprecated`: readable but not recommended for new writes.
   - `retired`: hidden from catalog, preserved in history/export.

6. **Apply data work**
   - Add deterministic backfill when possible.
   - Otherwise create quality findings or inbox items.
   - Large or risky repairs go through `propose` / `commit`, not direct rewrites.

7. **Release skills**
   - Regenerate ontology snippets used by skills.
   - Add workflow examples.
   - Verify installed skill drift via `potpie graph status`.

### Ontology Proposal Template

```markdown
# Ontology Change: <short name>

## Problem
What agent task fails or produces poor graph state today?

## Affected Subgraphs
- <subgraph>

## Contract Changes
- Entity types:
- Relation types:
- Views:
- Mutation operations:
- Identity rules:
- Source authority:
- Risk/commit policy:

## Examples
- catalog:
- describe:
- read:
- propose:

## Existing Data Impact
- none / backfill / quality findings / inbox processing

## Tests
- schema:
- read view:
- mutation validator:
- benchmark:

## Rollout
- draft / experimental / active
```

### Rules For Adding A New Entity Type

An entity type cannot be added until these are defined:

- owning subgraph;
- purpose and when agents should use it;
- identity policy and key pattern;
- required properties;
- alias/external ID behavior;
- valid relation endpoints;
- allowed mutation operations;
- authoritative sources for important fields;
- at least one read view that can return it, or a reason it is write-only;
- examples for creation and lookup.

### Rules For Adding A New Relation Type

A relation type cannot be added until these are defined:

- direction and plain-English meaning;
- allowed endpoint entity types;
- cardinality expectations when relevant;
- temporal validity behavior;
- source authority;
- whether LLM inference may assert it as `agent_claim`;
- whether it can auto-commit;
- inverse/read-view behavior when needed.

### Rules For Adding A New Read View

A read view cannot be added until these are defined:

- agent task it supports;
- required and optional scope;
- ranking and token budget;
- result schema;
- source policy behavior;
- coverage semantics;
- examples;
- fallback/unsupported behavior.

### Rules For Adding A New Mutation Operation

A mutation operation cannot be added until these are defined:

- JSON schema;
- validator rules;
- lowering behavior;
- idempotency key;
- expected version handling;
- risk policy;
- audit record shape;
- history output;
- tests for accepted, rejected, conflict, and review-required cases.

## Open Design Decisions

These should be answered before implementation locks:

1. Whether `Graph Workbench Port` is a new domain port or a set of methods on
   `GraphService`.
2. Whether MCP is shipped at all for Graph V2, and if so whether it exposes one
   `graph_command` tool or typed command-family tools.
3. Exact persistent format for ontology contracts: Python typed constants,
   JSON/YAML contracts, or generated Python from contracts.
4. Exact versioning syntax for ontology bundle and subgraphs.
5. Whether high-risk local plans require interactive approval, a config flag, or
   explicit `graph commit --approved-by <user-ref>`.
