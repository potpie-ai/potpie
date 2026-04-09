# Context Graph Architecture

## Objective

The context graph should let Potpie and downstream agents answer questions such as:

- What is this project, how is it structured, and what systems does it depend on?
- What decisions, constraints, and operating preferences should be respected here?
- How has the project evolved over time, and what changed recently?
- Who owns or influences different parts of the system?
- What incidents, alerts, bugs, or troubleshooting knowledge matter right now?

The graph must support two different jobs at the same time:

1. Capture rich, evolving knowledge from noisy external sources.
2. Expose a stable schema that agents can query predictably.

The target architecture should use Graphiti as the graph substrate, not as the sole owner of truth. Graphiti is strong at episodic ingestion, temporal semantics, provenance, hybrid retrieval, and typed extraction. Potpie still needs to own the ontology, deterministic identifiers, validation rules, and query contract that agents rely on.

## Design principles

1. Use one graph, not separate disconnected stores.
2. Use Graphiti as the storage and temporal graph substrate.
3. Keep Potpie-owned deterministic entities and edges as the ontology layer inside that same graph.
4. Require provenance and time semantics on every important fact.
5. Distinguish durable facts from inferred summaries.
6. Model for queryability first, ingestion second.
7. Prefer typed entities and typed relationships over large unstructured blobs.
8. Let the reconciliation agent propose mutations, but validate and apply them deterministically.
9. Treat the graph as a derived context layer, not the unquestionable source of truth.
10. Store rich source payloads primarily by reference, not by copying everything into the graph.

## Recommended graph shape

Use a single Graphiti-backed Neo4j graph with three logical layers.

| Layer | Responsibility | Owner |
|------|----------------|-------|
| Code graph | Files, symbols, imports, calls, code topology | Existing Potpie code graph |
| Episodic graph | Raw and summarized episodes, temporal memory, extraction, semantic search | Graphiti |
| Canonical ontology layer | Stable business and operating entities plus deterministic relationships in the same graph | Potpie reconciliation pipeline |

The important architectural decision is this:

- Graphiti is the graph engine, temporal memory layer, and retrieval substrate.
- Potpie ontology is the source of truth for canonical queryable context.

Graphiti should host the ontology layer, not replace it.

## Is Graphiti a good fit?

Yes, with a specific usage model.

Graphiti is a good fit for Potpie because it already provides the hardest parts of an evolving context graph:

- episodic ingestion
- provenance back to source episodes
- temporal validity and historical queries
- hybrid semantic plus lexical retrieval
- custom entity and edge types
- graph namespacing through `group_id`

That means Potpie does not need to build a graph engine, temporal fact model, or semantic graph retrieval stack from scratch.

Graphiti is not enough by itself for Potpie's needs if we treat raw extraction output as canonical truth. Potpie still needs deterministic control over:

- which labels and edge types are allowed
- which properties are required
- how `entity_key` values are formed
- how duplicates are merged
- how supersession and invalidation work
- which facts are durable versus inferred
- what read contract agents should use

The recommended conclusion is:

- use Graphiti through and through as the graph substrate
- define Potpie ontology on top of Graphiti
- use reconciliation to inject validated ontology-aligned facts into Graphiti
- use Graphiti search and temporal features to support recall and evidence lookup

## What the installed Graphiti package means for Potpie

The Graphiti package installed in this repo is sufficient to support the substrate model Potpie wants.

The important implementation facts are:

- Graphiti supports episode ingest with Potpie-provided custom `entity_types`, `edge_types`, and `edge_type_map`.
- Graphiti exposes direct CRUD namespaces for `EntityNode` and `EntityEdge`.
- Graphiti exposes `add_triplet(...)` for direct fact insertion.
- Graphiti stores temporal fact fields such as `valid_at`, `invalid_at`, and `expired_at` on entity edges.
- Graphiti search supports group scoping, node label filters, edge type filters, and temporal filtering.

This is enough to make Graphiti the underlying graph substrate for both:

- episodic memory and extraction
- Potpie canonical ontology facts

It also means Potpie should not build a second graph store or a parallel graph abstraction unless Graphiti proves insufficient later.

### Important implementation caveat

The installed package still has rough edges that Potpie should design around:

- node identity is fundamentally `uuid`-based, not business-key-based
- canonical business identity such as `entity_key` must be owned by Potpie
- Graphiti's generic graph model uses `:Entity` nodes with additional labels and `:RELATES_TO` edges with semantic type carried in edge properties
- some lower-level bulk-write behavior in the installed Neo4j path appears risky enough that Potpie should prefer controlled deterministic writes first and optimize later

This is manageable, but it changes how the architecture should be described.

## Potpie-on-Graphiti architecture

The architecture should be expressed as a thin Potpie ontology layer on top of Graphiti primitives.

### Graphiti layer

Graphiti should own:

- episode persistence
- episode-to-entity provenance links
- temporal edge semantics
- embeddings and semantic retrieval
- generic entity and fact storage primitives
- namespace isolation via `group_id`

### Potpie ontology adapter layer

Potpie should own a thin adapter on top of Graphiti that is responsible for:

- deterministic `entity_key` generation
- ontology label validation
- allowed edge validation
- canonical upsert semantics
- conflict handling and supersession
- exact read helpers for agent-facing queries

This adapter should not replace Graphiti. It should convert Potpie ontology operations into Graphiti node and edge writes.

### Reconciliation layer

The reconciliation agent should remain the planner, not the writer.

Its responsibilities:

- inspect source event payloads
- inspect Graphiti-derived candidates and evidence
- inspect existing canonical ontology state
- propose typed canonical mutations aligned to Potpie schema

The deterministic applier should then execute those mutations against Graphiti through the Potpie ontology adapter layer.

## Operating model

The architecture should be simple to describe and stable to extend.

### Write path

For each source event:

1. Record the normalized event.
2. Write the source narrative as a Graphiti episode.
3. Let Graphiti produce typed candidates and retrieval metadata.
4. Run reconciliation to decide what becomes canonical.
5. Apply canonical node and edge upserts with Potpie-managed identity, validation, and temporal state.
6. Link canonical facts back to supporting episodes, documents, conversations, or observations.
7. Invalidate or supersede older facts when new facts replace them.

### Query path

Potpie should expose two coordinated query modes over the same graph:

- canonical ontology reads for exact answers such as ownership, constraints, decisions, topology, and active operational context
- Graphiti retrieval for evidence lookup, fuzzy recall, historical rationale, and ambiguity resolution

Agent-facing tools should compose both modes instead of forcing every question through semantic search.

### Separation of concerns

Use Graphiti for:

- episodic source capture
- provenance to source episodes
- temporal retrieval
- typed extraction assistance
- hybrid semantic retrieval

Use Potpie-controlled writes for:

- canonical nodes with stable `entity_key`
- canonical edges with explicit temporal validity
- validation of allowed labels and edge types
- conflict handling, supersession, and invalidation
- exact read helpers and agent-facing query contracts

This is the core split: Graphiti is the substrate, Potpie is the schema owner.

### Storage policy

Most source data should not be duplicated in the graph in full.

The graph should primarily store:

- normalized canonical facts
- compact episodic summaries when useful for retrieval
- provenance and source references
- sync, freshness, and verification metadata
- ranking and lifecycle properties needed for query behavior

The graph should usually not be the long-term home for:

- full incident payloads
- full PR diffs
- entire conversations
- verbose logs
- large documents that already live in durable source systems

Recommended rule:

- store enough graph state to answer common agent questions quickly
- store references that let the agent fetch full detail from the source system when needed
- only persist large raw content when it materially improves recall, explainability, or offline operation

### Source references as first-class graph data

Every important entity, edge, and evidence item should be resolvable back to the source system.

Recommended fields:

- `source_type`
- `source_ref`
- `external_refs`
- `retrieval_uri` or equivalent resolver input
- `last_seen_at`

Recommended rule:

- agent answers should rely on graph facts for orientation
- detailed inspection should often resolve through source references rather than expecting the graph to contain every raw payload

### Source resolver contract

Source references should not be passive metadata. Potpie should define a resolver layer that can turn graph references into live source reads and verification actions.

Recommended resolver responsibilities:

- fetch current source detail for a `source_type` and `source_ref`
- fetch compact summaries when full payloads are unnecessary
- verify whether a canonical fact still matches the source of truth
- report source access failures, permission failures, and missing artifacts explicitly

Recommended resolver inputs:

- `source_type`
- `source_ref`
- `external_refs`
- `retrieval_uri` or structured locator fields
- optional scope hints such as repo, service, environment, file path, or incident id

Recommended resolver outputs:

- `found`
- `current_payload` or compact source summary
- `verified`
- `mismatch_reason`
- `source_unreachable`
- `permission_denied`
- `last_checked_at`

Recommended rule:

- source resolution should be a first-class product surface, not an implementation detail hidden behind ingestion code
- if the graph cannot resolve a source reference reliably, the fact should be treated as lower quality over time

## Graphiti constraints to design around

Graphiti is a good substrate, but Potpie should design around its limitations.

- Custom entity models cannot reuse Graphiti protected attribute names such as `uuid`, `name`, `group_id`, `labels`, `created_at`, `summary`, `attributes`, and `name_embedding`.
- If an entity pair is missing from `edge_type_map`, Graphiti can still capture the relationship with a generic fallback edge type. Potpie should not let those fallback edges become canonical facts automatically.
- Schema evolution is additive-friendly, but reclassifying old data into newly introduced types generally requires re-ingestion or reinterpretation.
- Graphiti is a flexible framework, not a full governance layer. Potpie still has to own validation, replay, identity, and exact query contracts.

These are acceptable tradeoffs, but they should shape the design from the start.

## Current feature focus

The system should be optimized first for the kinds of answers agents actually need to produce repeatedly.

The highest-value feature set is:

- explain why code looks the way it does
- tell an agent what rules, constraints, and preferences apply before it changes something
- identify owners, reviewers, and people with relevant context
- connect incidents and alerts to services, environments, recent changes, and runbooks
- expose evidence and historical rationale without forcing the agent to parse raw source material every time

That implies a practical build order:

- code-to-context bridges
- ownership and familiarity
- decisions, constraints, and preferences
- change history and evidence lookup
- runtime reliability context

## Canonical schema categories

The canonical schema should be organized around a few top-level domains instead of source-specific node types.

### 1. Scope and identity

These nodes define where knowledge belongs.

- `Pot`
  - The isolation boundary for context.
- `Repository`
  - Code repository mapped to the pot.
- `Service`
  - Deployable runtime unit.
- `Environment`
  - `local`, `staging`, `prod`, preview environments, region-specific deployments.
- `System`
  - Larger product or platform boundary containing many services or repos.

Core relationships:

- `(:Pot)-[:SCOPES]->(:Repository)`
- `(:System)-[:CONTAINS]->(:Service)`
- `(:Service)-[:BACKED_BY]->(:Repository)`
- `(:Service)-[:DEPLOYED_TO]->(:Environment)`
- `(:Environment)-[:HOSTS]->(:Service)`

### 2. Product and architecture knowledge

These nodes describe what the software does and how it is built.

- `Capability`
  - External functionality or product behavior.
- `Feature`
  - Concrete deliverable area within a capability.
- `Component`
  - Logical subsystem, module, package, or bounded context.
- `Interface`
  - API, event contract, queue, webhook, database contract.
- `DataStore`
  - Postgres, Redis, S3, Neo4j, external SaaS storage.
- `Dependency`
  - External system or library with operational significance.

Core relationships:

- `(:Feature)-[:IMPLEMENTS]->(:Capability)`
- `(:Component)-[:SUPPORTS]->(:Feature)`
- `(:Component)-[:EXPOSES]->(:Interface)`
- `(:Component)-[:DEPENDS_ON]->(:Dependency)`
- `(:Service)-[:USES_DATA_STORE]->(:DataStore)`
- `(:Service)-[:CALLS]->(:Service)`
- `(:Component)-[:OWNS_FILE]->(:CodeAsset)`

### 3. Delivery and operational context

These nodes capture the state of running systems and how they are operated.

- `Deployment`
  - Version or branch promoted into an environment.
- `Branch`
  - Git branch with operational meaning.
- `Alert`
  - Monitoring or incident signal.
- `Incident`
  - Operational issue with timeline and severity.
- `Runbook`
  - Human-usable remediation procedure.
- `Metric`
  - Named health indicator when worth modeling explicitly.

Core relationships:

- `(:Branch)-[:DEPLOYED_AS]->(:Deployment)`
- `(:Deployment)-[:TARGETS]->(:Environment)`
- `(:Alert)-[:FIRED_IN]->(:Environment)`
- `(:Alert)-[:INDICATES]->(:Incident)`
- `(:Runbook)-[:MITIGATES]->(:Incident)`
- `(:Incident)-[:IMPACTS]->(:Service)`

### 4. Team and ownership context

These nodes make agent answers actionable.

- `Person`
  - Human contributor or stakeholder.
- `Team`
  - Functional or product team.
- `Role`
  - On-call, tech lead, owner, reviewer, maintainer.

Core relationships:

- `(:Person)-[:MEMBER_OF]->(:Team)`
- `(:Person)-[:OWNS]->(:Service|:Component|:Feature)`
- `(:Person)-[:REVIEWS]->(:Change)`
- `(:Team)-[:OWNS]->(:Service|:Capability|:Runbook)`
- `(:Person)-[:ONCALL_FOR]->(:Service|:Environment)`

### 5. Change and decision memory

This is where Graphiti and reconciliation should work most closely.

- `Change`
  - Generic parent concept for important change events.
- `PullRequest`
- `Commit`
- `Issue`
- `Decision`
  - Canonicalized engineering or product decision.
- `Constraint`
  - Rules, do-not-do guidance, architecture constraints, compliance restrictions.
- `Preference`
  - Team/project style and workflow preferences.

Core relationships:

- `(:PullRequest)-[:PART_OF]->(:Change)`
- `(:Commit)-[:PART_OF]->(:PullRequest)`
- `(:PullRequest)-[:ADDRESSES]->(:Issue)`
- `(:Decision)-[:MADE_IN]->(:PullRequest|:Incident|:Document)`
- `(:Decision)-[:AFFECTS]->(:Feature|:Component|:Service|:CodeAsset)`
- `(:Constraint)-[:APPLIES_TO]->(:Service|:Component|:Feature|:Repository)`
- `(:Preference)-[:PREFERRED_FOR]->(:Repository|:Component|:Team)`

### 6. Knowledge artifacts and evidence

These nodes preserve why we believe something.

- `Document`
  - ADRs, product docs, design docs, Confluence pages.
- `Conversation`
  - Slack thread, incident thread, review discussion, planning thread.
- `Episode`
  - Graphiti ingested episode; remains the narrative source.
- `Observation`
  - Optional normalized evidence unit when direct modeling is useful.

Core relationships:

- `(:Episode)-[:DESCRIBES]->(:Change|:Incident|:Decision|:Document)`
- `(:Document)-[:DESCRIBES]->(:Feature|:Component|:Constraint)`
- `(:Conversation)-[:RESULTED_IN]->(:Decision)`
- `(:Observation)-[:SUPPORTS]->(:Decision|:Incident|:Constraint)`

## Code graph bridge model

The current code graph already knows files, functions, classes, and structural relationships. Do not duplicate that layer. Instead add a small bridge vocabulary from canonical context nodes to code nodes.

Recommended bridge targets:

- `CodeAsset`
  - Logical alias for existing file/symbol nodes in the code graph.

Recommended bridge relationships:

- `(:Component)-[:OWNS_FILE]->(:FILE)`
- `(:Feature)-[:TOUCHES_CODE]->(:FILE|:FUNCTION|:CLASS)`
- `(:Decision)-[:AFFECTS_CODE]->(:FILE|:FUNCTION|:CLASS)`
- `(:PullRequest)-[:MODIFIED]->(:FILE|:FUNCTION|:CLASS)`
- `(:Incident)-[:INVOLVES_CODE]->(:FILE|:FUNCTION|:CLASS)`
- `(:Runbook)-[:REFERENCES_CODE]->(:FILE)`

This lets agents move across:

- code -> why
- code -> owner
- code -> incidents
- code -> decisions
- feature -> code footprint

## Provenance model

Every canonical fact should be explainable. Use provenance as a first-class concern.

Each entity and edge written by reconciliation should carry:

- `pot_id`
- `entity_key` or deterministic relationship identity
- `source_event_id`
- `episode_uuid` when applicable
- `source_type`
- `source_ref`
- `confidence`
- `created_at`
- `updated_at`
- `invalidated_at` when superseded

This aligns with the existing reconciliation domain, where deterministic entity and edge upserts are applied with a `ProvenanceRef`.

Recommended rule:

- Graphiti owns narrative provenance.
- Potpie canonical nodes and edges own factual provenance.
- Agents should always be able to trace a fact back to the event and episode that produced it.

Recommended extension:

- facts should also be traceable back to the current source location needed to re-verify them later
- provenance is not only for explanation, it is also for future alignment and repair

## Temporal model

Temporal semantics are essential here. A graph without time will quickly become misleading.

Store at least three kinds of time:

1. `event_time`
   - When the underlying thing happened.
2. `observed_at`
   - When Potpie ingested or learned it.
3. `valid_from` / `valid_to`
   - When the fact should be considered true in-world.

Examples:

- A deployment happened at one time, but Potpie may ingest it later.
- A team ownership edge may be valid for a period and then superseded.
- A preference may still exist historically but should no longer be used by agents.

Recommended rule:

- Use Graphiti bi-temporal semantics for episodic memory.
- Mirror temporal state into canonical edges for any fact agents will directly rely on.
- Distinguish event time, observed time, and verification time.

## Reconciliation-agent architecture

The reconciliation flow should be treated as a two-stage contract.

### Stage 1: episodic write

Source events are converted into rich `EpisodeDraft` values. Episodes should include:

- a concise title
- source description
- normalized timestamps
- the raw or summarized narrative needed for later re-interpretation
- explicit sections for entities, relationships, evidence, and unresolved ambiguities when possible

### Stage 2: canonical mutation plan

The reconciliation agent should produce a constrained mutation plan that maps the episode into:

- `entity_upserts`
- `edge_upserts`
- `edge_deletes`
- `invalidations`

This already matches the existing reconciliation domain model and is the right boundary to keep.

### Critical design rule

The agent should not be allowed to invent arbitrary schema at write time.

Instead:

- define an approved catalog of labels and edge types
- define required properties for each major type
- validate every mutation plan against that catalog
- reject or quarantine uncertain mutations rather than writing ambiguous graph state

## Recommended ingestion contract by source

Different sources should populate different parts of the same schema.

| Source | Main episode content | Canonical entities likely produced |
|------|----------------------|------------------------------------|
| GitHub PRs | intent, diff summary, review discussion, linked issues | `PullRequest`, `Commit`, `Issue`, `Decision`, `Feature`, `Person` |
| Linear/Jira | bug reports, project planning, status changes | `Issue`, `Feature`, `Capability`, `Decision`, `Constraint` |
| Alerts/Sentry/PagerDuty | failures, symptoms, timeline, impact | `Alert`, `Incident`, `Service`, `Environment`, `Runbook` |
| Docs/ADR/Confluence | architecture, rationale, standards | `Document`, `Decision`, `Constraint`, `Preference`, `Component` |
| Agent sessions | local discoveries, debugging trails, temporary conclusions | `Observation`, `Decision`, `Constraint`, `Preference` |
| Dev tooling/CI | build failures, deployments, branch movement | `Deployment`, `Branch`, `Incident`, `Environment` |

The key is source normalization. Agents should query by domain meaning, not by source type.

## Schema shape for agent querying

To keep querying simple, agents should mostly retrieve through a small set of stable entrypoints.

Recommended query families:

1. Identity and topology
   - What services, components, repos, and environments exist?
2. Ownership and responsibility
   - Who owns this service, path, feature, or incident domain?
3. Decision and constraint recall
   - What decisions or constraints apply to this component or repo?
4. Change history
   - What PRs, incidents, or documents changed the understanding of this area?
5. Runtime context
   - What environments, alerts, incidents, and deployments affect this service?
6. Preferences and conventions
   - What coding, review, architecture, or operational preferences should be respected?

Recommended agent-facing patterns:

- Query canonical nodes first.
- Use Graphiti semantic search only to discover candidate evidence or fill recall gaps.
- Return supporting episodes/documents alongside canonical facts.
- Rank by recency, confidence, and scope proximity.

In other words:

- canonical graph for precision
- Graphiti search for recall

## Suggested canonical labels

Start with a smaller schema and expand carefully. A good first stable set is:

- `Pot`
- `Repository`
- `System`
- `Service`
- `Environment`
- `Component`
- `Capability`
- `Feature`
- `Interface`
- `DataStore`
- `Dependency`
- `Person`
- `Team`
- `PullRequest`
- `Commit`
- `Issue`
- `Decision`
- `Constraint`
- `Preference`
- `Document`
- `Incident`
- `Alert`
- `Metric`
- `Runbook`
- `Deployment`
- `Branch`
- `Observation`

Avoid adding labels that are just source-specific variants unless they materially improve querying.

## Suggested canonical edge vocabulary

Keep edge names semantically strong and reusable.

- `SCOPES`
- `CONTAINS`
- `BACKED_BY`
- `DEPLOYED_TO`
- `HOSTS`
- `IMPLEMENTS`
- `SUPPORTS`
- `EXPOSES`
- `DEPENDS_ON`
- `USES_DATA_STORE`
- `CALLS`
- `OWNS`
- `OWNS_FILE`
- `TOUCHES_CODE`
- `AFFECTS`
- `AFFECTS_CODE`
- `ADDRESSES`
- `PART_OF`
- `MADE_IN`
- `APPLIES_TO`
- `PREFERRED_FOR`
- `MITIGATES`
- `IMPACTS`
- `FIRED_IN`
- `INDICATES`
- `DESCRIBES`
- `RESULTED_IN`
- `SUPPORTS`
- `MODIFIED`
- `INVOLVES_CODE`
- `REFERENCES_CODE`

Do not create separate edge names for every source system. Normalize source events into shared semantics.

## Required properties for high-value entities

High-value entities should have a small, consistent required property set.

### `Service`

- `entity_key`
- `name`
- `description`
- `system_key`
- `criticality`
- `lifecycle_state`

### `Component`

- `entity_key`
- `name`
- `component_type`
- `repository_key`
- `path_hint`

### `Decision`

- `entity_key`
- `title`
- `summary`
- `status`
- `decision_time`
- `confidence`

### `Constraint`

- `entity_key`
- `statement`
- `constraint_type`
- `status`

### `Preference`

- `entity_key`
- `statement`
- `preference_type`
- `scope_kind`
- `strength`

### `Incident`

- `entity_key`
- `title`
- `severity`
- `status`
- `started_at`

### `PullRequest`

- `entity_key`
- `number`
- `title`
- `repo_name`
- `author`
- `merged_at`

## What should be canonicalized vs left episodic

Canonicalize:

- stable project topology
- ownership
- explicit decisions
- active constraints
- current preferences
- incidents and alerts with operational significance
- important change records

Leave primarily episodic:

- raw conversations
- low-confidence speculation
- verbose debugging transcripts
- temporary hypotheses
- duplicate mentions that do not introduce a new durable fact

Promote episodic content into canonical facts only when:

- it materially changes future agent behavior
- it is likely to be queried repeatedly
- it has enough evidence to stand as a graph fact

## Validation rules for the reconciliation agent

The reconciliation agent should be constrained by schema-aware validation.

Validation should check:

- allowed labels
- allowed edge types
- required properties by label
- allowed start/end label families for each edge type
- scoped writes only within the target `pot_id`
- deterministic `entity_key` format
- provenance presence
- confidence thresholds for sensitive facts

Useful policy split:

- high confidence -> write canonical fact
- medium confidence -> write observation + evidence link
- low confidence -> keep only in episode

## Query architecture recommendation

Expose retrieval through two coordinated paths:

- canonical graph queries for exact answers
- episodic search for recall, evidence, and disambiguation

Best practice for agent tools:

1. Resolve the target scope.
2. Query canonical graph first.
3. Use episodic search to enrich or justify.
4. Return both facts and evidence references.

The later `Agent query contract` section defines the public read surface in more detail.

## Scenario walkthroughs

The best way to validate this architecture is to test it against concrete agent workflows. The scenarios below are deliberately chosen to stress the parts most likely to break: code-to-context traversal, stale facts, partial evidence, and ambiguous extraction.

### Scenario 1: "Why is this function implemented this way?"

User asks:

- Why does `billing/retry_failed_invoice` use a delayed queue instead of retrying inline?

#### Happy path

1. Agent resolves the code target to an existing code node such as `FUNCTION`.
2. Structured retrieval follows bridge edges:
   - `FUNCTION -> MODIFIED <- PullRequest`
   - `FUNCTION -> AFFECTS_CODE <- Decision`
   - `FUNCTION -> TOUCHES_CODE <- Feature`
3. Canonical graph returns:
   - recent PRs touching the function
   - linked decisions affecting the function
   - linked incidents or constraints if retries were changed after an outage
4. Graphiti search is used only to enrich:
   - review discussion
   - ADR language
   - rationale from the episode body
5. Agent answers with:
   - the active decision
   - the originating PR/issue
   - the operational reason, for example rate limiting or duplicate charge prevention
   - supporting evidence refs

#### Non-happy path

Possible failure modes:

- No function-level bridge exists because old diff hunks no longer align.
- Graphiti extracted a `Decision`, but reconciliation did not canonicalize it.
- Multiple PRs mention retries, but only one is still relevant.
- A historical decision was later superseded, but the old node still looks active.

Required schema/architecture behavior:

- Always maintain file-level fallback bridges even when symbol-level mapping fails.
- Add `status` and `valid_to` on `Decision`, `Constraint`, and `Preference`.
- Model supersession explicitly:
  - `(:Decision)-[:SUPERSEDED_BY]->(:Decision)`
- Keep `Decision -> AFFECTS_CODE` and `Decision -> AFFECTS -> Component|Service|Feature` both available.
- Require evidence links for high-value decisions:
  - `(:Decision)-[:SUPPORTED_BY]->(:Episode|:Document|:Conversation|:Observation)`

Refinement implied by this scenario:

- `Decision` should be treated as a first-class canonical type, not just a Graphiti extraction artifact.

### Scenario 2: "What should I look at for this production alert?"

User asks:

- We have elevated 5xx errors in checkout-prod. What services, recent changes, and runbooks matter?

#### Happy path

1. Agent resolves `checkout-prod` to `Environment`.
2. Structured retrieval traverses:
   - `Environment <- TARGETS - Deployment`
   - `Environment <- FIRED_IN - Alert`
   - `Incident <- INDICATES - Alert`
   - `Incident -> IMPACTS -> Service`
   - `Service -> BACKED_BY -> Repository`
   - `PullRequest -> MODIFIED -> FILE|FUNCTION`
3. Agent asks for:
   - recent deployments in that environment
   - active incidents
   - impacted services
   - runbooks mitigating those incidents
   - recent PRs touching impacted code or service components
4. Graphiti search supplements:
   - incident timeline
   - debugging notes
   - postmortem conclusions
5. Agent returns:
   - likely impacted services
   - recent risky changes
   - current owners/on-call
   - runbook links

#### Non-happy path

Possible failure modes:

- Alert exists, but no incident was created yet.
- Environment is known, but deployment lineage is missing.
- A runbook exists in docs, but was never canonicalized.
- The alert is noisy and should not imply a durable incident.
- Multiple services share the environment and the graph cannot rank likely blast radius.

Required schema/architecture behavior:

- Allow `Alert` to exist without requiring `Incident`.
- Add confidence and severity to `INDICATES` edges, not only to nodes.
- Model direct environment/service impact as fallback:
  - `(:Alert)-[:IMPACTS]->(:Service|:Environment)`
- Permit `Runbook` retrieval via Graphiti/document search when no canonical link exists yet.
- Add edge properties for ranking:
  - `confidence`
  - `last_observed_at`
  - `severity`
  - `impact_score`

Refinement implied by this scenario:

- Some operational meaning belongs on relationships, not only entities. This is especially true for alert-to-incident and service-to-environment impact edges.

### Scenario 3: "What conventions should I follow before changing this repo?"

User asks:

- I’m about to modify the worker system. What project conventions and constraints should I respect?

#### Happy path

1. Agent resolves the repo and affected component.
2. Structured retrieval traverses:
   - `Repository <- PREFERRED_FOR - Preference`
   - `Component <- APPLIES_TO - Constraint`
   - `Team -> OWNS -> Component`
   - `Document -> DESCRIBES -> Constraint|Preference|Component`
3. Agent collects:
   - coding preferences
   - architecture constraints
   - ownership and review expectations
   - linked docs and ADRs
4. Graphiti search fills gaps from:
   - prior code review episodes
   - agent session observations
   - recent decisions not yet lifted into durable constraints
5. Agent answers with:
   - hard constraints
   - softer preferences
   - owner/team to involve
   - supporting docs and recent decisions

#### Non-happy path

Possible failure modes:

- Preferences were inferred from reviews but never stabilized.
- Constraints conflict, for example "use Celery" vs "Hatchet allowed for context graph only".
- Preferences are stale and no longer valid.
- A rule applies only to one subtree or environment, not the whole repo.

Required schema/architecture behavior:

- Separate `Constraint` from `Preference`; do not collapse both into one generic rule type.
- Add scope specificity:
  - `scope_kind` such as `repo`, `component`, `path`, `service`, `environment`
  - `scope_ref`
- Add status fields:
  - `active`
  - `deprecated`
  - `proposed`
  - `exception`
- Model exceptions explicitly:
  - `(:Constraint)-[:EXCEPTION_FOR]->(:Component|:Service|:Environment|:CodeAsset)`
- Keep low-confidence conventions as `Observation` until reinforced across multiple events.

Refinement implied by this scenario:

- You need rule scoping and exception modeling early. Otherwise agent guidance will become over-broad and wrong.

### Scenario 4: "Who should review or own this change?"

User asks:

- I’m changing the ingestion queue path. Who likely owns it and who has the most context?

#### Happy path

1. Agent resolves the file/function/component.
2. Structured retrieval traverses:
   - `FILE|FUNCTION <- MODIFIED - PullRequest`
   - `PullRequest <- REVIEWS - Person`
   - `Person -[:OWNS]-> Component|Service`
   - `Person -[:MEMBER_OF]-> Team`
3. Ranking combines:
   - recency of changes
   - frequency of authored/reviewed changes
   - explicit ownership edges
   - on-call or incident involvement
4. Agent returns:
   - current owner
   - likely reviewers
   - recent decision-makers in this area

#### Non-happy path

Possible failure modes:

- Frequent contributors are no longer owners.
- Ownership is at team level only.
- Review history is present, but the changed file has moved.
- Recent contributors fixed incidents there but are not maintainers.

Required schema/architecture behavior:

- Treat ownership as explicit and durable, not purely inferred from change frequency.
- Keep inferred familiarity separate:
  - `(:Person)-[:FAMILIAR_WITH]->(:Component|:Service|:CodeAsset)`
- Add temporal weighting on familiarity edges.
- Model ownership transfer:
  - `valid_from`
  - `valid_to`
  - `ownership_type`

Refinement implied by this scenario:

- Distinguish `OWNS` from `FAMILIAR_WITH`. Agents need both, but they mean different things.

## Cross-scenario schema adjustments

These scenario traces suggest a few concrete improvements to the schema.

### Add explicit lifecycle semantics

The following types should have `status`, `valid_from`, and `valid_to` by default:

- `Decision`
- `Constraint`
- `Preference`
- `Incident`
- `Deployment`
- ownership and impact edges

### Add evidence links as first-class edges

High-value canonical facts should be backed by explicit evidence:

- `SUPPORTED_BY`
- `DERIVED_FROM`
- `SUPERSEDED_BY`
- `EXCEPTION_FOR`

This prevents canonical facts from becoming opaque.

### Distinguish hard facts from soft inferences

Not every extracted statement should become a durable node.

Recommended split:

- hard fact -> canonical node or edge
- soft but useful inference -> `Observation`
- raw unresolved narrative -> episode only

### Put ranking properties on edges

Many agent questions are really ranking problems. Edge properties matter for this.

Useful edge-level properties:

- `confidence`
- `strength`
- `last_observed_at`
- `observation_count`
- `valid_from`
- `valid_to`
- `source_priority`

### Prefer domain semantics over source semantics

Across all four scenarios, the agent succeeds when it can ask:

- what changed
- why it changed
- who owns it
- what applies here
- what evidence supports that

It does not help the agent to ask whether the source was GitHub, Slack, Sentry, or Confluence unless it is specifically doing evidence lookup.

That means source-specific details belong mostly in provenance and episode metadata, not in the public schema.

## Phased rollout

### Phase 1

Build the canonical foundation:

- `Repository`, `Service`, `Environment`, `Component`
- `Person`, `Team`
- `PullRequest`, `Issue`, `Decision`
- code graph bridge edges
- provenance and temporal properties

### Phase 2

Add project behavior and operating guidance:

- `Constraint`
- `Preference`
- `Document`
- `Capability`
- `Feature`

### Phase 3

Add runtime and reliability context:

- `Incident`
- `Alert`
- `Runbook`
- `Deployment`
- `Branch`

### Phase 4

Improve recall and agent ergonomics:

- observation modeling
- query scoring
- schema-specific prompts for reconciliation
- source-specific episode templates

## Identity model

Long-lived graph quality depends on stable identity rules. Potpie should treat identity as a first-class subsystem, not as an incidental property on nodes.

Each canonical entity should carry three distinct identity forms:

- `entity_key`
  - Potpie-owned canonical identity used for deterministic writes and exact reads.
- `external_refs`
  - Source-specific identifiers such as GitHub PR number, Jira issue key, PagerDuty incident id, doc URL, or service registry id.
- `aliases`
  - Alternate names, previous names, path variants, human shorthand, or source-local labels.

Recommended rule:

- `entity_key` is stable across time and source systems.
- display names may change without changing identity.
- source-specific ids should never become the only identity unless the entity is inherently source-native.

### Identity categories

Different domains need different key rules.

#### Stable business entities

Examples:

- `Repository`
- `Service`
- `System`
- `Team`
- `Environment`

Recommended key policy:

- use Potpie-owned slugs or scoped natural keys
- avoid keys based on mutable descriptions or titles
- allow name changes through alias/history tracking rather than key churn

#### Scoped logical entities

Examples:

- `Component`
- `Feature`
- `Interface`
- `DataStore`

Recommended key policy:

- make keys scope-aware
- include the owning repo, service, or system boundary where needed
- prefer semantic anchors over raw source ids

#### Time-bound source entities

Examples:

- `PullRequest`
- `Issue`
- `Deployment`
- `Alert`
- `Incident`

Recommended key policy:

- source ids may be part of the canonical key
- still normalize them into Potpie-owned `entity_key` format
- preserve original ids in `external_refs`

### Alias and change history

Potpie should preserve identity history explicitly.

Recommended edges:

- `ALIASES`
- `RENAMED_FROM`
- `MERGED_FROM`
- `SPLIT_FROM`

Recommended rules:

- never silently rewrite history when entities are merged
- preserve predecessor links when a service, component, or team is renamed
- model ambiguous identity as contested until reconciliation is confident

### Entity creation vs reuse

Reconciliation should use deterministic rules for deciding whether to create a new entity or attach evidence to an existing one.

Recommended decision order:

1. Exact `external_ref` match
2. Exact `entity_key` match
3. Alias match within scope
4. High-confidence semantic candidate within scope
5. Otherwise create a new provisional entity or observation

### Code identity guidance

Code-linked identity is more volatile than business identity.

Recommended rule:

- treat file/symbol identity separately from business entities
- preserve bridge edges even when code moves
- prefer file-level fallback when symbol-level resolution becomes unstable
- allow historical code references to remain valid through alias or relocation metadata

## Truth and conflict resolution

As Potpie ingests more sources, conflict handling becomes a core product behavior. The graph should not assume that the latest extracted statement is automatically true.

### Fact states

Canonical facts should carry an explicit truth state.

Recommended states:

- `accepted`
- `provisional`
- `contested`
- `superseded`
- `rejected`

This applies especially to:

- `Decision`
- `Constraint`
- `Preference`
- ownership edges
- impact edges
- incident relationships

### Authority vs confidence

Potpie should distinguish source authority from inference confidence.

- authority answers:
  - how much should this source count for this domain
- confidence answers:
  - how likely is this extracted or reconciled statement to be correct

Both should influence canonicalization, but they are not the same thing.

### Domain-specific source precedence

Each fact family should define its preferred evidence sources.

Examples:

- ownership:
  - explicit ownership config, team directories, or maintained metadata should outrank inferred familiarity
- runtime state:
  - monitoring and incident systems should outrank casual discussion
- decisions and constraints:
  - approved docs, merged PR outcomes, or explicit operator statements should outrank speculative chat

The important rule is:

- source precedence should vary by fact domain
- no global source ranking is sufficient

### Conflict policy

When sources disagree, Potpie should preserve the disagreement in evidence while being conservative in canonical truth.

Recommended rules:

- multiple contradictory evidence items may coexist
- only facts that pass reconciliation policy become active canonical facts
- unresolved conflicts should remain `contested`, not forced into false precision
- inferred facts should not invalidate explicit higher-authority facts on their own
- stale facts should decay in ranking and may become inactive when contradicted by newer authoritative evidence

### Override and exception handling

Manual or operator-confirmed truth should be modeled explicitly.

Recommended edges:

- `OVERRIDES`
- `EXCEPTION_FOR`
- `SUPERSEDED_BY`

Recommended properties:

- `override_reason`
- `authority_level`
- `source_priority`
- `last_observed_at`
- `observation_count`

### Canonicalization thresholds

Recommended policy split:

- high authority + high confidence:
  - write active canonical fact
- high authority + incomplete context:
  - write provisional canonical fact
- medium confidence or conflicting support:
  - write `Observation` plus evidence links
- low confidence:
  - keep in episode only

## Agent query contract

The graph should expose a stable public read contract for agents. The ontology alone is not enough. Agents need predictable entrypoints, consistent response shapes, and explicit explanation payloads.

### Query principles

- canonical queries answer precise questions
- episodic retrieval supplies evidence, recall, and ambiguity resolution
- every high-value answer should include supporting facts or evidence
- partial coverage should be visible to the caller
- temporal perspective should be explicit through `as_of`

### Recommended query families

Start with a small public surface.

#### 1. Identity and topology

Questions:

- what is this thing
- how is it connected
- what repo, service, environment, or component does it belong to

#### 2. Ownership and familiarity

Questions:

- who owns this area
- who is familiar with it
- who should review changes here

#### 3. Rules and guidance

Questions:

- what constraints apply here
- what preferences or conventions should be followed
- what exceptions exist

#### 4. Change and decision context

Questions:

- what changed recently
- why was this implemented this way
- which PRs, issues, incidents, or docs shaped this area

#### 5. Runtime and reliability context

Questions:

- what incidents, alerts, deployments, and runbooks matter here
- what environments or services are impacted

#### 6. Evidence lookup

Questions:

- what documents, conversations, or episodes support this answer
- what unresolved evidence exists

### Recommended response shape

Agent-facing query tools should return a common envelope.

Recommended fields:

- `answer`
- `facts`
- `evidence_refs`
- `confidence`
- `as_of`
- `open_conflicts`
- `coverage_gaps`
- `freshness`
- `verification_state`
- `needs_alignment`

Recommended behavior:

- `facts` should contain canonical graph results
- `evidence_refs` should point to supporting episodes, docs, or conversations
- `open_conflicts` should surface contested or contradictory support
- `coverage_gaps` should state when the graph is incomplete instead of pretending certainty
- `freshness` should indicate whether the answer is recent enough to trust directly
- `verification_state` should indicate whether the fact was checked against an external source recently
- `needs_alignment` should tell the caller when the graph should be refreshed before high-impact action

### Ranking policy

Many agent questions are ranking problems.

Recommended ranking inputs:

- source authority
- reconciliation confidence
- recency
- temporal validity
- scope proximity
- repetition or reinforcement count
- explicit ownership or override status
- freshness
- source reachability

### Temporal query contract

Every important read path should support:

- current state
- historical `as_of` reads
- inclusion or exclusion of superseded facts

Recommended rule:

- agents should know whether they are receiving current truth, historical truth, or both

### Suggested initial public queries

Examples:

- `resolve_scope`
- `get_owners`
- `get_familiar_people`
- `get_applicable_constraints`
- `get_preferences`
- `get_decisions`
- `get_change_context`
- `get_runtime_context`
- `search_evidence`
- `verify_against_source`
- `refresh_scope`

The names can change, but the concepts should stay stable.

## Operational alignment and drift management

The main long-term failure mode of a context graph is drift. Potpie should plan for the graph to become partially stale, incomplete, or contradictory over time, and should make recovery part of the architecture instead of an afterthought.

### Core stance

The context graph is a derived memory and alignment layer.

It is not the ultimate source of truth for:

- code structure
- live incident status
- external ticket state
- current ownership metadata
- large source-system payloads

The codebase and external systems remain the authoritative sources. The graph exists to add queryable memory, cross-source joins, ranking, recall, and agent ergonomics on top of them.

### Freshness and sync metadata

Important facts should carry explicit sync and verification state.

Recommended fields:

- `last_verified_at`
- `verified_against`
- `freshness_ttl`
- `sync_status`
- `staleness_reason`

Recommended states:

- `fresh`
- `stale`
- `needs_verification`
- `verification_failed`
- `source_unreachable`

This matters especially for:

- ownership
- service/environment topology
- incidents and alerts
- code bridges
- external artifact links

### Source-of-truth policy

Each fact family should declare where truth comes from and how the graph should treat it.

Recommended categories:

- authoritative external truth
  - the graph caches and contextualizes it
- authoritative code truth
  - the graph bridges to it and explains it
- canonicalized memory
  - the graph is the best queryable representation, but still retains source references
- soft inference
  - the graph suggests, but should not overrule stronger sources

Examples:

- ownership:
  - explicit ownership systems or maintained metadata outrank inferred familiarity
- code structure:
  - the codebase or code graph outranks graph memory
- incident state:
  - monitoring and incident tools outrank graph memory
- decisions:
  - approved docs, merged PR outcomes, and explicit records are strong sources
- preferences:
  - often soft unless repeatedly reinforced or documented

### Drift detection and housekeeping

Potpie should run recurring alignment and cleanup workflows, not just append more data.

Recommended job families:

- `verify_entity`
- `verify_edge`
- `refresh_scope`
- `resync_source_scope`
- `rebuild_scope_from_truth`
- `repair_code_bridges`
- `expire_stale_facts`
- `compact_or_archive_evidence`
- `resolve_alias_candidates`
- `cleanup_orphans`

Housekeeping should check for:

- entities no longer seen in authoritative sources
- broken or outdated source references
- stale ownership or topology facts
- code bridge breakage after file or symbol movement
- duplicate entities that should be merged
- expired observations that should no longer affect ranking

### Agent verification behavior

Agents should be explicitly aware that graph state may be stale.

Recommended behavior:

- if a fact is `fresh` and low-risk, the agent may answer directly from the graph
- if a fact is stale or high-impact, the agent should verify against the source when possible
- if graph and source disagree, the agent should prefer the source and emit an alignment signal
- if graph coverage is poor, the agent should fall back to source tools without pretending graph certainty

Recommended rule:

- the graph should accelerate understanding, not suppress verification when verification matters

### Read-through verification and write-back correction

Agent workflows should support graph-first orientation followed by source verification.

Recommended flow:

1. Query the graph for context, ranking, and likely facts.
2. Inspect freshness and verification metadata.
3. Resolve detailed information from source systems or the codebase when needed.
4. If the source confirms the graph, update verification metadata.
5. If the source disagrees, emit an alignment event for reconciliation.
6. Write corrected canonical facts and supersede stale ones.

This lets Potpie add value on top of existing tools without making the graph a confusing shadow copy.

### Fact-family freshness policy

Different fact families should have different freshness expectations.

Recommended policy examples:

- ownership:
  - medium-lived; refresh regularly and verify before high-impact reviewer/owner recommendations
- code bridges:
  - refresh whenever code indexing changes and repair when paths or symbols move
- incident and alert state:
  - short-lived; refresh aggressively because runtime truth changes quickly
- deployments and branch/environment mappings:
  - short-lived to medium-lived depending on source stability
- decisions:
  - usually durable; re-verify when contradicted or when linked source artifacts change materially
- constraints and preferences:
  - medium-lived; verify when stale, conflicted, or before high-impact agent guidance
- documents and runbooks:
  - medium-lived; refresh links and summaries when source docs change

Recommended rule:

- freshness TTL should be declared per fact family, not globally
- verification should be risk-aware, with more aggressive checks for facts that influence agent actions directly

### Graph quality metrics

Potpie should monitor graph quality as a product concern.

Recommended metrics:

- freshness coverage
- stale fact count
- contested fact count
- source sync lag
- verification success and failure rates
- graph/source disagreement count
- broken source reference count
- orphan count
- unresolved alias candidate count
- percentage of agent answers requiring source fallback

## Materialized access patterns

Indexes alone are not enough. Potpie should identify the query paths that agents will ask repeatedly and be willing to maintain compact derived access structures for them.

### High-value materialized patterns

Recommended early patterns:

- code artifact -> recent PRs
- code artifact -> linked decisions
- code artifact -> likely owners and familiar people
- component or service -> active constraints and preferences
- environment -> active alerts, incidents, deployments, and runbooks
- entity -> current source references and best supporting evidence

These materialized paths may be represented through:

- direct canonical edges
- compact summary nodes
- precomputed ranking properties
- read models optimized for agent query families

Recommended rule:

- materialize what is repeatedly queried and expensive to reconstruct
- avoid materializing low-value joins that can be resolved on demand

### Materialization criteria

Before adding a derived access path, require:

- a repeated agent use case
- a measurable latency or quality benefit
- a defined refresh trigger
- a clear invalidation rule when source truth changes

### Refresh triggers for materialized paths

Derived access paths should refresh when:

- source events arrive that affect their endpoints
- code graph updates invalidate code bridges
- freshness TTL expires for facts used in ranking
- drift detection marks a path as unreliable

## Agent safety rules

The context graph should improve agent performance without encouraging false certainty.

### Core safety rules

- do not treat stale graph facts as hard truth for high-impact actions
- prefer authoritative source data when graph and source disagree
- surface uncertainty instead of hiding it behind polished answers
- distinguish remembered context from verified current state
- use the graph for orientation and ranking before using it for irreversible recommendations

### High-impact action guidance

High-impact actions include:

- code changes that depend on constraints or ownership
- incident response recommendations
- reviewer or escalation recommendations
- decisions based on current runtime status

Recommended rule:

- for these actions, stale or contested graph facts should trigger source verification whenever possible

### Response behavior when uncertainty remains

If verification cannot be completed:

- return the best graph-backed answer available
- include freshness and verification caveats
- identify the authoritative source that should be checked next
- avoid presenting unverified graph memory as confirmed truth

### Product stance

Potpie should add value on top of existing tools, not compete with them by pretending to be a perfect mirror.

The graph helps the agent:

- know where to look
- rank what matters
- remember what changed
- connect evidence across systems

The source systems and codebase still decide what is currently true.

## Query-oriented indexing and access patterns

The graph should be indexed and shaped around the queries agents will actually ask later, not only around ingestion convenience.

### Index the things needed for exact retrieval

Recommended index targets:

- `entity_key`
- `group_id` or `pot_id`
- high-value labels
- `source_ref` and `external_refs`
- `status`
- `valid_from` and `valid_to`
- `last_verified_at`
- freshness or sync fields used by agent queries

### Optimize for common traversal patterns

The graph should support fast traversals for:

- code -> decision
- code -> recent changes
- code -> owner or familiar people
- component/service -> constraints and preferences
- environment -> incidents, alerts, deployments, runbooks
- entity -> supporting evidence and source references

### Support hybrid reads intentionally

Recommended read pattern:

- graph for entity resolution, joins, ranking, and memory
- source systems for full detail and current-state confirmation

The graph should not force the agent to choose between speed and correctness.

### Keep retrieval payloads compact

Graph query results should prefer:

- summaries
- small fact sets
- references to source detail
- freshness and confidence indicators

Avoid using the graph as the default transport for large raw payloads unless there is a clear retrieval benefit.

## Ontology governance

If Potpie is expected to grow for years, ontology evolution must be governed explicitly.

### Type classes

The graph should distinguish three classes of schema.

#### Public canonical types

These are stable agent-facing entities and edges. They define the query contract.

#### Internal extraction types

These help Graphiti extraction or staging, but are not part of the public agent contract.

#### Source-local staging types

These are temporary structures used while onboarding a source before it is normalized into domain semantics.

Recommended rule:

- add new source-specific staging types freely if needed
- add public canonical types only when they materially improve repeated querying

### Ontology versioning

Potpie should version the canonical ontology.

Recommended rules:

- every canonical type and edge belongs to an ontology version
- additive changes are preferred
- breaking changes require compatibility strategy
- deprecated types remain readable before they stop being writable

### Introducing new canonical types

Before adding a new public entity or edge type, require:

- a target use case
- a clear query benefit
- identity rules
- required properties
- lifecycle semantics
- provenance expectations
- conflict policy
- migration or backfill strategy if needed

### Reserved common fields

Most canonical entities should share a common core.

Recommended common fields:

- `entity_key`
- `status`
- `valid_from`
- `valid_to`
- `confidence`
- `created_at`
- `updated_at`
- `source_refs`

Recommended common edge properties:

- `confidence`
- `strength`
- `valid_from`
- `valid_to`
- `last_observed_at`
- `source_priority`

### Deprecation policy

Recommended lifecycle:

1. introduce a successor type or edge
2. keep old reads compatible
3. stop writing deprecated shapes
4. migrate or reinterpret existing data as needed
5. remove deprecated shapes only after query/tool consumers have moved

### Default governance rule

Prefer mapping new sources into existing domain semantics before adding new public types.

This keeps the graph extensible without making the public schema unstable.

## Security and retention model

As Potpie expands to more sources, it will ingest data with different sensitivity and retention expectations. The graph design should model this directly.

### Visibility classes

Canonical facts and evidence should carry visibility metadata.

Recommended classes:

- `public_within_pot`
- `restricted`
- `sensitive`
- `secret_reference_only`

### Fact visibility vs evidence visibility

A canonical fact and its underlying evidence should not be assumed to have the same access level.

Recommended rules:

- a fact may be visible while the raw evidence is redacted
- a fact may need to be hidden if it is derived entirely from restricted evidence with no safe summary
- agents should know when supporting evidence exists but cannot be shown

### Source ACL inheritance

Each source integration should define how access control flows into:

- episode visibility
- evidence visibility
- canonical fact visibility

Recommended rule:

- provenance should preserve the source ACL context even after reconciliation

### Redaction model

Potpie should support safe summaries of restricted material.

Recommended behavior:

- preserve a minimal explainability trail even when raw content cannot be exposed
- allow evidence references to indicate restricted support without leaking underlying text
- avoid storing unnecessary sensitive raw text when a structured summary is sufficient

### Retention classes

Different data should age differently.

Recommended classes:

- durable
- medium-lived
- transient

Examples:

- explicit decisions and ownership facts:
  - usually durable
- raw chat transcripts and debugging threads:
  - often medium-lived or transient
- temporary observations and noisy alerts:
  - often transient unless promoted into durable facts

### Retention rules

Recommended rules:

- raw episodes may expire earlier than canonical facts derived from them
- expired evidence should not silently invalidate durable facts if Potpie has already retained structured provenance
- retention policy should be defined per source and artifact class
- agents should be able to see when evidence has expired or been redacted

### Sensitive-source onboarding rule

Before adding a new source, Potpie should define:

- default visibility class
- retention class
- redaction strategy
- whether canonical facts derived from that source are broadly shareable
- whether source text should be stored raw, summarized, or not stored at all

## Recommended implementation stance

The most important decision is to treat Graphiti as an episodic reasoning substrate and Potpie as the canonical schema owner.

Concretely:

- keep raw source richness in episodes
- use reconciliation to write typed facts
- preserve provenance on every fact
- keep schema small and strongly typed
- design query tools around canonical labels and bridge edges
- let Graphiti improve recall, not define the public schema

That gives you a graph that can evolve with new sources while staying stable enough for agents to query reliably.
