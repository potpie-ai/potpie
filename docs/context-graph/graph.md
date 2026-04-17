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

## Product model

The context graph is the project memory layer for Potpie and for external agents that need to work inside the context of a real project.

The primary product abstraction is a **pot**. A pot maps to an end-to-end project and contains the context needed to reason about that project across repositories, people, features, services, deployments, docs, integrations, operations, historical decisions, incidents, fixes, preferences, and agent instructions.

The graph should help agents answer:

- what project am I working in
- why does this feature, service, file, or workflow exist
- where does this work sit in the larger project
- which repos, services, docs, tickets, users, deployments, and environments are related
- what decisions, constraints, and preferences should be respected
- what changed recently
- what similar issue was debugged before and how it was fixed
- whether available context is fresh, stale, partial, missing, or source-unverified

The design target is an extensible context platform, not a single GitHub PR history feature.

### Pot

A pot is the tenant, project, and reasoning boundary for graph operations.

A pot can contain:

- repositories across one or more providers
- users, maintainers, reviewers, teams, and preferences
- features and user-facing functionality
- backend services, frontend apps, workers, jobs, packages, libraries, and integrations
- docs, tickets, PRs, incidents, alerts, runbooks, deployments, and environments
- local development workflows, scripts, setup notes, and project conventions

Agents should normally start from the active pot, then narrow by repo, feature, service, file, PR, ticket, incident, environment, or user.

### Context wrap

A **context wrap** is the agent-facing bundle returned for a task or query. It should be assembled from canonical graph facts, Graphiti recall, and live source resolvers. It is not just semantic search results.

A useful context wrap includes:

- orientation: pot, repos, feature/service scope, and likely task type
- evidence: relevant decisions, tickets, docs, PRs, incidents, discussions, and code links
- operational context: local scripts, deployments, environments, alerts, runbooks
- constraints and preferences: project rules, prior decisions, and team/user guidance
- freshness and fallback state: what was searched, what was unavailable, what is stale, and when graph context was last updated or verified
- source references: IDs, URLs, and resolver inputs that `context_resolve` can use for source-backed summaries, verification, or full source snippets when the request explicitly asks for them

### Source-reference-first storage

The context graph should stitch together project context. It should not become a second source of truth for every source payload.

The graph should store:

- stable references to source artifacts
- compact summaries and indexable key details
- canonical relationships across sources
- provenance, confidence, lifecycle, and freshness metadata
- resolver hints for fetching full source data through integrations, MCPs, or APIs

The graph should usually avoid storing:

- full PR diffs
- full documents
- full Slack or review threads
- large logs
- complete incident payloads
- high-volume telemetry streams

Exceptions are appropriate when small snippets materially improve recall, explanation, or offline usefulness.

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
- Potpie ontology owns the canonical queryable context contract.

Graphiti should host the ontology layer, not replace it.

This does not mean the context graph becomes the source of truth for every project fact. Source systems and the codebase remain authoritative for their own data. Potpie's canonical ontology is the stable graph representation agents query for orientation, joins, ranking, and memory.

## Target logical architecture

```text
Agents / IDEs / Potpie / MCP clients / CLI
    |
    |  context_resolve / context_search / context_record / context_status
    v
Context Intelligence Layer
    |
    |  evidence planning, coverage, fallbacks, source resolution
    v
Context Graph Application Layer
    |
    |  ingest events, reconcile, validate, apply, query
    v
Graph Storage and Workflow
    |
    +-- Graphiti episodic + temporal graph on Neo4j
    +-- Potpie canonical ontology nodes and edges on Neo4j
    +-- Existing code graph nodes and bridges on Neo4j
    +-- Postgres event ledger, reconciliation ledger, and step status
    +-- External source resolvers for full payloads
```

The current `app/src/context-engine` package already has many pieces of this shape:

- hexagonal package boundaries: `domain`, `application`, `adapters`, `bootstrap`
- pot-scoped Graphiti episodic ingestion using `group_id`
- structural Neo4j bridges from code graph nodes to PR, decision, and file history context
- Postgres-backed context events, ingestion events, reconciliation ledger, and step status
- Celery/Hatchet queue abstractions for async graph jobs
- HTTP, CLI, and MCP entrypoints
- raw episode ingest, merged GitHub PR ingest, semantic search, change history, file owners, decisions, PR review context, PR diff, project graph, and `resolve-context`
- a first provider-neutral context intelligence layer with `IntelligenceBundle`, coverage, errors, and capability metadata

The main implementation gap is not the skeleton. It is the domain breadth: the current graph mostly understands GitHub/code-history context, while the target system needs broader project context around features, users, services, environments, docs, integrations, operations, debugging knowledge, and preferences.

## Agent integration and context injection

The context graph should feel natural to agents. Agents should not need to understand Graphiti, Cypher, Neo4j labels, or source-specific schemas to benefit from the graph.

Supported integration surfaces should include:

- `AGENTS.md` instructions that teach an agent how to identify the active pot and when to ask for context
- skills that encode task-specific context recipes such as feature work, debugging, operations, docs lookup, or reviewer selection
- MCP tools for external agents and IDEs
- CLI commands for local workflows and setup checks
- HTTP APIs for Potpie agents and other services
- future SDKs for structured integrations

The default behavior should be:

1. Resolve the active pot.
2. Resolve the task scope: repo, feature, service, file, PR, ticket, incident, environment, or user.
3. Request a context wrap through `context_resolve`.
4. Inspect coverage, freshness, and source verification state.
5. Ask `context_resolve` for source-backed summaries, verification, or selected full source snippets only when the graph indicates it is needed.
6. Emit alignment events when source truth disagrees with graph memory.

The public agent surface should stay intentionally small. Do not add a new MCP or HTTP tool for every use case. Use-case-specific skills should describe `context_resolve` parameter recipes, not hard-code graph internals or call separate feature/debugging/operations tools. For example:

- a feature skill calls `context_resolve` with `intent: "feature"` and includes feature map, service map, docs, tickets, decisions, recent changes, owners, and preferences
- a debugging skill calls `context_resolve` with `intent: "debugging"` and includes prior fixes, diagnostic signals, incidents, alerts, recent changes, config, deployments, and owners
- an operations skill calls `context_resolve` with `intent: "operations"` and includes deployment targets, runbooks, alerts, incidents, scripts, config references, and source status
- a reviewer skill calls `context_resolve` with `intent: "review"` and includes ownership, familiarity, recent changes, decisions, and team context

This is how context becomes an agent operating layer rather than a search box.

### Minimal agent context port

The recommended public tool surface is:

- `context_resolve`
  - primary tool for task context, source-backed context wraps, source verification, and bounded source retrieval
- `context_search`
  - secondary tool for narrow follow-up lookup when the agent already knows what it is looking for
- `context_record`
  - generic write tool for durable learnings such as decisions, fixes, preferences, workflows, feature notes, source references, and incident summaries
- `context_status`
  - lightweight health, freshness, source coverage, and pot/scope readiness check

Do not create public tools such as `context_get_feature_context`, `context_get_debugging_context`, `context_get_operational_context`, or `context_get_source`. Those concepts should be expressed through `context_resolve` parameters.

The extension points are:

- `intent`
- `scope`
- `include`
- `exclude`
- `mode`
- `budget`
- `source_policy`
- `filters`
- `record.type`

This keeps the agent interface stable while allowing new context use cases to be added over time.

### `context_resolve` request contract

`context_resolve` should accept enough structure to route the request without requiring agents to know graph internals.

Recommended fields:

```json
{
  "pot_id": "pot_123",
  "query": "Fix timeout during GitHub repository ingestion",
  "intent": "debugging",
  "scope": {
    "repo": "potpie/api",
    "branch": "main",
    "files": [
      "app/src/context-engine/application/services/ingestion_submission_service.py"
    ],
    "services": ["context-engine"],
    "features": ["context graph ingestion"],
    "environment": "staging",
    "pr_number": 1234,
    "ticket_ids": ["ENG-42"],
    "user": "nandan",
    "source_refs": ["github_pr:1234", "doc:docs/context-graph/graph.md"]
  },
  "include": [
    "purpose",
    "feature_map",
    "service_map",
    "docs",
    "tickets",
    "decisions",
    "recent_changes",
    "owners",
    "prior_fixes",
    "operations",
    "preferences",
    "source_status"
  ],
  "exclude": [
    "full_diffs",
    "full_docs",
    "large_threads"
  ],
  "mode": "fast",
  "source_policy": "references_only",
  "budget": {
    "max_items": 12,
    "max_tokens": 2500,
    "timeout_ms": 3500,
    "freshness": "prefer_fresh"
  },
  "as_of": null
}
```

`intent` is a routing hint, not a separate endpoint. Initial values should include:

- `feature`
- `debugging`
- `review`
- `operations`
- `planning`
- `docs`
- `onboarding`
- `refactor`
- `test`
- `security`
- `unknown`

`include` controls evidence families. Initial values should include:

- `purpose`
- `feature_map`
- `service_map`
- `repo_map`
- `docs`
- `tickets`
- `decisions`
- `recent_changes`
- `owners`
- `prior_fixes`
- `incidents`
- `alerts`
- `deployments`
- `runbooks`
- `local_workflows`
- `scripts`
- `config`
- `preferences`
- `agent_instructions`
- `source_status`

`mode` controls depth and latency:

- `fast`
  - graph facts, compact summaries, source references, freshness, and fallbacks; no full external source payloads by default
- `balanced`
  - fast path plus selective source-backed summaries or verification for top-ranked references
- `deep`
  - may fetch larger source details when necessary and within budget
- `verify`
  - focuses on checking graph facts against authoritative source systems

`source_policy` controls how source references are used inside `context_resolve`:

- `references_only`
  - return source refs and resolver metadata without fetching source payloads
- `summary`
  - fetch compact source summaries for top-ranked refs when useful
- `verify`
  - verify whether selected graph facts still match source truth
- `snippets`
  - return bounded source snippets when the source is available and allowed
- `full_if_needed`
  - allow larger source reads only when `mode` and `budget` permit

This folds source fetching and verification into `context_resolve` without introducing a separate `context_source` tool.

### `context_resolve` response contract

Every `context_resolve` response should use a consistent envelope.

Recommended shape:

```json
{
  "ok": true,
  "request_id": "ctxreq_123",
  "pot": {
    "id": "pot_123",
    "name": "Potpie"
  },
  "resolved_scope": {
    "repos": ["potpie/api"],
    "features": ["context graph ingestion"],
    "services": ["context-engine"],
    "confidence": 0.82,
    "inferred": true
  },
  "answer": {
    "summary": "This task is in the context graph ingestion path. The service submits ingestion events and delegates async execution to Celery by default or Hatchet optionally.",
    "purpose": [],
    "decisions": [],
    "recent_changes": [],
    "prior_fixes": [],
    "debugging_memory": [],
    "docs": [],
    "owners": [],
    "operations": [],
    "preferences": []
  },
  "evidence": [
    {
      "id": "ev_1",
      "type": "decision",
      "title": "Celery is default queue backend",
      "summary": "Context graph jobs use the context-graph-etl queue unless Hatchet is explicitly configured.",
      "confidence": 0.94,
      "freshness": "fresh",
      "verification_state": "verified",
      "source_refs": ["doc:AGENTS.md"]
    }
  ],
  "source_refs": [
    {
      "ref": "doc:AGENTS.md",
      "type": "doc",
      "uri": "repo://AGENTS.md",
      "fetchable": true,
      "last_seen_at": "2026-04-17T08:00:00Z",
      "last_verified_at": "2026-04-17T08:00:00Z",
      "access": "allowed"
    }
  ],
  "coverage": {
    "status": "partial",
    "searched": ["docs", "decisions", "recent_changes", "prior_fixes"],
    "missing": ["production alerts integration is not connected"],
    "stale": [],
    "confidence": 0.78
  },
  "freshness": {
    "status": "mostly_fresh",
    "last_graph_update": "2026-04-16T18:22:00Z",
    "last_source_verification": "2026-04-17T08:00:00Z"
  },
  "quality": {
    "status": "watch",
    "metrics": {
      "source_ref_count": 4,
      "stale_ref_count": 0,
      "needs_verification_ref_count": 2
    },
    "issues": [],
    "recommended_maintenance": [
      {
        "job": "verify_entity",
        "reason": "Verify high-value facts before agents rely on them."
      }
    ]
  },
  "fallbacks": [
    {
      "code": "source_not_connected",
      "message": "No alert source is connected for this pot.",
      "impact": "Operational debugging context may be incomplete."
    }
  ],
  "open_conflicts": [],
  "recommended_next_actions": [
    {
      "action": "resolve",
      "mode": "verify",
      "source_policy": "verify",
      "reason": "Verify deployment facts before production-impacting changes."
    }
  ]
}
```

Agents should always inspect `coverage`, `freshness`, `quality`, `fallbacks`, `open_conflicts`, and `source_refs` before relying on the answer.

As of Phase 4, project-map includes are backed by a canonical `project_map_context` read path. When `context_resolve` is called with `intent: "feature"`, `intent: "operations"`, `intent: "onboarding"`, or includes such as `service_map`, `feature_map`, `docs`, `deployments`, `runbooks`, `local_workflows`, `scripts`, `config`, `preferences`, or `agent_instructions`, the response envelope can include:

```json
{
  "answer": {
    "project_map": [
      {
        "family": "service_map",
        "kind": "Service",
        "entity_key": "service:context-engine",
        "name": "context-engine",
        "summary": "Context graph ingestion and agent context service.",
        "relationships": [
          {
            "type": "BACKED_BY",
            "direction": "out",
            "target_kind": "Repository",
            "target_name": "potpie"
          }
        ],
        "source_uri": "repo://app/src/context-engine/README.md"
      }
    ]
  },
  "facts": {
    "project_map": []
  }
}
```

This is still reference-first context. The graph returns compact canonical facts and relationship references; agents should request source-backed verification through `context_resolve` when they need authoritative details.

As of Phase 5, debugging includes are backed by a canonical `debugging_memory_context` read path. When `context_resolve` is called with `intent: "debugging"` or includes such as `prior_fixes`, `diagnostic_signals`, `incidents`, or `alerts`, the response envelope can include:

```json
{
  "answer": {
    "debugging_memory": [
      {
        "family": "prior_fixes",
        "kind": "Fix",
        "entity_key": "fix:repo-ingestion-timeout",
        "title": "Repository ingestion timeout fix",
        "summary": "Increased provider timeout and added retry budget for large repositories.",
        "root_cause": "Repository metadata fetch exceeded the previous request timeout.",
        "fix_type": "configuration",
        "affected_scope": [
          {
            "type": "SEEN_IN",
            "target_labels": ["Service"],
            "target_name": "context-engine"
          }
        ],
        "diagnostic_signals": [
          {
            "type": "HAS_SIGNAL",
            "target_labels": ["DiagnosticSignal"],
            "target_name": "repository ingestion timeout"
          }
        ],
        "source_ref": "github:pr:1234"
      }
    ]
  },
  "facts": {
    "debugging_memory": []
  }
}
```

This remains a compact memory layer. Full incident timelines, alert payloads, logs, PR diffs, and debugging transcripts should stay in source systems and be fetched only through source-backed `context_resolve` modes when needed.

### Skill recipes over tool sprawl

Skills and `AGENTS.md` files should encode parameter presets.

Feature work recipe:

```json
{
  "intent": "feature",
  "include": [
    "purpose",
    "feature_map",
    "service_map",
    "docs",
    "tickets",
    "decisions",
    "recent_changes",
    "owners",
    "preferences",
    "source_status"
  ],
  "mode": "fast",
  "source_policy": "references_only"
}
```

Debugging recipe:

```json
{
  "intent": "debugging",
  "include": [
    "prior_fixes",
    "recent_changes",
    "incidents",
    "alerts",
    "diagnostic_signals",
    "config",
    "deployments",
    "owners",
    "source_status"
  ],
  "mode": "fast",
  "source_policy": "references_only"
}
```

Review recipe:

```json
{
  "intent": "review",
  "include": [
    "purpose",
    "decisions",
    "owners",
    "recent_changes",
    "prior_fixes",
    "docs",
    "preferences",
    "source_status"
  ],
  "mode": "balanced",
  "source_policy": "summary"
}
```

Operations recipe:

```json
{
  "intent": "operations",
  "include": [
    "service_map",
    "deployments",
    "incidents",
    "alerts",
    "runbooks",
    "scripts",
    "config",
    "owners",
    "source_status"
  ],
  "mode": "balanced",
  "source_policy": "summary"
}
```

The agent still calls `context_resolve`; the skill only changes parameters.

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
- store references that let `context_resolve` or another authorized source integration fetch full detail from the source system when needed
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
- detailed inspection should often resolve through source references via `context_resolve` source policies or source integrations rather than expecting the graph to contain every raw payload

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

- source resolution should be a first-class capability inside the context port, primarily exposed through `context_resolve` parameters rather than as a separate public tool
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
- `Functionality`
  - More granular behavior under a feature.
- `Requirement`
  - Expected behavior, product requirement, or acceptance criterion.
- `RoadmapItem`
  - Planned evolution or future direction.
- `Component`
  - Logical subsystem, module, package, or bounded context.
- `Interface`
  - API, event contract, queue, webhook, database contract.
- `DataStore`
  - Postgres, Redis, S3, Neo4j, external SaaS storage.
- `Integration`
  - External API, SDK, webhook, MCP, database, queue, or cloud service.
- `Dependency`
  - External system, service, or library with operational significance.

Core relationships:

- `(:Feature)-[:IMPLEMENTS]->(:Capability)`
- `(:Feature)-[:HAS_FUNCTIONALITY]->(:Functionality)`
- `(:Requirement)-[:DEFINES]->(:Feature|:Functionality)`
- `(:RoadmapItem)-[:EVOLVES]->(:Feature|:Capability)`
- `(:Component)-[:SUPPORTS]->(:Feature)`
- `(:Component)-[:EXPOSES]->(:Interface)`
- `(:Component)-[:USES]->(:Integration)`
- `(:Component)-[:DEPENDS_ON]->(:Dependency|:Service)`
- `(:Service)-[:USES_DATA_STORE]->(:DataStore)`
- `(:Service)-[:CALLS]->(:Service)`
- `(:Component)-[:OWNS_FILE]->(:CodeAsset)`

### 3. Delivery and operational context

These nodes capture the state of running systems and how they are operated.

- `Deployment`
  - Version or branch promoted into an environment.
- `DeploymentTarget`
  - GCP, AWS, Kubernetes cluster, Vercel, Render, or another deployment target.
- `DeploymentStrategy`
  - Rolling, blue-green, canary, manual, preview, or another strategy.
- `Branch`
  - Git branch with operational meaning.
- `Alert`
  - Monitoring or incident signal.
- `Incident`
  - Operational issue with timeline and severity.
- `Runbook`
  - Human-usable remediation procedure.
- `Script`
  - Local, CI, debug, or deployment command commonly used by the team.
- `ConfigVariable`
  - Important configuration variable or secret reference. Store references and usage context, not secret values.
- `Metric`
  - Named health indicator when worth modeling explicitly.

Core relationships:

- `(:Branch)-[:DEPLOYED_AS]->(:Deployment)`
- `(:Deployment)-[:TARGETS]->(:Environment)`
- `(:Environment)-[:HOSTED_ON]->(:DeploymentTarget)`
- `(:Service)-[:USES_DEPLOYMENT_STRATEGY]->(:DeploymentStrategy)`
- `(:Alert)-[:FIRED_IN]->(:Environment)`
- `(:Alert)-[:INDICATES]->(:Incident)`
- `(:Runbook)-[:MITIGATES]->(:Incident)`
- `(:Incident)-[:IMPACTS]->(:Service)`
- `(:Script)-[:RUNS]->(:Service|:Component)`
- `(:ConfigVariable)-[:CONFIGURES]->(:Service|:Environment)`

### 3a. Debugging and reliability memory

These nodes let the graph remember how issues were investigated and fixed across users and agents.

- `BugPattern`
  - Repeated failure mode, symptom cluster, or known class of issue.
- `Investigation`
  - Debugging session, diagnostic path, or incident investigation.
- `Fix`
  - Resolution, mitigation, workaround, or permanent code/config change.
- `DiagnosticSignal`
  - Error message, stack trace signature, metric, log query, alert fingerprint, or symptom.

Core relationships:

- `(:Incident)-[:MATCHES_PATTERN]->(:BugPattern)`
- `(:Investigation)-[:DEBUGGED]->(:Incident|:BugPattern)`
- `(:DiagnosticSignal)-[:OBSERVED_IN]->(:Investigation|:Incident)`
- `(:Fix)-[:RESOLVED]->(:Incident|:BugPattern)`
- `(:Fix)-[:CHANGED_BY]->(:PullRequest|:Commit)`
- `(:BugPattern)-[:SEEN_IN]->(:Service|:Environment|:Component)`

This is the direct support for the workflow where one user debugs a problem, the fix is captured, and another user later retrieves that prior fix when similar symptoms appear.

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
- `AgentInstruction`
  - AGENTS.md, skill, prompt, MCP guidance, or other agent-facing instruction.
- `LocalWorkflow`
  - How people usually run, test, debug, or deploy locally.

Core relationships:

- `(:PullRequest)-[:PART_OF]->(:Change)`
- `(:Commit)-[:PART_OF]->(:PullRequest)`
- `(:PullRequest)-[:ADDRESSES]->(:Issue)`
- `(:Decision)-[:MADE_IN]->(:PullRequest|:Incident|:Document)`
- `(:Decision)-[:AFFECTS]->(:Feature|:Component|:Service|:CodeAsset)`
- `(:Constraint)-[:APPLIES_TO]->(:Service|:Component|:Feature|:Repository)`
- `(:Preference)-[:PREFERRED_FOR]->(:Repository|:Component|:Team)`
- `(:AgentInstruction)-[:INFORMS]->(:Repository|:Service|:Feature|:Agent)`
- `(:LocalWorkflow)-[:RUNS]->(:Service|:Component|:Repository)`

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
- `SourceSystem`
  - GitHub, Linear, Slack, Docs, Sentry, GCP, AWS, or another provider.
- `SourceReference`
  - Stable pointer to an external artifact with resolver hints and freshness metadata.

Core relationships:

- `(:Episode)-[:DESCRIBES]->(:Change|:Incident|:Decision|:Document)`
- `(:Document)-[:DESCRIBES]->(:Feature|:Component|:Constraint)`
- `(:Conversation)-[:RESULTED_IN]->(:Decision)`
- `(:Observation)-[:SUPPORTS]->(:Decision|:Incident|:Constraint)`
- `(:SourceReference)-[:FROM_SOURCE]->(:SourceSystem)`
- `(:Entity)-[:EVIDENCED_BY]->(:SourceReference)`

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
- `DeploymentTarget`
- `DeploymentStrategy`
- `Component`
- `Capability`
- `Feature`
- `Functionality`
- `Requirement`
- `RoadmapItem`
- `Interface`
- `DataStore`
- `Integration`
- `Dependency`
- `Person`
- `Team`
- `PullRequest`
- `Commit`
- `Issue`
- `Decision`
- `Constraint`
- `Preference`
- `AgentInstruction`
- `LocalWorkflow`
- `Document`
- `Conversation`
- `SourceSystem`
- `SourceReference`
- `Incident`
- `Alert`
- `BugPattern`
- `Investigation`
- `Fix`
- `DiagnosticSignal`
- `Metric`
- `Runbook`
- `Script`
- `ConfigVariable`
- `QualityIssue`
- `MaintenanceJob`
- `MaterializedAccessPath`
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
- `HOSTED_ON`
- `IMPLEMENTS`
- `HAS_FUNCTIONALITY`
- `DEFINES`
- `SUPPORTS`
- `EXPOSES`
- `USES`
- `DEPENDS_ON`
- `USES_DATA_STORE`
- `USES_DEPLOYMENT_STRATEGY`
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
- `INFORMS`
- `MITIGATES`
- `IMPACTS`
- `FIRED_IN`
- `INDICATES`
- `MATCHES_PATTERN`
- `DEBUGGED`
- `OBSERVED_IN`
- `RESOLVED`
- `CHANGED_BY`
- `SEEN_IN`
- `DESCRIBES`
- `RESULTED_IN`
- `SUPPORTS`
- `EVIDENCED_BY`
- `FROM_SOURCE`
- `FLAGS`
- `REPAIRS`
- `MATERIALIZES`
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

The graph should expose a stable public read contract for agents. The ontology alone is not enough. Agents need predictable entrypoints, consistent response shapes, explicit explanation payloads, and clear fallback behavior.

The public contract should stay small:

- `context_resolve`
- `context_search`
- `context_record`
- `context_status`

Everything else in this section describes internal query planning or request parameters behind those tools.

### Query principles

- `context_resolve` is the default starting point for non-trivial agent work
- canonical graph queries answer precise questions inside the resolver
- episodic retrieval supplies evidence, recall, and ambiguity resolution inside the resolver
- every high-value answer should include supporting facts or evidence
- partial coverage should be visible to the caller
- temporal perspective should be explicit through `as_of`
- source-backed reads and verification are controlled by `mode`, `source_policy`, and `budget`
- broad `context_search` is a follow-up tool, not the default entrypoint

### Internal query families

These query families are stable product capabilities, but they should not each become a separate agent tool.

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

The resolver should compose these families based on `intent`, `scope`, `include`, `mode`, and `source_policy`.

### Recommended response shape

Agent-facing tools should return a common envelope. `context_resolve` should provide the richest envelope; `context_search`, `context_record`, and `context_status` should preserve the same ideas where applicable.

Recommended fields:

- `answer`
- `facts`
- `evidence`
- `source_refs`
- `confidence`
- `as_of`
- `open_conflicts`
- `coverage`
- `freshness`
- `quality`
- `verification_state`
- `fallbacks`
- `recommended_next_actions`

Recommended behavior:

- `facts` should contain canonical graph results
- `evidence` should contain compact supporting facts, episodes, docs, conversations, observations, or source-backed summaries
- `source_refs` should point to source artifacts and resolver inputs
- `open_conflicts` should surface contested or contradictory support
- `coverage` should state what was searched, what was missing, and whether the answer is complete enough for the task
- `freshness` should indicate whether the answer is recent enough to trust directly
- `quality` should expose graph health metrics, drift issues, source-sync gaps, freshness policy, and recommended maintenance jobs
- `verification_state` should indicate whether the fact was checked against an external source recently
- `fallbacks` should tell the caller when context is missing, stale, unsupported, unreachable, or permission-denied
- `recommended_next_actions` should suggest source verification, narrower scope, deeper resolve mode, or graph alignment only when useful

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

### Tool responsibilities

#### `context_resolve`

Primary read tool. It should:

- resolve pot, repo, feature, service, file, PR, ticket, incident, environment, user, and source-reference scope
- plan internal query families from `intent` and `include`
- combine canonical graph facts, Graphiti recall, source references, and optional source-backed summaries or verification
- return compact context wraps with coverage, freshness, fallbacks, conflicts, and recommended next actions
- enforce `budget` and return partial context instead of blocking the agent for too long

#### `context_search`

Secondary lookup tool. It should:

- support narrow follow-up searches over graph memory and evidence
- accept filters by labels/types, repo, feature, service, source type, time, freshness, and status
- return compact results with source refs and confidence
- avoid becoming the default task-context tool

#### `context_record`

Generic write tool. It should:

- record durable learnings and corrections from agent work
- accept `record.type` values such as `decision`, `fix`, `preference`, `workflow`, `feature_note`, `service_note`, `runbook_note`, `integration_note`, `incident_summary`, and `doc_reference`
- require scope, summary, source refs, confidence, visibility, and idempotency key when possible
- route writes through the same reconciliation and validation pipeline as other source events

#### `context_status`

Lightweight status tool. It should:

- report pot and scope readiness
- summarize connected sources, last ingestion, last verification, freshness, known gaps, and permission issues
- return the stable context port manifest and the recommended `context_resolve` recipe when an intent is provided
- be cheap enough for agents to call before large tasks or when `context_resolve` reports poor coverage

### Non-goals for the public tool surface

Do not add separate public tools for every query family.

Avoid public tools like:

- `context_get_feature_context`
- `context_get_service_context`
- `context_get_debugging_context`
- `context_get_operational_context`
- `context_get_preferences`
- `context_get_source`

Those are recipes or internal resolver paths. They should be expressed through `context_resolve` parameters and skills.

Current implementation note: Phase 6 adds a code-level context port manifest and recipe catalog. `context_status` returns the manifest plus the recommended recipe for optional intent values, MCP descriptions steer agents toward the four-tool port, and the generated `context-engine-agent-context` skill documents the operating loop and presets.

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
- if graph coverage is poor, the agent should request source-backed `context_resolve` output or use available source integrations without pretending graph certainty

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

Current implementation note: Phase 7 adds `domain/graph_quality.py` as the first code-level policy for these metrics. `context_resolve` and `context_status` return a `quality` report with metrics, issues, source-of-truth policy, freshness TTL policy, and recommended maintenance jobs. The ontology now includes `QualityIssue`, `MaintenanceJob`, and `MaterializedAccessPath` nodes plus `FLAGS`, `REPAIRS`, and `MATERIALIZES` edges so future jobs can write durable quality state into the graph.

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

## Current implementation review

This section reviews the current `app/src/context-engine` implementation against the target architecture above.

### Requirement coverage against the product ask

| Product requirement | Coverage in this spec | Current implementation state |
| --- | --- | --- |
| Pot as end-to-end project context | Covered by `Pot`, pot-scoped graph operations, and context wraps | Pot scoping exists for Graphiti, HTTP, CLI, MCP, and tenancy |
| Add repos, users, features, docs, integrations, deployments, preferences | Covered by ontology categories and source-reference policy | Phase 4 ontology and `project_map_context` reads exist; ingestion coverage still needs expansion |
| Agent has purpose and knows where work sits | Covered by feature/service/component mappings and context wraps | `context_resolve` now returns `project_map` facts when backed by canonical data |
| Feature/functionality mapped across frontend, backend, services, docs | Covered by `Feature`, `Functionality`, `Service`, `Component`, `Document`, `Integration` | Phase 4 read model exists; durable cross-repo population remains incremental |
| Deployment and local operating context | Covered by `Environment`, `DeploymentTarget`, `DeploymentStrategy`, `Script`, `Runbook`, `ConfigVariable`, `LocalWorkflow` | Phase 4 read model exists through `operations` and related includes; source ingestion remains incremental |
| Debugging memory and prior fixes | Covered by `BugPattern`, `Investigation`, `Fix`, `DiagnosticSignal`, incidents, and capture-fix workflow | Phase 5 read model exists through `debugging_memory`; ingestion population remains incremental |
| Extensible future use cases | Covered by ontology governance, source normalization, `context_resolve` recipes, and phased rollout | Ontology validation and minimal-port planning are implemented; future use cases should add includes/providers, not new public tools |
| Use Graphiti temporal/update features but build on top | Covered by Graphiti substrate + Potpie canonical ontology split | Current implementation already uses Graphiti and deterministic structural writes |
| Avoid huge graph / avoid source duplication | Covered by source-reference-first storage and resolver contract | Implemented for source refs and Phase 4 project-map reads; full source payloads stay external |
| Good fallbacks for missing/stale data | Covered by coverage gaps, freshness, verification, resolver status, and drift management | Phase 7 quality report adds freshness/source-sync metrics, issues, and maintenance recommendations |
| Natural agent injection via skills, AGENTS.md, MCP | Covered by the minimal context port and recipe-based skills | Phase 6 recipe manifest, generated `AGENTS.md`, MCP descriptions, and `context-engine-agent-context` skill are implemented |

### What is already aligned

The implementation is directionally sound and has the right skeleton for an extensible context platform:

- package boundaries are hexagonal: `domain`, `application`, `adapters`, `bootstrap`
- pot scoping exists through `pot_id`, Graphiti `group_id`, API tenancy checks, CLI pot resolution, and MCP pot allowlisting
- event-first ingestion is emerging through `ContextEvent`, ingestion kinds, event stores, reconciliation ledgers, idempotency, and async step status
- Graphiti is used in the right role for episodic ingest, custom entity extraction, semantic retrieval, invalidation filtering, and temporal `as_of` reads
- deterministic structural writes already stamp PR, commit, developer, decision, file, and code bridge context instead of trusting extraction alone
- agent-facing `resolve-context` already returns normalized evidence families, coverage, metadata, and recoverable errors
- queueing is abstracted behind Celery/Hatchet/noop adapters
- HTTP, CLI, and MCP are all present as integration surfaces

The key strength is that the implementation is already a platform-shaped system, not a script. The missing work is mostly richer source resolvers, ingestion breadth, scheduled quality operations, and source-specific repair adapters.

### Important gaps

#### Canonical ontology breadth

The current canonical schema is mostly GitHub/code-history oriented. It understands PRs, commits, issues, decisions, developers, features, file owners, review context, and semantic search.

The target needs first-class concepts for:

- repositories, users, teams, services, components, and systems
- functionality, requirements, roadmap, docs, and integrations
- environments, deployment targets, strategies, scripts, runbooks, and config references
- incidents, alerts, bug patterns, investigations, fixes, and diagnostic signals
- preferences, conventions, local workflows, and agent instructions
- source systems and source references

Without these concepts, new use cases will overload generic episodes or existing PR/Decision/Feature nodes, making deterministic agent queries weak.

#### Source references and resolvers

The graph should store references plus compact summaries, then fetch full payloads through source integrations when needed. Today, source identity appears in several places, but there is no unified `SourceReference` model or `SourceResolverPort`.

This makes it hard to answer with precise fallback states such as:

- no graph fact exists
- graph fact exists but source could not be reached
- graph fact exists but source access was denied
- graph fact exists but is stale
- graph fact was verified recently
- source truth disagrees with graph memory

#### Freshness and verification

Graphiti supports temporal semantics and the intelligence bundle has coverage/errors, but canonical facts do not yet consistently expose:

- `last_seen_at`
- `last_verified_at`
- source last modified time
- freshness policy
- stale/superseded/invalidated status
- confidence
- resolver verification result

Freshness is product behavior, not only metadata. Agents need to know when to trust graph memory and when to verify against the source.

#### Resolver planning families

Current query kinds are valuable but narrow:

- semantic search
- change history
- file owners
- decisions
- PR review context
- PR diff
- project graph
- resolve context

The target needs high-level internal resolver families for:

- feature context
- service context
- operational and deployment context
- debugging and prior-fix lookup
- docs and integration context
- preferences, conventions, and local workflows
- user/team context
- source/freshness verification

These should be context-wrap sections and internal resolver paths behind `context_resolve`, not separate public agent tools or ad hoc raw graph queries.

#### Debugging memory

Raw episodes and PR history can remember some fixes, but debugging memory needs direct modeling:

- symptom signatures
- diagnostic steps
- alerts, log queries, stack traces, and error messages
- investigation timelines
- mitigations and confirmed fixes
- affected service/environment/component
- recurrence patterns

This is now a first-class read domain through `debugging_memory_context`. The remaining work is richer ingestion and reconciliation so agent sessions, alert systems, incidents, PRs, and runbooks consistently populate these canonical records.

#### Preferences and instructions

Preferences can be personal, team-wide, project-wide, stale, conflicting, or low-confidence. The graph should distinguish:

- hard constraints
- soft preferences
- conventions
- user preferences
- team preferences
- local workflows
- agent instructions
- temporary observations

Each should carry scope, owner, source, confidence, and freshness.

## Recommended next implementation steps

### Phase A: Ontology foundation

- Add a code-level ontology catalog for canonical labels, edges, key formats, required fields, metadata, and allowed edge pairs.
- Add schema-aware validation for `ReconciliationPlan` before graph writes.
- Add common metadata requirements for confidence, status, source refs, validity, last seen, last verified, and schema version.
- Add unit tests under `app/src/context-engine/tests/unit/`.

### Phase B: Source references and resolvers

- Define `SourceReference` and freshness models.
- Define `SourceResolverPort`.
- Implement GitHub resolver first for PRs, issues, commits, review discussions, and URLs already present in the graph.
- Return resolver status in context intelligence bundles.
- Add fallback reasons such as `not_ingested`, `empty_result`, `source_unreachable`, `permission_denied`, `stale`, and `not_supported`.

### Phase C: Minimal context port and agent operating workflows

Implementation status: first pass implemented. The MCP agent surface now exposes the four-tool port, `context_resolve` returns the common envelope with budget/as-of support, `context_record` routes generic learnings through reconciliation, and `context_status` reports cheap pot readiness plus a recommended recipe. Generated `AGENTS.md`, CLI docs, MCP descriptions, and the `context-engine-agent-context` skill now guide agents to use parameter presets over `context_resolve`. Dedicated feature/service/operations/debugging evidence families remain Phase D/E expansion work behind the same port.

- Evolve `context_resolve` into the primary context-wrap orchestrator with `intent`, `scope`, `include`, `exclude`, `mode`, `source_policy`, `budget`, and `as_of`.
- Extend the intelligence bundle with feature, service, operations, debugging, docs, preferences, and source-status evidence families.
- Add or update MCP/HTTP support for the stable minimal public surface:
  - `context_resolve`
  - `context_search`
  - `context_record`
  - `context_status`
- Fold source fetching and verification into `context_resolve` through `source_policy`; do not add a separate public `context_source` tool.
- Keep generated `AGENTS.md` and skills aligned so feature, debugging, review, operations, docs, and onboarding workflows remain parameter recipes over `context_resolve`.
- Keep any source-specific or use-case-specific helpers internal to the resolver/application layer.

### Phase D: Debugging memory

- First read-model pass implemented: `context_resolve` can return `debugging_memory` records for `Fix`, `BugPattern`, `Investigation`, `DiagnosticSignal`, `Incident`, and `Alert` when canonical data exists.
- `context_record` accepts debugging-oriented records such as `fix`, `bug_pattern`, `investigation`, `diagnostic_signal`, and `incident_summary`.
- Next ingestion work should turn agent debugging session summaries into canonical symptoms, root causes, fixes, changed files/PRs, affected service/environment, and source refs.
- Add richer similarity lookup over symptoms plus deterministic filters by service, environment, file path, and error signature.

### Phase E: Feature, service, and operations map

- First read-model pass implemented: `context_resolve` can return `project_map` records for first-class `Service`, `Component`, `Functionality`, `Requirement`, `Integration`, `Document`, `Environment`, `DeploymentTarget`, `DeploymentStrategy`, `Script`, `Runbook`, `ConfigVariable`, `Preference`, `AgentInstruction`, and `LocalWorkflow` support when canonical data exists.
- Link existing PR/Decision/File history into those entities.
- Ingest compact references from README, AGENTS.md, package scripts, compose files, CI workflows, deployment docs, and runbooks.
- Avoid copying full docs or logs into the graph by default.

### Phase F: Quality, drift, and scale

- First policy/readiness pass implemented: `context_resolve` and `context_status` return `quality` reports with source-ref metrics, stale and unverified counts, access gaps, coverage gaps, freshness TTL policy, source-of-truth policy, and recommended maintenance jobs.
- Ontology support exists for `QualityIssue`, `MaintenanceJob`, and `MaterializedAccessPath` plus `FLAGS`, `REPAIRS`, and `MATERIALIZES` relationships.
- Next production work should schedule recurring verification and cleanup jobs, then connect source-specific adapters for alias repair, orphan cleanup, code bridge repair, retention, and materialized access path refresh.

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
