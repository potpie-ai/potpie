# Context Graph Features And Functionalities

## Purpose

This document is the product-facing feature and API reference for Potpie's context graph. It describes what the context graph does, which user workflows it supports, and which HTTP, CLI, and MCP surfaces should be treated as stable integration contracts.

Implementation architecture details live in [`graph.md`](graph.md). The implementation migration plan for the one-port Graphiti application layer lives in [`unified-graphiti-application-architecture.md`](unified-graphiti-application-architecture.md). The refactor plan for tightening the current API and product surface lives in [`planning-next-steps.md`](planning-next-steps.md).

## Product Contract

The context graph is Potpie's project memory layer. It combines:

- Episodic memory: raw events, notes, PR episodes, decisions, prior fixes, incidents, preferences, and other time-aware facts.
- Structural project context: repositories, files, symbols, services, features, ownership signals, PR changes, and code relationships.
- Source and provenance metadata: source refs, event IDs, episode UUIDs, timestamps, freshness, quality, and confidence.
- Agent-ready context bundles: bounded answers and evidence for feature work, debugging, code review, operations, docs, and onboarding.

The graph is backed by Graphiti and Neo4j. Public consumers should treat it as one context graph, not as separate episodic and structural systems.

The desired public contract is:

- One graph read API: `POST /api/v2/context/query/context-graph`.
- One query request model: `ContextGraphQuery`.
- One graph result envelope: `ContextGraphResult`.
- One small agent tool port: `context_resolve`, `context_search`, `context_record`, and `context_status`.
- Source-reference-first storage: graph reads return compact facts, relationships, source refs, freshness, and fallbacks; full source payloads are fetched through source resolvers only when requested and authorized.
- New use cases should add `intent`, `include`, `scope`, provider, or resolver support, not new public agent tools.
- Compatibility routes and provider-specific graph write branches are temporary
  migration aids, not product architecture. New implementation should target
  source events, generic reconciliation plans, and the unified graph query/write
  layer.

## Surface Boundaries

The current implementation exposes several kinds of endpoints under `/api/v1/context` and `/api/v2/context`. They should not all be presented as the same kind of product API.

### 1. Agent And API-Client Surface

This is the stable surface for agents, IDEs, automation clients, and future SDKs.

| Capability | Method And Path | Stability |
| --- | --- | --- |
| Unified graph read | `POST /api/v2/context/query/context-graph` | Stable product contract |
| Record durable memory | `POST /api/v2/context/record` | Stable product contract |
| Readiness and trust status | `POST /api/v2/context/status` | Stable product contract, needs deeper implementation |
| Raw note ingest | `POST /api/v2/context/ingest` | Stable simple-write contract |

MCP exposes these capabilities through the four-tool port:

- `context_resolve`: primary task context wrap.
- `context_search`: narrow follow-up semantic lookup after `context_resolve`.
- `context_record`: durable project memory write.
- `context_status`: cheap pot readiness, graph quality, freshness, and recommended recipe.

Agents should use these tools instead of one-off tools for feature context, debugging context, source lookup, operations context, or docs lookup.

### 2. UI And Application Surface

This surface supports the Potpie product UI and app workflows. It is public to the Potpie application, but it is not the core agent contract.

| Capability | Method And Path | Notes |
| --- | --- | --- |
| List/create/update pots | `/api/v2/context/pots*` | Pot tenancy and project boundary |
| Manage members and invitations | `/api/v2/context/pots/{pot_id}/members*`, `/invitations*` | Application authorization workflow |
| Manage sources (preferred) | `/api/v2/context/pots/{pot_id}/sources*` | Source-first attachment path; covers every supported source kind |
| Attach GitHub repo source | `POST /api/v2/context/pots/{pot_id}/sources/github/repository` | Creates the source row and mirrors a repository routing row for code-graph behavior |
| Attach Linear team source | `POST /api/v2/context/pots/{pot_id}/sources/linear/team` | Example non-code source |
| Patch source sync | `PATCH /api/v2/context/pots/{pot_id}/sources/{source_id}` | Toggle `sync_enabled` |
| Detach source | `DELETE /api/v2/context/pots/{pot_id}/sources/{source_id}` | For repository sources, also removes the mirrored repository routing row |
| Transitional repository routes | `/api/v2/context/pots/{pot_id}/repositories*` | Temporary GitHub/code-graph routing surface while clients move to source-first APIs |
| UI raw ingest | `POST /api/v2/context/pots/{pot_id}/ingest/raw` | Pot-scoped note/link submission for the app |
| Inspect events | `GET /api/v2/context/events/{event_id}`, `GET /api/v2/context/pots/{pot_id}/events` | UI and CLI event inspection |

**Source-first model.** New UI and integration flows should attach data via `/pots/{pot_id}/sources/*`. Repository routes are transitional for existing GitHub/code-graph flows; creating a repository there mirrors a row into the source table automatically, and deleting either side keeps the two consistent while the migration is active. Future source kinds (docs, Slack, incidents, deployments, alerts) only need source rows — they must not synthesize repository-shaped records.

### 3. Ingestion And Automation Surface

This surface is used by webhooks, workers, scheduled jobs, and trusted automation.

| Capability | Method And Path | Notes |
| --- | --- | --- |
| Repository/code backfill | `POST /api/v2/context/sync` | Enqueues configured context-graph job queue in Potpie |
| Merged PR ingest | `POST /api/v2/context/ingest-pr` | Also driven by GitHub webhook events when mapped to a pot |
| Normalized event reconcile | `POST /api/v2/context/events/reconcile` | Canonical event submission path |
| Compatibility event alias | `POST /api/v2/context/events/ingest` | Deprecated alias; hidden from OpenAPI, emits `Deprecation`/`Warning`/`Link` headers, counted + logged for migration tracking |
| Replay event | `POST /api/v2/context/events/replay` | Operational retry/debug path |

These endpoints are necessary, but they should be described as ingestion workflow APIs rather than as the primary agent-facing feature surface.

### 4. Operator And Admin Surface

This surface is for graph repair, reset, and maintenance. It is protected,
documented separately, and treated as an operational escape hatch. Every
route is grouped under the OpenAPI tag **`context:operator`** and carries an
``[operator]`` prefix in its summary so clients and generated SDKs see the
admin boundary clearly.

| Capability | Method And Path | Notes |
| --- | --- | --- |
| Reset pot graph | `POST /api/v2/context/reset` | Destructive; clears graph and optionally ledger rows. No dry-run. Audited. |
| List conflicts | `POST /api/v2/context/conflicts/list` | Quality/repair read. Pair with `/conflicts/resolve`. |
| Resolve conflicts | `POST /api/v2/context/conflicts/resolve` | Mutates graph state. Audited with actor, pot, issue uuid, and action. |
| Edge classification maintenance | `POST /api/v2/context/maintenance/classify-modified-edges` | Dry-run by default (`dry_run=true`). Writes require BOTH `CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES=1` and `CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE=1`. Audited. |

**Audit trail.** Each destructive or graph-mutating call emits a structured
log record via the `context_engine.operator_audit` logger with fields:
`action`, `pot_id`, `actor` (email/id where available), `dry_run`, and
action-specific details (`skip_ledger`, `issue_uuid`, `conflict_action`,
`outcome`, `error`). Operators can tail this logger or route it to a
durable audit store.

These APIs are not the everyday context graph product contract.

## Public Entrypoints

### HTTP API

Potpie mounts the context graph API at:

- `/api/v1/context` with Potpie application authentication.
- `/api/v2/context` with API-key authentication.

CLI and MCP clients use the v2 API.

### CLI

The `potpie` CLI is the command-line entrypoint for local developer and agent setup workflows.

Common commands:

- `potpie login`: store API credentials.
- `potpie doctor`: validate configuration and API readiness.
- `potpie pot use`: select an active context pot.
- `potpie pot create`: create a server-side context pot and store a local alias.
- `potpie pot repo add`: transitional GitHub repository attachment for CLI users. The server mirrors the repository into the pot's source list automatically; non-GitHub source kinds should be attached via `/pots/{pot_id}/sources/*`.
- `potpie add`: inspect the current git remote and print provider-scoped repo identity; this does not ingest content.
- `potpie ingest`: submit raw context episodes.
- `potpie search`: run semantic graph search.
- `potpie event show`, `potpie event list`, `potpie event wait`: inspect ingestion and reconciliation events.
- `potpie conflict list`, `potpie conflict resolve`: inspect and resolve graph conflicts.
- `potpie pot hard-reset`: **destructive** operator command that deletes all context-graph data for a pot. Prompts for confirmation unless `--yes`/`-y` is passed. Server-side logs a `context_engine.operator_audit` entry.

### MCP

The `potpie-mcp` server exposes only the minimal agent-facing tool surface:

- `context_resolve`
- `context_search`
- `context_record`
- `context_status`

Feature, debugging, review, operations, docs, onboarding, planning, refactor, test, and security workflows should be expressed as `context_resolve` recipes through `intent`, `include`, `scope`, `mode`, `source_policy`, and `budget`.

## End-To-End Features

### 1. Context Pot Management

A context pot is the tenant, project, and reasoning boundary for graph data. Potpie provides flows for:

- Creating and listing pots.
- Selecting an active pot.
- Managing pot members and invitations.
- Managing sources such as GitHub repositories and Linear teams.
- Mapping Git repositories to pots for CLI and agent workflows.

Pot management is implemented by the Potpie host application. The portable context-engine package consumes pot IDs and access checks but does not own every host CRUD behavior.

### 2. Source Management

Sources describe what external systems belong to a pot. A source can be a GitHub repository, Linear team, documentation source, Slack channel, incident system, deployment system, or future integration.

The preferred extensible model is:

- `ContextGraphPotSource` stores source type, provider, scope, sync settings, and source state.
- Repository rows remain available only for GitHub/code-graph routing during the source-first migration.
- Source resolvers should later use source rows to fetch summaries, verification signals, and bounded snippets for `context_resolve`.

### 3. Repository And Codebase Ingestion

Repository ingestion builds structural project context for a pot.

Supported data includes:

- Repositories.
- Files.
- Code symbols and functions.
- Project components and services.
- Code graph relationships.
- Ownership and change signals when available.

Primary automation API:

```http
POST /api/v2/context/sync
```

In the Potpie host, long-running ingestion is submitted to the configured context graph job queue.

### 4. Pull Request Ingestion

PR ingestion records merged PR context and connects it to the project graph.

Supported graph data should include:

- Pull request source references.
- Changed files and symbols.
- Review context summaries and source refs.
- Decisions inferred from PR activity.
- Commit and source references.
- Links between PRs, decisions, and affected code.

Primary automation API:

```http
POST /api/v2/context/ingest-pr
```

Full PR diffs should not be treated as durable graph payloads. Store changed-file summaries, touched symbols, decisions, and source refs in the graph. Fetch full diffs through source-backed `context_resolve` modes only when needed.

### 5. Raw Context Ingestion

Raw ingestion records externally supplied memory into the graph.

Supported inputs include:

- Notes.
- Docs and doc references.
- Links.
- Incident summaries.
- Bug reports.
- Prior fixes.
- Decisions.
- Preferences.
- Workflow and runbook notes.
- Feature notes.
- User-provided context snippets.

Primary simple-write API:

```http
POST /api/v2/context/ingest
```

When durable storage is configured, raw ingestion persists an event first and applies it asynchronously by default. Callers can request synchronous application when needed.

### 6. Durable Event And Reconciliation Pipeline

The event pipeline turns source input into canonical graph facts.

Supported operations:

- Submit normalized events for reconciliation.
- Replay existing events.
- Inspect event status.
- Inspect reconciliation runs.
- Inspect episode apply steps.
- Track work events and failures.

Primary API:

```http
POST /api/v2/context/events/reconcile
```

Deprecated transitional alias:

```http
POST /api/v2/context/events/ingest
```

The alias is hidden from the OpenAPI schema and attaches deprecation
signalling on every response:

- ``Deprecation: true``
- ``Warning: 299 - "POST /events/ingest is a deprecated alias of /events/reconcile; migrate clients to /events/reconcile."``
- ``Link: </api/v2/context/events/reconcile>; rel="successor-version"``

Each alias call also increments a process-local counter
(``events_ingest_alias_call_count()``) and logs a WARNING so operators can
track migration. New clients and docs must target ``/events/reconcile``. The
alias should be removed after known clients migrate.

### 7. Unified Graph Querying

The primary graph read API is:

```http
POST /api/v2/context/query/context-graph
```

This endpoint supports semantic search, exact scoped reads, graph traversal, temporal reads, aggregate reads, and task-oriented context answers through one request model.

There should be no separate public graph read APIs for episodic search, structural search, feature context, debugging context, operations context, source lookup, PR context, ownership, decisions, or context resolution. Those behaviors are expressed through query arguments and `context_resolve` recipes.

### 8. Agent Memory Recording

Applications and agents can record durable project memory without knowing the internal graph schema.

Primary API:

```http
POST /api/v2/context/record
```

Supported record types include:

- Decisions.
- Fixes.
- Bug patterns.
- Investigations.
- Diagnostic signals.
- Incident summaries.
- Preferences.
- Workflow notes.
- Feature notes.
- Service notes.
- Runbook notes.
- Integration notes.
- Documentation references.

Recorded memory is submitted through the event and reconciliation pipeline so it can become canonical graph state.

### 9. Status, Quality, And Readiness

Context graph consumers can inspect whether a pot is ready and whether graph memory is safe to rely on.

Primary API:

```http
POST /api/v2/context/status
```

Status responses should include:

- Pot readiness.
- Attached sources and repositories.
- Last successful ingestion per pot/source.
- Queued and failed event counts.
- Coverage.
- Freshness.
- Quality status.
- Known gaps.
- Open conflicts.
- Source sync state.
- Resolver capability matrix.
- Recommended maintenance.
- Recommended `context_resolve` recipe for an intent.

Current implementation returns a first-pass readiness and quality envelope. The desired outcome requires deeper wiring to source rows, event ledgers, source resolver capabilities, and last verification state.

### 10. Conflict Management

Conflict APIs expose graph conflicts that need review or resolution.

Primary operator APIs:

```http
POST /api/v2/context/conflicts/list
POST /api/v2/context/conflicts/resolve
```

Typical conflicts include competing facts, stale facts, ambiguous entity families, and unresolved reconciliation output. Conflict resolution should be auditable and should not be the default agent workflow.

### 11. Maintenance

Maintenance APIs support graph hygiene and quality workflows.

Primary operator API:

```http
POST /api/v2/context/maintenance/classify-modified-edges
```

This can be used in dry-run mode to inspect edge classification behavior before applying changes. Write mode should remain protected by server configuration and permissions.

## Unified Query Model

All graph reads use a `ContextGraphQuery` request.

### Request Fields

| Field | Purpose |
| --- | --- |
| `pot_id` | Required pot boundary. |
| `query` | Natural-language query for semantic, fuzzy, or answer-style requests. |
| `goal` | What the caller wants back. |
| `strategy` | How the graph should retrieve data. |
| `include` | Data families to include. |
| `exclude` | Data families to exclude. |
| `scope` | Repo, file, symbol, PR, feature, service, ticket, environment, source ref, or user filters. |
| `node_labels` | Restrict semantic search to selected graph labels. |
| `source_descriptions` | Restrict search to selected source descriptions. |
| `episode_uuids` | Restrict search to selected episodes. |
| `as_of` | Read graph state as of a point in time where supported. |
| `include_invalidated` | Include invalidated or superseded facts when supported. |
| `limit` | Maximum result count for retrieve/timeline/aggregate queries. |
| `consumer_hint` | Caller type or response-shaping hint. |
| `intent` | Task intent such as feature, debugging, review, operations, docs, onboarding, planning, refactor, test, or security. |
| `source_policy` | Source behavior such as `references_only`, `summary`, `snippets`, or `verify`. |
| `artifact` | Optional artifact being reviewed or explained. |
| `budget` | Limits for tokens, items, timeout, freshness preference, or response detail. |

### Source Policy Maturity

`source_policy` is the right long-term extension point, but implementations can mature over time.

| Policy | Desired Behavior | Current/Interim Behavior |
| --- | --- | --- |
| `references_only` | Return compact graph facts, source refs, freshness, and fallbacks. | Implemented as the baseline. |
| `summary` | Fetch bounded source-backed summaries through authorized resolvers. | May return fallback when resolver is unavailable. |
| `verify` | Check selected facts against source-of-truth systems and update verification state. | May return fallback and recommended verification action until resolvers are wired. |
| `snippets` | Fetch bounded source snippets within budget and permissions. | Should be opt-in and source-reference scoped. |

Do not add a separate public `context_source` tool. Source-backed behavior belongs behind `context_resolve` and `ContextGraphQuery` fields.

### Goals

| Goal | Use When |
| --- | --- |
| `retrieve` | You want matching graph items or evidence. |
| `answer` | You want an agent-ready context bundle or synthesized answer. |
| `neighborhood` | You want nearby graph nodes and relationships. |
| `timeline` | You want change history or time-ordered context. |
| `aggregate` | You want summaries such as ownership or graph overview. |

### Strategies

| Strategy | Use When |
| --- | --- |
| `auto` | Let the graph choose the best retrieval path. |
| `semantic` | Use fuzzy semantic search over graph memory. |
| `exact` | Prefer deterministic scoped reads. |
| `hybrid` | Combine semantic and structured retrieval. |
| `traversal` | Traverse graph relationships from a scoped anchor. |
| `temporal` | Prefer time-aware reads. |

### Scope Fields

| Scope Field | Purpose |
| --- | --- |
| `repo_name` | Limit to a repository. |
| `branch` | Limit to a branch. |
| `file_path` | Limit to a file. |
| `function_name` | Limit to a function. |
| `symbol` | Limit to a symbol. |
| `pr_number` | Limit to a pull request. |
| `services` | Limit to services. |
| `features` | Limit to features. |
| `environment` | Limit to an environment. |
| `ticket_ids` | Limit to tickets or issue IDs. |
| `user` | Limit to a user-related context scope. |
| `source_refs` | Limit to known source references. |

## Queryable Data

### Semantic Memory

Use semantic retrieval for fuzzy search across project memory.

```json
{
  "pot_id": "pot_123",
  "goal": "retrieve",
  "strategy": "semantic",
  "query": "prior fixes for timeout failures",
  "scope": {
    "repo_name": "potpie"
  },
  "limit": 10
}
```

### Task-Oriented Context Resolution

Use answer-style queries when an agent or application needs a bounded context bundle for a task.

```json
{
  "pot_id": "pot_123",
  "goal": "answer",
  "strategy": "hybrid",
  "query": "what context do I need before changing CLI search?",
  "intent": "feature",
  "include": [
    "purpose",
    "service_map",
    "recent_changes",
    "decisions",
    "source_status"
  ],
  "source_policy": "references_only",
  "budget": {
    "max_items": 20
  }
}
```

### Decisions

Use `include: ["decisions"]` to retrieve decision records.

```json
{
  "pot_id": "pot_123",
  "goal": "retrieve",
  "strategy": "auto",
  "include": ["decisions"],
  "scope": {
    "repo_name": "potpie"
  },
  "limit": 20
}
```

### Pull Request Review Context

Use `intent: "review"` or `include: ["artifact", "discussions", "recent_changes", "decisions"]` with a PR scope.

```json
{
  "pot_id": "pot_123",
  "goal": "answer",
  "strategy": "hybrid",
  "intent": "review",
  "query": "review PR 42",
  "scope": {
    "repo_name": "potpie",
    "pr_number": 42
  },
  "source_policy": "summary"
}
```

### Pull Request Diff Detail

Full PR diffs are not a documented graph family. The graph stores changed files, touched symbols, PR/review summaries, and decisions; diff text is fetched on demand from GitHub through the source resolver layer. Request it by combining `artifact={kind:"pr", identifier:"<n>"}` with `source_policy="summary"` (compact PR summary) or `source_policy="snippets"` (bounded hunks). Both are clamped by `ResolverBudget` (`max_chars_per_item`, `max_total_chars`, `max_snippets_per_ref`).

```json
{
  "pot_id": "pot_123",
  "goal": "answer",
  "strategy": "hybrid",
  "intent": "review",
  "query": "summarize risky diff areas in PR 42",
  "scope": {
    "repo_name": "potpie",
    "pr_number": 42
  },
  "artifact": {
    "kind": "pr",
    "identifier": "42"
  },
  "source_policy": "snippets",
  "budget": {
    "max_items": 20,
    "max_tokens": 8000
  }
}
```

### File Ownership

Use aggregate queries with `include: ["owners"]` and a file path.

```json
{
  "pot_id": "pot_123",
  "goal": "aggregate",
  "strategy": "exact",
  "include": ["owners"],
  "scope": {
    "repo_name": "potpie",
    "file_path": "app/src/context-engine/adapters/inbound/cli/main.py"
  }
}
```

### Change History

Use timeline queries for time-ordered context.

```json
{
  "pot_id": "pot_123",
  "goal": "timeline",
  "strategy": "temporal",
  "scope": {
    "repo_name": "potpie",
    "file_path": "app/src/context-engine/domain/graph_query.py"
  },
  "limit": 25
}
```

### Project Graph Neighborhood

Use neighborhood queries for graph traversal around scoped entities.

```json
{
  "pot_id": "pot_123",
  "goal": "neighborhood",
  "strategy": "traversal",
  "scope": {
    "repo_name": "potpie",
    "symbol": "ContextGraphQuery"
  },
  "limit": 50
}
```

### Graph Overview

Use aggregate queries without a narrow owner request to get a graph overview.

```json
{
  "pot_id": "pot_123",
  "goal": "aggregate",
  "strategy": "auto",
  "scope": {
    "repo_name": "potpie"
  }
}
```

## Agent Recipes

Agents should prefer `context_resolve` with explicit intent, include, source policy, and budget. These recipes map to unified context graph queries internally.

### Feature Work

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

### Debugging

```json
{
  "intent": "debugging",
  "include": [
    "prior_fixes",
    "diagnostic_signals",
    "incidents",
    "alerts",
    "recent_changes",
    "config",
    "deployments",
    "owners",
    "source_status"
  ],
  "mode": "fast",
  "source_policy": "references_only"
}
```

### Review

```json
{
  "intent": "review",
  "include": [
    "artifact",
    "discussions",
    "owners",
    "recent_changes",
    "decisions",
    "preferences",
    "source_status"
  ],
  "mode": "balanced",
  "source_policy": "summary"
}
```

### Operations

```json
{
  "intent": "operations",
  "include": [
    "deployments",
    "runbooks",
    "alerts",
    "incidents",
    "scripts",
    "config",
    "owners",
    "source_status"
  ],
  "mode": "balanced",
  "source_policy": "summary"
}
```

## Response Expectations

Graph query responses use a `ContextGraphResult` shape.

Common fields:

- `kind`: response kind.
- `goal`: effective query goal.
- `strategy`: effective query strategy.
- `result`: returned data.
- `error`: error details when a query cannot be fulfilled.
- `meta`: source, routing, coverage, or adapter metadata.

Answer-style responses should include:

- `answer`.
- `facts`.
- `evidence`.
- `source_refs`.
- `coverage`.
- `freshness`.
- `quality`.
- `verification_state`.
- `fallbacks`.
- `open_conflicts`.
- `recommended_next_actions`.

Consumers should inspect coverage, freshness, quality, fallbacks, open conflicts, and source refs before relying on graph memory for high-impact changes.

## API Surface Summary

### Stable Agent/API-Client Contract

| Feature | Method And Path |
| --- | --- |
| Unified graph query | `POST /api/v2/context/query/context-graph` |
| Record memory | `POST /api/v2/context/record` |
| Status | `POST /api/v2/context/status` |
| Raw ingestion | `POST /api/v2/context/ingest` |

### UI/Application Contract

| Feature | Method And Path |
| --- | --- |
| Context pots | `/api/v2/context/pots*` |
| Pot members | `/api/v2/context/pots/{pot_id}/members*` |
| Pot invitations | `/api/v2/context/pots/{pot_id}/invitations*` |
| Pot sources | `/api/v2/context/pots/{pot_id}/sources*` |
| UI raw ingest | `POST /api/v2/context/pots/{pot_id}/ingest/raw` |
| Get event | `GET /api/v2/context/events/{event_id}` |
| List pot events | `GET /api/v2/context/pots/{pot_id}/events` |

### Ingestion/Automation Contract

| Feature | Method And Path |
| --- | --- |
| Repository sync | `POST /api/v2/context/sync` |
| PR ingestion | `POST /api/v2/context/ingest-pr` |
| Reconcile event | `POST /api/v2/context/events/reconcile` |
| Replay event | `POST /api/v2/context/events/replay` |

### Operator/Admin Contract

| Feature | Method And Path |
| --- | --- |
| List conflicts | `POST /api/v2/context/conflicts/list` |
| Resolve conflict | `POST /api/v2/context/conflicts/resolve` |
| Classify modified edges | `POST /api/v2/context/maintenance/classify-modified-edges` |
| Reset pot graph | `POST /api/v2/context/reset` |
