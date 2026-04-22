# Context Graph Implementation Next Steps

Last reviewed: 2026-04-22

This document is the code-facing migration plan after reviewing
`app/src/context-engine` against the desired architecture:

- one context graph application layer backed by Graphiti
- event-first ingestion for every source
- deterministic canonical graph mutations validated by Potpie ontology
- versatile bulk query planning for agents, CLI, MCP, HTTP, and UI consumers
- no backwards-compatibility branches as an accepted end state

Use [`graph.md`](graph.md) for product architecture and
[`unified-graphiti-application-architecture.md`](unified-graphiti-application-architecture.md)
for the target one-port Graphiti application model.

## Verified Current State

The assessment is directionally correct. The code has a strong foundation, but
it is still mid-migration.

What is real today:

- Domain models and ports are mostly framework-free.
- `ContextGraphQuery` and `ContextGraphPort` exist.
- `ContextGraphPort` owns the application-facing read/write surface:
  `query()`, `query_async()`, `apply_plan()`, and `write_raw_episode()`.
- `POST /api/v2/context/query/context-graph` routes through `ContextGraphPort.query_async()`.
- The MCP agent surface is the intended four-tool surface:
  `context_resolve`, `context_search`, `context_record`, and `context_status`.
- Raw ingest, event reconciliation, step splitting, reconciliation ledger rows,
  and queue-backed apply jobs exist.
- Ontology validation exists before generic canonical mutations are applied.
- Source references, freshness, quality envelopes, and resolver ports exist.
- Reconciliation execution, durable episode-step apply, event replay, raw ingest,
  and the submission service now write through the unified graph adapter.

What is not yet true:

- Some application use cases still import `EpisodicGraphPort` and
  `StructuralGraphPort` directly for adapter-owned ingestion/query helpers.
- `GraphitiContextGraphAdapter` is currently a facade over the old episodic and
  structural adapters, not the single graph layer.
- Query dispatch in `GraphitiContextGraphAdapter` now runs through
  `GraphQueryPlanner` (`application/services/graph_query_planner.py`) plus a
  per-family executor registry. A single declarative `ContextGraphQuery` can
  request multiple evidence families via `include=[...]`; the adapter returns
  one merged envelope with per-leg metadata and consistent fallbacks.
- `context_status` has useful first-pass readiness data, but not the full source
  sync, event ledger, resolver capability, verification, and conflict picture.
- Source resolvers exist for GitHub PR and documentation URI refs, but non-GitHub
  sources such as Linear, Slack, incidents, deployments, and alerts are not yet
  deep enough for the desired product.

## Target Design

Every context update follows one path:

1. Normalize incoming source input into `ContextEvent`.
2. Persist the event with pot scope, source identity, source event id,
   idempotency key, payload, occurred time, received time, and ingestion kind.
3. Run the Ingestion Agent.
4. Let the agent inspect current context and source data through bounded tools.
5. Produce a `ReconciliationPlan` containing episode drafts, canonical entity
   upserts, canonical edge upserts/deletes, invalidations, evidence refs,
   confidence, and warnings.
6. Validate the plan against ontology and source/provenance requirements.
7. Apply the plan through `ContextGraphPort.apply_plan()`.
8. Persist work events, tool calls, plan metadata, apply steps, failures, and
   result counts to the reconciliation ledger.

Every consumer read follows one path:

1. Build a `ContextGraphQuery` directly or through a recipe/preset.
2. Run it through `ContextGraphPort.query_async()`.
3. Let the query planner execute exact, traversal, temporal, semantic, and source
   resolver legs as needed.
4. Return one envelope with facts, evidence, source refs, provenance, freshness,
   fallbacks, quality, open conflicts, and recommended next actions.

## Priority Order

### Completed: Remove GitHub PR Compatibility From Graph Writes

Merged PR ingestion now builds generic ontology mutations through the
deterministic GitHub PR planner. `ReconciliationPlan` no longer carries a
provider-specific PR bundle, `apply_reconciliation_plan()` has no GitHub branch,
and `stamp_pr_entities()` has been removed from the structural graph port and
Neo4j adapter.

Follow-up work should broaden the generic PR plan to capture more source
details, but those additions must remain normal `EntityUpsert`, `EdgeUpsert`,
and invalidation operations.

### Completed: Collapse Core Execution Writes Behind The Unified Graph Port

The old `ContextGraphWriter` / `DefaultContextGraphWriter` wrapper has been
deleted. `GraphitiContextGraphAdapter` now implements the bounded write
operations used by event reconciliation and episode-step execution:

- `apply_plan()` for validated reconciliation plan slices
- `write_raw_episode()` for direct raw episode writes

The following flows now depend on `ContextGraphPort` for writes:

- `reconcile_event`
- `apply_episode_step_for_event`
- `record_and_reconcile_context_event`
- `replay_context_event`
- `record_raw_episode_ingestion`
- `run_raw_episode_ingestion`
- context graph background apply jobs
- direct merged-PR ingest
- merged-PR backfill
- hard reset

Remaining direction:

- Finish moving any remaining application and intelligence callers behind
  `ContextGraphPort.query()`.
- Replace the named-handler query dispatcher with a richer planner that can run
  multiple evidence legs and merge them for bulk agent context.
- Rename or retire bridge-specific ledger fields now that direct bridge writes
  have been removed from application flow.
- Leave low-level Graphiti and Neo4j helper classes as adapter internals.

Acceptance criteria:

- Execution-owned ingestion writes no longer import `EpisodicGraphPort` or
  `StructuralGraphPort`.
- `domain/ports/episodic_graph.py` and `domain/ports/structural_graph.py` are
  adapter-internal or deleted.
- Hard reset and read-helper use cases go through the same graph layer.

### Completed: Keep Sync Query Safe In Async Contexts

`GraphitiContextGraphAdapter.query()` now raises `RuntimeError` for
answer queries when an event loop is already running; callers in async
contexts must use `query_async()`. Covered by
`test_context_graph_adapter_sync_answer_query_rejects_running_loop` in
`tests/unit/test_context_graph_query.py`.

### Completed: Replace Adapter If/Else Dispatch With Query Planning

`GraphitiContextGraphAdapter` no longer contains a named-handler chain.
`GraphQueryPlanner` compiles a `ContextGraphQuery` into an
`ExecutionPlan` of typed `QueryLeg` entries (exact, temporal, semantic,
hybrid, traversal, answer), and the adapter runs each leg through a
per-family executor registry. Results are merged deterministically:
single-family requests preserve the legacy `kind`/`result` envelope, and
multi-family requests return `kind="multi"` with a family→payload map
plus `meta.legs` and `meta.fallbacks`.

`pr_diff` is kept as a compat-only leg (`meta.compat=true`) since full
diffs belong behind source resolvers per the planning-next-steps phase 5
direction.

Follow-up work:

- Extend plans with budget allocation per leg and cross-leg provenance
  merging once additional families require it.
- Replace remaining direct callers of `query_context` helpers with
  `ContextGraphQuery` presets or inline `ContextGraphQuery` bodies.

### P1: Make The Ingestion Agent Context-Aware

Current code:

- The reconciliation agent receives a `ReconciliationRequest`.
- Agent work events can be recorded.
- The agent has limited structured access to existing graph context.

Required direction:

- Give the Ingestion Agent a bounded tool set for current context lookup,
  source fetch/verify, entity lookup, conflict lookup, and prior event lookup.
- Require the agent to emit evidence refs and confidence for every important
  entity, edge, invalidation, and summary.
- Validate that every mutation carries provenance and event/source time fields.
- Persist agent reasoning, tool calls, tool results, warnings, and errors in the
  reconciliation run.

Acceptance criteria:

- Reconciliation plans explain why a new event changes, confirms, invalidates,
  or conflicts with existing graph facts.
- Consumers can inspect where a fact came from and when it was observed,
  written, and last verified.

### Completed: Deepen Provenance As A First-Class Contract

`ProvenanceRef` (domain/graph_mutations.py) now carries all 13 required
fields: `pot_id`, `source_event_id`, `episode_uuid`, `source_system`,
`source_kind`, `source_ref`, `event_occurred_at`, `event_received_at`,
`graph_updated_at`, `valid_from`, `valid_to`, `confidence`,
`created_by_agent`, `reconciliation_run_id`. A new `ProvenanceContext`
carries the subset the plan cannot reconstruct from `event_ref`
(source_kind, source_ref, event times, agent identity, run id), and
`ContextGraphPort.apply_plan` accepts it.

`apply_reconciliation_plan` builds the full `ProvenanceRef` for every
mutation; callers that thread context today: `reconcile_event`,
`apply_episode_step_for_event`, and `ingest_merged_pr`.

The Neo4j mutation applier now stamps every provenance field as
`prov_*` properties on entity upserts, edge upserts, and invalidations.
`delete_edges` records `prov_deleted_by` / `prov_deleted_at` before
deleting so the audit is preserved in any edge-history side store.

`_search_result_row` extracts `prov_*` attributes from Graphiti edges
into a compact `provenance` dict on each evidence row. `FreshnessReport`
now distinguishes `last_graph_update` vs `last_source_event_at` vs
`last_source_verification`, so freshness no longer conflates the three
clock lines.

Contract tests in `tests/unit/test_provenance_contract.py` pin the
behavior end-to-end.

Follow-ups landed 2026-04-22:

- `adapters/outbound/graphiti/apply_episode_provenance.py` runs after
  `g.add_episode()` and stamps `prov_*` on the Episodic node and every
  extracted entity edge whose ``episodes`` array contains the new uuid.
  Wired through `add_episode` / `add_episode_async` / `write_episode_drafts`
  via an optional ``provenance`` arg on `EpisodicGraphPort`.
- `assess_freshness` accepts optional evidence rows and now populates
  `last_source_event_at` from `provenance.event_occurred_at` across
  semantic hits. `context_resolve` passes `semantic_hits` through; the
  `/status` route includes `last_source_event_at` in its freshness block.
- `_search_result_row` returns a nested `provenance` dict on every row
  (flat `prov_*` stays internal to Neo4j). The HTTP route serializes
  `ContextGraphResult.model_dump()` without flattening — pinned by
  `test_context_graph_result_model_dump_keeps_provenance_nested`.

### P2: Deepen `context_status`

Current code returns first-pass readiness, quality, manifest, recipes, and some
source/resolver data.

Required direction:

- Include attached source rows and sync state.
- Include event ledger counts by status and recent failures.
- Include reconciliation ledger health and stuck/failed apply steps.
- Include resolver capability matrix by provider/source kind.
- Include last successful ingestion and last verification per source.
- Include open conflicts and recommended maintenance jobs.

Acceptance criteria:

- An agent can decide whether to rely on graph memory, ask for verification, or
  fall back to source truth from `context_status` alone.

### P2: Add Non-GitHub Source Depth

Required next resolver/ingestion targets:

- Linear issues/teams for tickets and planning context.
- Slack channels/threads for discussions, decisions, and incidents.
- Incident/alert systems for operations and debugging memory.
- Deployment systems for environment and release context.
- Documentation sources for design docs, runbooks, and setup guides.

Acceptance criteria:

- At least one non-GitHub source can ingest events, resolve source refs, verify
  facts, and return bounded summaries/snippets through `context_resolve`.

### P3: Retire Legacy Ports And Compatibility APIs

After P0-P2 are complete:

- Delete or move `EpisodicGraphPort` and `StructuralGraphPort` behind adapter
  internals.
- Remove deprecated aliases after clients migrate.

## Non-Negotiables

- Do not add a public tool per context family.
- Do not make source-specific graph write branches.
- Do not copy full source payloads into the graph by default.
- Do not make agents know Graphiti, Cypher, or Neo4j labels.
- Do not return facts without enough provenance for the consumer to understand
  where they came from and how fresh they are.
