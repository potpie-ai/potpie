# Agent Reconciliation Ingestion Plan

> Historical design document. The current source of truth is [`INGESTION_ASYNC_PLAN.md`](./INGESTION_ASYNC_PLAN.md).
> Keep this file for migration context and prior rationale; implement new ingestion architecture from the async plan.

## Goal

Replace the current direct ingestion flow:

- event arrives
- fetch source payload
- format one episode
- write Graphiti
- stamp structural graph

with a reconciliation-based flow:

- event arrives
- persist raw event and normalize scope
- run a bounded reconciliation agent inside `context-engine`
- the agent uses library-defined tools to query context and external systems
- the agent returns a structured mutation plan
- deterministic appliers validate and apply graph updates
- ledger records the full lifecycle for replay, retries, and audit

This plan assumes a strict package boundary:

- `app/src/context-engine` must remain a standalone library
- code inside `app/src/context-engine` must not import from the rest of Potpie
- Potpie may import and wire `context-engine`
- any functionality needed by the library must be ported into `context-engine` behind ports/adapters

## Current State

The package currently has two ingestion paths:

- raw episode ingest via `application/use_cases/ingest_episode.py`
- GitHub PR ingest/backfill via `application/use_cases/ingest_single_pr.py`, `application/use_cases/backfill_pot.py`, and `application/use_cases/ingest_merged_pr.py`

The current PR pipeline is tightly coupled to a deterministic GitHub PR model:

1. fetch PR bundle from `SourceControlPort`
2. build a single Graphiti episode body
3. write the episode to Graphiti
4. stamp PR/commit/decision entities in structural Neo4j
5. write bridge edges
6. update Postgres ledger rows

This is too narrow for an agent-driven reconciliation model because:

- ingestion assumes the source of truth is one fetched payload
- structural writes are PR-shaped rather than generic
- the ledger tracks ingestion/bridge status, not agent planning or replay
- there is no write-safe abstraction for agent-produced graph mutations
- there is no library-owned tool abstraction for querying external systems during reconciliation

## Target Architecture

### Core principle

The agent must plan, not write directly.

The model should never execute arbitrary graph mutations or arbitrary adapter code. It should produce a constrained, typed reconciliation plan. The library should validate and apply that plan deterministically.

### Ownership split

Inside `context-engine`:

- event normalization models
- reconciliation use cases
- agent runner interface
- tool interface definitions
- reconciliation ledger interfaces
- graph mutation models
- mutation validation
- graph mutation appliers
- default inbound workflows for HTTP, CLI, MCP, or host-triggered execution

Outside `context-engine`:

- concrete adapter implementations for GitHub, Linear, Jira, Confluence, or other integrations
- concrete agent runtime implementation if the host wants to reuse an existing LLM stack
- concrete tool implementations bound to host auth, secrets, and tenancy
- concrete pot/project resolution adapters

### New ingestion flow

1. An inbound adapter receives an event or a backfill job.
2. The event is normalized into a library-owned `ContextEvent`.
3. The ledger records the event as received and deduplicates by scoped event identity.
4. A reconciliation use case resolves pot scope and available tool capabilities.
5. A reconciliation agent is invoked with:
   - normalized event
   - pot scope
   - graph read tools
   - external integration tools
   - optional prior event state from the ledger
6. The agent returns a typed `ReconciliationPlan`.
7. The library validates the plan.
8. The library applies:
   - Graphiti episode writes
   - structural graph entity and edge mutations
   - invalidations or supersessions
9. The ledger records success, skip, or failure with provenance.

## Design Constraints

### Standalone library rule

No code in `app/src/context-engine` should import from:

- `app.modules.*`
- `app.core.*`
- any Potpie-specific auth, DB, tool service, or agent classes

This means any existing Potpie functionality needed by reconciliation must be expressed as:

- a domain port in `context-engine`
- an outbound adapter in `context-engine` if it can remain generic
- a host-provided implementation outside the package

### Deterministic write path

The agent should not:

- emit Cypher
- emit arbitrary Graphiti API calls
- directly run tool side effects
- update ledger rows itself

The only permitted write path is:

- agent returns typed plan
- library validates plan
- library appliers execute allowed mutations

### Multi-integration model

The package should stop assuming all ingestion is GitHub PR ingestion.

The reconciliation flow should support events from:

- GitHub
- Linear
- Jira
- Confluence
- future systems

The source system should affect:

- event normalization
- available tools
- evidence used by the agent
- the shape of generated episodes and structural mutations

but not the overall lifecycle.

## Proposed Library Changes

## 1. New domain models

Add a new module group under `domain/` for reconciliation.

Suggested files:

- `domain/reconciliation.py`
- `domain/context_events.py`
- `domain/graph_mutations.py`

Suggested models:

- `ContextEvent`
- `EventRef`
- `EventScope`
- `ReconciliationRequest`
- `ReconciliationPlan`
- `EvidenceRef`
- `EpisodeDraft`
- `EntityUpsert`
- `EdgeUpsert`
- `EdgeDelete`
- `InvalidationOp`
- `MutationSummary`
- `ReconciliationResult`

Suggested event fields:

- `event_id`
- `source_system`
- `event_type`
- `action`
- `pot_id`
- `provider`
- `provider_host`
- `repo_name`
- `artifact_refs`
- `occurred_at`
- `received_at`
- `payload`

Suggested plan fields:

- `event_ref`
- `summary`
- `episodes`
- `entity_upserts`
- `edge_upserts`
- `edge_deletes`
- `invalidations`
- `evidence`
- `confidence`
- `warnings`

## 2. New ports

Add ports that let `context-engine` define the reconciliation contract without depending on Potpie code.

### Agent execution port

Add `domain/ports/reconciliation_agent.py`.

Responsibilities:

- execute one reconciliation request
- expose available capability metadata if useful
- return a typed `ReconciliationPlan`

Example shape:

- `run_reconciliation(request: ReconciliationRequest) -> ReconciliationPlan`

This port allows Potpie to provide an adapter backed by its current agent runtime without importing that runtime inside the package.

### Integration tool port

Add `domain/ports/reconciliation_tools.py`.

This should not mirror LangChain or any existing Potpie tool class. It should be library-owned and minimal.

Suggested responsibilities:

- list tool descriptors available for this run
- execute a named read-only query tool with typed arguments

Example categories:

- context graph read
- GitHub read
- Linear read
- Jira read
- Confluence read

Important rule:

the initial reconciliation pipeline should use read-only tools only. Side-effect tools such as creating Jira issues or updating Linear should remain outside this flow.

### Mutation applier ports

Add:

- `domain/ports/episode_writer.py`
- `domain/ports/graph_mutation_applier.py`

These ports separate:

- episodic writes
- structural graph writes

from the current PR-specific methods in `StructuralGraphPort`.

### Reconciliation ledger port

Add `domain/ports/reconciliation_ledger.py`.

Responsibilities:

- append event
- claim event for processing
- record plan metadata
- record apply success or failure
- load prior attempts
- support replay and idempotency checks

## 3. Refactor existing ports

### Structural graph

`domain/ports/structural_graph.py` is currently too specialized around PR stamping and bridge writes.

Refactor strategy:

- keep existing query methods intact
- keep existing reset intact
- deprecate PR-shaped write methods over time
- add generic mutation application methods

Suggested additions:

- `upsert_entities(...)`
- `upsert_edges(...)`
- `delete_edges(...)`
- `apply_invalidations(...)`

Existing methods:

- `stamp_pr_entities`
- `write_bridges`

should become compatibility helpers built on top of the generic mutation layer, then eventually become internal-only or deprecated.

### Episodic graph

`domain/ports/episodic_graph.py` currently only exposes add/search/reset behavior.

Extend it carefully to support:

- writing one or more `EpisodeDraft` objects
- optionally tagging episodes with source event metadata and provenance

The implementation can still map to `Graphiti.add_episode(...)`, but the library contract should become more explicit than raw argument lists.

### Source control

`domain/ports/source_control.py` is GitHub PR-shaped today.

Do not remove it immediately. It still supports current ingestion.

Add new provider-neutral integration query ports instead of forcing GitHub semantics to represent Linear/Jira:

- `ArtifactQueryPort`
- `IssueTrackerPort`
- `WorkTrackingPort`
- or one generic `IntegrationQueryPort`

A good compromise is:

- keep `SourceControlPort` for code-host operations
- add separate ports for issue/workflow systems

## 4. New application use cases

Add a new use case group under `application/use_cases/`.

Suggested files:

- `reconcile_event.py`
- `apply_reconciliation_plan.py`
- `record_context_event.py`
- `replay_context_event.py`
- `build_reconciliation_request.py`

### `record_context_event`

Responsibilities:

- normalize raw inbound payload
- validate required event identifiers
- persist raw event and dedupe
- return whether processing should continue

### `reconcile_event`

Responsibilities:

- resolve pot and repo scope
- gather prior event state
- build agent request
- invoke the reconciliation agent port
- validate the returned plan
- call plan applier
- update ledger status

### `apply_reconciliation_plan`

Responsibilities:

- write episodes first
- then apply structural mutations
- then update event/application status
- ensure idempotency for retries where possible

This must remain deterministic and testable without an LLM.

## 5. New ledger schema

The current schema in `adapters/outbound/postgres/models.py` tracks:

- sync state
- ingestion log
- raw events

That is not enough for agentized reconciliation.

Add new tables or evolve the existing ones.

Recommended new tables:

### `context_events`

Stores the canonical inbound event.

Suggested fields:

- `id`
- `pot_id`
- `provider`
- `provider_host`
- `repo_name`
- `source_system`
- `event_type`
- `action`
- `source_id`
- `source_event_id`
- `payload`
- `occurred_at`
- `received_at`
- `status`

### `context_reconciliation_runs`

Stores each attempt to reconcile an event.

Suggested fields:

- `id`
- `event_id`
- `attempt_number`
- `status`
- `agent_name`
- `agent_version`
- `toolset_version`
- `plan_summary`
- `episode_count`
- `entity_mutation_count`
- `edge_mutation_count`
- `error`
- `started_at`
- `completed_at`

### `context_reconciliation_artifacts`

Optional table for storing:

- validated plan JSON
- evidence refs
- mutation summaries

This is useful for replay, debugging, and audit.

Compatibility approach:

- keep `raw_events` temporarily for backward compatibility
- keep `context_ingestion_log` for existing PR ingestion while the new path rolls out
- do not force old and new pipelines into the same table semantics

## 6. Graph mutation model

The package needs a generic write vocabulary for structural updates.

Suggested supported operations for v1:

- upsert entity by deterministic key
- update entity properties
- upsert edge by deterministic relationship identity
- delete edge by deterministic identity
- mark fact/entity invalidated with provenance

Suggested constraints:

- every mutation must include `pot_id`
- every mutation must include a provenance ref to the source event or produced episode
- entity and edge types must be validated against a registry
- mutation counts should be capped per run

PR bridging should become one specific mutation family, not a special hardcoded ingestion path.

## 7. Reconciliation agent contract

The most important design rule is to keep the agent contract narrow.

### Inputs

- normalized event
- pot scope
- available tool descriptors
- optional previous graph summary
- optional previous attempts

### Outputs

- plan summary
- evidence used
- episode drafts
- structural mutations
- warnings

### Tool policy

Allow only read tools during planning:

- search semantic context
- load PR or issue details
- fetch review discussions
- fetch linked Jira issues
- fetch linked Linear issues
- query change history

Do not allow the reconciliation agent to call mutation tools against external systems in this flow.

### Validation

Reject plans that:

- write outside the requested pot
- mutate unsupported labels or edge types
- contain unbounded text blobs where summaries are expected
- omit provenance
- exceed size limits

## 8. Adapter strategy inside the standalone package

The package should include only generic adapters or adapters backed by generic dependencies.

Allowed examples:

- Postgres ledger adapter
- Neo4j structural adapter
- Graphiti episodic adapter
- GitHub source-control adapter if packaged as an optional dependency

Do not add Potpie-specific agent adapters inside `context-engine`.

Instead:

- define the agent port in `context-engine`
- let Potpie implement that port and pass it into the container

If useful, the package can include a minimal generic adapter such as:

- `adapters/outbound/reconciliation/null_agent.py`
- `adapters/outbound/reconciliation/rule_based_agent.py`

for tests and fallback operation.

## 9. Container and wiring changes

Update `bootstrap/container.py` so the library container can optionally carry:

- reconciliation agent
- reconciliation ledger
- integration query ports

Do not hardwire Potpie services into the container builder.

Suggested approach:

- extend `ContextEngineContainer`
- keep `build_container(...)` generic
- let host code provide optional adapters

Example new dependencies:

- `reconciliation_agent`
- `reconciliation_ledger`
- `artifact_query`
- `issue_tracker`

The package should not know whether the host backed those with Potpie integrations, standalone credentials, or mocks.

## 10. Inbound adapter changes

### HTTP

Current routes under `adapters/inbound/http/api/v1/context/router.py` should evolve as follows:

- keep raw `/ingest` for direct episode ingest
- keep `/ingest-pr` temporarily for compatibility
- add event-oriented routes if needed:
  - `/events/reconcile`
  - `/events/replay`
  - `/events/{id}`

Compatibility recommendation:

- keep `/ingest-pr` but reimplement it internally by creating a normalized GitHub PR merged event and sending it through `reconcile_event`

### CLI

Current CLI ingest should stay for raw episodes.

If event-driven reconciliation needs CLI coverage, add a separate command such as:

- `context-engine reconcile-event --source github --event-file payload.json`

Do not overload raw `ingest` with agent-driven reconciliation semantics.

### MCP

No write-focused MCP changes are required initially.

## 11. Migration path from current PR ingestion

### Stage 1

Introduce the new reconciliation types, ports, and ledger schema without changing existing behavior.

### Stage 2

Implement a deterministic compatibility planner for merged GitHub PR events that produces the same effective writes as current `ingest_merged_pr`.

This provides:

- regression baseline
- replay support
- shape validation for the new plan format

### Stage 3

Introduce the true agent-backed planner for GitHub PR events behind a feature flag.

### Stage 4

Route `/ingest-pr` and backfill through `reconcile_event`.

### Stage 5

Expand to Linear and Jira events.

### Stage 6

Retire direct PR-shaped write methods once generic mutation appliers fully cover them.

## 12. Potpie integration strategy

Potpie should remain a host and adapter provider, not a dependency of `context-engine`.

Potpie responsibilities would be:

- implement `ReconciliationAgentPort`
- implement integration query ports using existing auth and secret storage
- wire them into the `ContextEngineContainer`
- trigger reconciliation from event bus and Celery

When reviewing Potpie code for reuse, the expected source material is:

- event bus handlers and tasks
- existing integration clients and tools
- existing agent runtime

But any reused logic needed by the library should be copied or re-expressed under `app/src/context-engine` in a package-owned abstraction.

## 13. Testing plan

### Unit tests

Add focused tests for:

- event normalization
- ledger deduplication
- plan validation
- plan application ordering
- mutation caps and rejection behavior
- compatibility planning for GitHub PR merged events

### Adapter tests

Add tests for:

- Postgres reconciliation ledger adapter
- structural mutation applier
- episode writer changes

### Integration tests

Add end-to-end tests for:

- GitHub merged PR event -> reconciliation plan -> applied graph changes
- replay of a duplicate event
- partially failed apply with retry

### Backward compatibility tests

For a fixture PR payload, compare:

- old direct ingestion outputs
- new compatibility-plan outputs

to ensure the generic pipeline preserves existing semantics before the agent is introduced.

## 14. Rollout and feature flags

Suggested flags:

- `CONTEXT_ENGINE_RECONCILIATION_ENABLED`
- `CONTEXT_ENGINE_AGENT_PLANNER_ENABLED`
- `CONTEXT_ENGINE_COMPAT_PR_RECONCILER_ENABLED`

Recommended rollout:

1. ship schema and passive event recording
2. ship compatibility planner in shadow mode
3. compare outputs against current PR ingestion
4. enable deterministic reconciler for PRs
5. enable agent planner for selected tenants or sources

## 15. Immediate implementation order

1. Add new domain models for events, plans, and mutations.
2. Add new ports for reconciliation agent, tools, mutation applier, and ledger.
3. Add new Postgres schema and adapter for reconciliation lifecycle.
4. Add generic mutation applier support to structural and episodic adapters.
5. Add `reconcile_event` and `apply_reconciliation_plan` use cases.
6. Add a deterministic GitHub PR compatibility planner inside `context-engine`.
7. Re-route existing PR ingest use cases through the compatibility planner.
8. Add host wiring for a real agent-backed planner outside the package.
9. Add support for Linear and Jira event normalization and query ports.

## 16. Non-goals for the first iteration

- allowing the agent to execute arbitrary write tools
- replacing raw episodic ingest
- removing existing PR ingestion code on day one
- solving every integration in one pass
- making standalone CLI mode fully parity-complete with Potpie-hosted integrations

The first iteration should focus on:

- a sound standalone-library design
- a generic reconciliation lifecycle
- safe typed graph mutations
- compatibility with current GitHub PR ingestion
