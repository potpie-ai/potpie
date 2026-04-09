# Ingestion Event Store Implementation Plan

## Goal

Implement the ingestion system described in [ingestion-event-store-architecture.md](/Users/nandan/Desktop/Dev/potpie/docs/implementation-plans/ingestion-event-store-architecture.md) with clean boundaries:

- submission
- control plane
- planning
- execution
- query/read

This plan assumes refactoring is acceptable. It does not optimize for preserving current service boundaries if they are architecturally wrong.

## Target End State

At the end of this work:

- every producer submits canonical ingestion events
- the event store is the source of truth for lifecycle and dashboard state
- planners produce durable ordered steps
- executors are the only writers to the context graph
- sync and async use the same submission path
- status APIs and dashboard reads come from the event query service
- provider-specific ingestion code lives behind source normalization and planner interfaces

## Desired Module Boundaries

The implementation should converge on the following logical modules, even if file paths differ initially.

### 1. Submission Layer

Owns:

- inbound HTTP / CLI / webhook adapters
- canonical request validation
- source normalization
- calls into submission service

Exports:

- `IngestionSubmissionService`

### 2. Control Plane Layer

Owns:

- event ids
- dedupe
- event persistence
- queue submission
- sync wait behavior
- status transitions

Exports:

- `IngestionEventStore`
- `IngestionQueue`
- `EventQueryService`

### 3. Planning Layer

Owns:

- event-to-plan transformation
- evidence fetching through read-only connectors
- plan persistence

Exports:

- `EventPlanner`
- `PlanStore`
- `SourceConnector`

### 4. Execution Layer

Owns:

- claiming executable steps
- deterministic step application
- graph writes
- step and event progress updates

Exports:

- `StepExecutor`
- `ContextGraphWriter`

## Canonical Interfaces to Introduce

These interfaces should be introduced explicitly before large migration work.

### `IngestionSubmissionService`

Methods:

- `submit(request, wait=False, timeout=None) -> EventReceipt`

Responsibilities:

- accept canonical submission request
- compute dedup behavior
- store event
- enqueue processing
- optionally wait for terminal state

### `IngestionEventStore`

Methods:

- `create_event(...)`
- `get_event(event_id)`
- `find_duplicate(pot_id, dedup_key, ingestion_kind)`
- `transition_event(...)`
- `list_events(pot_id, filters, cursor, limit)`
- `record_progress(...)`

Responsibilities:

- event lifecycle only

### `PlanStore`

Methods:

- `save_plan(...)`
- `get_plan(plan_id)`
- `replace_steps_for_event(...)`
- `get_step(...)`
- `list_steps_for_event(...)`
- `claim_next_step_for_pot(...)`
- `record_step_result(...)`

Responsibilities:

- plan and step durability
- execution coordination data

### `IngestionQueue`

Methods:

- `enqueue_event(event_id)`
- `enqueue_step(event_id, step_id, pot_id)`

Responsibilities:

- transport only

### `EventPlanner`

Methods:

- `plan(event) -> IngestionPlan`

Responsibilities:

- produce ordered durable steps
- never write to graph directly

### `StepExecutor`

Methods:

- `execute(step) -> ExecutionResult`

Responsibilities:

- deterministic execution only

### `ContextGraphWriter`

Methods:

- implementation-specific deterministic graph write operations

Responsibilities:

- only place allowed to mutate graph state

## Canonical Data Model

The implementation should stabilize around three durable stores.

### 1. Event Store

Stores event lifecycle:

- `event_id`
- `pot_id`
- `source_channel`
- `source_system`
- `event_type`
- `action`
- `dedup_key`
- `status`
- `stage`
- `submitted_at`
- `started_at`
- `completed_at`
- `error`
- `payload`
- `metadata`
- `step_total`
- `step_done`
- `step_error`

### 2. Plan Store

Stores planner output:

- `plan_id`
- `event_id`
- `planner_type`
- `version`
- `summary`
- `artifacts`

### 3. Step Store

Stores execution units:

- `step_id`
- `event_id`
- `pot_id`
- `sequence`
- `kind`
- `status`
- `input`
- `attempt_count`
- `result`
- `error`
- `queued_at`
- `started_at`
- `completed_at`

These may share one Postgres database, but they should not share one overloaded interface.

## Work Phases

## Phase 1: Establish the Domain Contracts

### Goal

Introduce the clean interfaces and canonical models before moving producers.

### Work

1. Define domain models:
   - `IngestionEvent`
   - `IngestionPlan`
   - `EpisodeStep`
   - `ExecutionResult`
   - `EventReceipt`

2. Define interfaces:
   - `IngestionSubmissionService`
   - `IngestionEventStore`
   - `PlanStore`
   - `IngestionQueue`
   - `EventPlanner`
   - `StepExecutor`
   - `ContextGraphWriter`
   - `EventQueryService`

3. Rename reconciliation-specific top-level concepts out of the main architecture.

### Outcome

The codebase has the right architectural vocabulary before migration starts.

## Phase 2: Build the Control Plane Properly

### Goal

Make event lifecycle a first-class operational subsystem.

### Work

1. Implement Postgres-backed `IngestionEventStore`.
2. Add or adjust schema for:
   - external event status
   - internal stage
   - dedup key
   - progress counters
   - lifecycle timestamps
3. Implement `EventQueryService`.
4. Implement sync wait behavior against event terminal state.

### Important decision

The event store is the operational truth. Do not hide it behind planner-specific interfaces.

### Outcome

Submission, status lookup, list APIs, and dashboard reads now have a coherent storage model.

## Phase 3: Build the Plan / Step Store Boundary

### Goal

Separate planning durability from event lifecycle durability.

### Work

1. Implement `PlanStore`.
2. Persist plan artifacts before any execution begins.
3. Persist ordered step rows as first-class execution units.
4. Make replay and resume operate from stored plans and steps, not from rerunning the planner by default.

### Outcome

Planner output becomes durable, replayable, and independent from queue delivery.

## Phase 4: Introduce Submission Service and Move Adapters Behind It

### Goal

Make all inbound surfaces call one service.

### Work

1. Implement `IngestionSubmissionService`.
2. Move HTTP raw ingest to use it.
3. Move HTTP event ingest to use it.
4. Move CLI ingest to use it.
5. Move webhook handlers to normalize then submit.

### Rules

- adapters may normalize
- adapters may authenticate
- adapters may authorize
- adapters may not enqueue directly
- adapters may not write Graphiti directly

### Outcome

All producers share one entrypoint and one event lifecycle.

## Phase 5: Split Planning Workers from Execution Workers

### Goal

Make planning and execution independent runtime stages.

### Work

1. Implement event-processing worker that:
   - claims event
   - marks event `processing`
   - calls planner
   - stores plan
   - stores steps
   - enqueues steps

2. Implement step-processing worker that:
   - claims next executable step
   - executes deterministically
   - records result
   - updates event progress

3. Remove any remaining direct graph writes from planning flows.

### Outcome

Workers reflect the intended architecture instead of a mixed orchestration model.

## Phase 6: Protect Graph Writes Behind `ContextGraphWriter`

### Goal

Make deterministic execution the only write path.

### Work

1. Introduce `ContextGraphWriter`.
2. Move Graphiti and structural graph side effects behind it.
3. Update executors to call only this boundary.
4. Ensure planners and source connectors cannot bypass it.

### Outcome

Graph mutation safety is enforced structurally, not by convention.

## Phase 7: Implement Pot-Scoped Execution Serialization

### Goal

Enforce the actual correctness boundary for graph writes.

### Work

1. Add pot-scoped execution claims or leases.
2. Ensure only one step executes for a given `pot_id` at a time.
3. Preserve sequence ordering within an event.
4. Allow concurrency across different pots.

### Recommended direction

Use database-backed pot claims first. Do not depend on broker partition guarantees as the only ordering mechanism.

### Outcome

Execution ordering is correct for shared pot-scoped graph state.

## Phase 8: Migrate Producer-Specific Logic into Normalizers and Planners

### Goal

Stop carrying source-specific ingestion behavior as top-level ingestion flows.

### Work

1. GitHub merged PR becomes:
   - source normalizer
   - optional evidence connector
   - planner implementation

2. Raw ingest becomes:
   - deterministic normalizer
   - deterministic planner

3. Backfill becomes:
   - event producer only
   - no direct graph writes

4. Future integrations follow the same model:
   - normalizer
   - connector
   - planner

### Outcome

New sources become additions to the planning ecosystem, not forks of the ingestion architecture.

## Phase 9: Finalize Query APIs and Dashboard Contract

### Goal

Make read surfaces first-class.

### Work

1. `GET /events/{event_id}`
2. `GET /pots/{pot_id}/events`
3. multi-status filtering
4. cursor pagination
5. latest-first ordering
6. event progress fields for dashboard

### Optional later

- SSE or websocket event updates
- retry actions
- aggregate counts

### Outcome

Dashboard and API users rely on one event read model.

## Refactoring Guidance

During implementation, prefer these directions:

### Good refactors

- extract lifecycle logic out of planner code
- move queueing behind submission service
- move graph writes behind execution interfaces
- rename reconciliation-specific system boundaries
- split event store and plan store responsibilities

### Avoid

- adding more status logic into source-specific handlers
- letting adapters call queue and DB separately
- storing planner and event lifecycle concerns behind one monolithic ledger interface
- making the dashboard depend on broker state
- keeping sync-only special paths for convenience

## Suggested Delivery Order

1. Introduce interfaces and canonical domain models.
2. Implement event store and query service.
3. Implement plan store and step store.
4. Implement submission service.
5. Migrate adapters onto submission service.
6. Implement planning worker and execution worker split.
7. protect graph writes behind writer interface.
8. enforce pot-scoped execution ordering.
9. migrate GitHub/backfill/raw flows into normalizer + planner model.
10. finalize event list/detail APIs.

## Definition of Done

This architecture is successfully implemented when:

- every producer calls `IngestionSubmissionService`
- every event is represented in the event store
- every plan is durable before execution
- every graph write happens only through execution
- event and step statuses are queryable from Postgres
- sync requests submit events and wait, instead of taking a separate code path
- per-pot event list APIs support pagination and multi-status filters
- dashboard reads the same event model as the APIs
- new integrations can be added by supplying a normalizer, optional connector, and planner rather than creating a new ingestion pipeline
