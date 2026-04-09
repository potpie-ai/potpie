# Ingestion Event Store Architecture

## Goal

Define the ingestion system from first principles, with clean interfaces and boundaries, for a world where:

- every inbound ingestion request becomes a durable event
- every event is processed through queue-driven workers
- one event may produce multiple episodes
- event completion depends on all required episodes succeeding
- sync callers use the same architecture and only differ in response behavior
- the dashboard reads ingestion state from the same source of truth as APIs

This document intentionally optimizes for proper system boundaries, not for preserving current implementation shapes.

## Core Design Principles

1. The event store is the system of record for ingestion lifecycle, not a side ledger.
2. Producers only submit events. They do not perform ingestion work.
3. Planning and execution are separate responsibilities.
4. Integration-specific code must stop at normalization and evidence collection boundaries.
5. The write path to the context graph must be deterministic.
6. Operational status APIs and dashboard reads must come from the same ingestion model, not from special-purpose tables.
7. Sync mode is not a separate pipeline. It is async architecture with a waiter.

## System Boundaries

The architecture should be split into six logical subsystems.

### 1. Ingestion API Boundary

Responsibility:

- accept requests from CLI, HTTP, webhooks, and integrations
- authenticate and authorize
- validate input
- normalize input into a canonical ingestion event request
- call the ingestion submission service

Not responsible for:

- provider-specific fetch logic
- planning episodes
- writing Graphiti or Neo4j
- polling worker internals directly

This boundary should be very thin. It is an adapter layer only.

### 2. Ingestion Control Plane

Responsibility:

- assign event id
- compute dedup identity
- persist canonical event
- return existing event on duplicate
- enqueue processing
- expose event status and event listing APIs
- support sync waiting on terminal state

This is the orchestration layer. It owns lifecycle and durability, not ingestion semantics.

The control plane is where the dashboard gets its truth.

### 3. Source Normalization Boundary

Responsibility:

- transform producer-specific payloads into canonical event envelopes
- preserve raw payload
- define source metadata such as source system, source channel, event type, action, and dedup key

Examples:

- GitHub webhook payload -> canonical event
- CLI raw text -> canonical event
- Linear webhook payload -> canonical event

This layer should not decide how many episodes are produced or how graph writes happen.

### 4. Ingestion Planning Boundary

Responsibility:

- turn one canonical event into one or more ordered ingestion operations
- optionally fetch external evidence through read-only provider interfaces
- produce a deterministic plan payload for execution

Important:

- this is where an LLM-backed planner may exist
- this is also where deterministic planners may exist
- the planner outputs instructions, not writes

The planning boundary should not own persistence of lifecycle state beyond returning plan artifacts to the control plane.

### 5. Ingestion Execution Boundary

Responsibility:

- consume durable planned episode steps
- apply one step at a time
- write to Graphiti and structural graph through deterministic executors
- update step and event status

Important:

- no inbound surface should call Graphiti directly
- no planner should call Graphiti directly
- execution owns graph mutation side effects

### 6. Query / Read Boundary

Responsibility:

- serve event detail and event list APIs
- support dashboard views
- eventually support realtime subscriptions

This is a read model over ingestion state. It should not need to know planner internals.

## Canonical Domain Objects

The architecture should revolve around four core domain objects.

### 1. Ingestion Event

Represents the durable unit of requested work.

Fields:

- `event_id`
- `pot_id`
- `source_channel`
- `source_system`
- `event_type`
- `action`
- `dedup_key`
- `status`
- `submitted_at`
- `started_at`
- `completed_at`
- `error`
- `payload`
- `metadata`

This is the parent object seen by APIs and dashboard.

### 2. Ingestion Plan

Represents the planner output for one event.

Fields:

- `event_id`
- `plan_id`
- `version`
- `steps`
- `summary`
- `artifacts`

The plan is durable and replayable. It should be stored before execution begins.

### 3. Episode Step

Represents one ordered execution unit derived from a plan.

Fields:

- `event_id`
- `step_id`
- `sequence`
- `kind`
- `status`
- `input`
- `attempt_count`
- `result`
- `error`

An event can have one or many episode steps.

### 4. Execution Result

Represents the deterministic output of a step executor.

Fields:

- `step_id`
- `success`
- `episode_ref`
- `structural_effects`
- `error`

## Status Model

Keep the external event lifecycle simple:

- `queued`
- `processing`
- `done`
- `error`

Keep step lifecycle separate:

- `queued`
- `processing`
- `done`
- `error`

Avoid leaking internal planning vocabulary such as `reconciled` into the public contract.

If more observability is needed, add a separate stage field:

- `accepted`
- `planning`
- `planned`
- `executing`
- `completed`
- `failed`

Public `status` stays stable while `stage` gives operators more detail.

## First-Class Interfaces

These are the interfaces that matter architecturally.

### 1. `IngestionSubmissionService`

Responsibility:

- submit an event for a pot
- deduplicate
- persist event
- enqueue processing
- optionally wait for completion

Contract:

- input: canonical ingestion submission request
- output: event receipt with `event_id`, `status`, optional final result

This is the only service inbound adapters should call.

### 2. `IngestionEventStore`

Responsibility:

- create event
- get event by id
- find duplicate by dedup key
- list events for pot with pagination and filters
- transition event state
- persist progress counters

This is not a reconciliation ledger. It is the primary operational store.

### 3. `IngestionQueue`

Responsibility:

- enqueue event processing
- enqueue step execution

It should not contain business logic. It is transport only.

### 4. `EventPlanner`

Responsibility:

- load any required evidence
- convert one event into an ordered plan

Contract:

- input: canonical event
- output: durable ingestion plan

This may have multiple implementations:

- deterministic raw planner
- GitHub merged PR planner
- Linear issue planner
- generic agent-backed planner

### 5. `PlanStore`

Responsibility:

- persist plan artifacts
- persist ordered steps
- fetch the next steps for an event
- support replay and resume

This should be distinct from the event store, even if the same database backs both.

Reason:

- event lifecycle and step execution are related but not the same thing
- separating their interfaces prevents lifecycle logic from becoming coupled to execution internals

### 6. `StepExecutor`

Responsibility:

- apply one planned step deterministically
- return execution result

There may be different executors for:

- raw episode writes
- graph episode writes plus structural mutations

### 7. `ContextGraphWriter`

Responsibility:

- the only interface allowed to write to Graphiti / structural graph

This boundary is important. It prevents planners, source connectors, and HTTP surfaces from bypassing execution rules.

### 8. `SourceConnector`

Responsibility:

- read provider data needed by planners

Examples:

- GitHub connector
- Linear connector
- Jira connector

Important rule:

- source connectors are read-only in the ingestion path

They provide evidence, not side effects.

### 9. `EventQueryService`

Responsibility:

- get event detail
- list events for a pot
- filter by multiple statuses
- paginate latest first

This should power both dashboard and public API.

## Proper Separation of Concerns

The main architectural mistake to avoid is combining these concerns in one service:

- submission
- dedupe
- provider fetch
- planning
- graph writes
- status tracking

That creates tight coupling and makes retries, dashboards, and integrations hard to reason about.

The clean split is:

1. submit event
2. plan event
3. execute steps
4. read status

Each stage has its own contract and its own storage responsibility.

## Event Processing Model

### Submission

Producer submits canonical request.

The control plane:

1. validates pot scope
2. computes dedup key
3. stores event as `queued`
4. enqueues event processing
5. returns receipt

### Planning

Planner worker:

1. claims event
2. marks event `processing`
3. builds ingestion plan
4. stores plan and ordered steps
5. enqueues step execution

The planner should never directly mark an event `done`.

### Execution

Step executor worker:

1. claims next executable step
2. applies deterministic write
3. marks step `done` or `error`
4. updates event progress
5. marks event `done` only when all required steps are `done`

### Sync Behavior

Sync requests:

1. call the same submission service
2. receive an event id
3. wait on event terminal state
4. return final event result

This gives one architecture, one audit trail, and one dashboard story.

## Ordering and Concurrency

The correct partition for write serialization is `pot_id`.

Reason:

- the graph is pot-scoped shared state
- ordering within an event is necessary but insufficient
- two different events for the same pot can conflict

Rule:

- planning may run concurrently
- execution must be serialized per `pot_id`
- execution may run concurrently across different pots

This is a system-level invariant, not an optimization.

## Dedupe Model

Dedupe belongs at the event boundary, not deep inside ingestion handlers.

Each source normalizer must define how `dedup_key` is computed.

Examples:

- GitHub merged PR: provider + repo + PR number + merge commit or merge action version
- raw CLI event: explicit client idempotency key if supplied, otherwise no dedupe
- webhook delivery: provider delivery id when semantics allow it

Important:

- `event_id` is identity
- `dedup_key` is submission equivalence

Do not overload one field for both.

## Storage Model

A proper model has at least three durable concerns:

### 1. Event Store

Stores:

- event identity
- status
- metadata
- timestamps
- progress

### 2. Plan Store

Stores:

- planner output
- ordered step definitions
- planner artifacts

### 3. Execution Log

Stores:

- attempt counts
- step results
- errors
- applied refs

These may live in one Postgres database, but they should not be treated as one conceptual table or one overloaded interface.

## API Surface

The correct external API model is minimal.

### Submit event

All ingest surfaces eventually call one submission API:

- async returns `202` with `event_id`
- sync returns `200` with terminal event result

### Get event

- `GET /events/{event_id}`

Returns:

- event metadata
- status
- progress
- error
- child step summary

### List events for pot

- `GET /pots/{pot_id}/events`

Requirements:

- latest first by default
- cursor pagination
- multi-status filtering

This is the main dashboard feed.

## Dashboard Architecture

The dashboard should not read workers directly and should not depend on broker state.

It should read the event query service only.

Why:

- broker state is transient
- database state is durable
- dashboard semantics match event semantics, not transport semantics

Optional later additions:

- live updates via SSE or websocket
- aggregate counts
- retry actions

## Recommended Naming

Use ingestion terminology consistently:

- `IngestionEvent`
- `IngestionPlan`
- `EpisodeStep`
- `IngestionSubmissionService`
- `IngestionEventStore`
- `EventPlanner`
- `StepExecutor`
- `EventQueryService`

Avoid using `reconciliation` as the top-level system name. It is one planning strategy, not the architecture.

## Recommended Non-Goals

Do not build these into the first version:

- broker-driven status as the source of truth
- direct graph writes from producers
- planner-generated arbitrary graph mutations
- per-integration custom lifecycle stores
- separate sync-only ingestion path

## Summary

The proper architecture is:

- adapters submit canonical ingestion events
- a control plane owns lifecycle, dedupe, and queueing
- planners convert events into durable ordered steps
- executors apply steps deterministically
- the event store backs status APIs and the dashboard

The most important architectural decision is this one:

- event submission, planning, execution, and querying must be separate interfaces

If those stay separate, the system will scale cleanly to CLI, APIs, webhooks, GitHub, Linear, and future integrations without collapsing into source-specific ingestion code.
