# Unified Graphiti Application Architecture

## Purpose

This document describes how to unify the current episodic and structural graph split into one application-layer graph model backed by Graphiti.

The broader product architecture is in [`graph.md`](graph.md). This document is narrower: it is the implementation architecture for replacing the current dual-port application model:

- `EpisodicGraphPort`
- `StructuralGraphPort`

with a single graph application layer whose default adapter writes and reads from Graphiti-backed Neo4j. Compatibility branches are not part of the target design.

## Current State

The implementation already stores both layers in Neo4j and often uses the same `Entity` nodes:

- Graphiti writes `Episodic`, `Entity`, and `RELATES_TO` data scoped by `group_id = pot_id`.
- Potpie structural code writes canonical labels and deterministic properties on `Entity` nodes.
- Potpie also writes code bridge edges from `FILE` / `NODE` code graph nodes to canonical `Entity` nodes.
- PR ingest writes an episode, receives an `episode_uuid`, stamps canonical PR/commit/decision entities, then writes code bridges.
- `context_resolve` already composes Graphiti search and structural reads through `HybridGraphIntelligenceProvider`.

So storage is partially unified, but the application layer is not. The code still has two mental models, and the current `GraphitiContextGraphAdapter` is a facade over these older surfaces:

```text
application/use_cases
    |
    +-- EpisodicGraphPort     -> GraphitiEpisodicAdapter
    +-- StructuralGraphPort   -> Neo4jStructuralAdapter
```

That split leaks into query code, reconciliation, hard reset, tests, and intelligence providers.

## Target State

The application layer should treat the context graph as one graph:

```text
application/use_cases
    |
    +-- ContextGraphPort
            |
            +-- GraphitiContextGraphAdapter
                    |
                    +-- Graphiti APIs for episodes, search, temporal facts
                    +-- controlled Neo4j writes for canonical Graphiti entities/edges
                    +-- code graph bridge reads/writes where existing FILE/NODE data remains external
```

Graphiti becomes the graph substrate. Potpie remains the ontology and query-contract owner.

This does not mean every write must go through Graphiti's LLM extraction. It means every application-level graph operation targets the same Graphiti-backed graph model:

- episodes are Graphiti episodes
- canonical facts are Graphiti `Entity` nodes and fact edges with Potpie-managed identity
- temporal state uses Graphiti-compatible fields
- semantic search and exact reads operate over the same entity space
- code graph bridges are treated as bridge facts into the context graph, not a second context graph

Source-specific behavior belongs before or after this graph layer:

- planners translate incoming GitHub, Linear, Slack, docs, incident, deployment,
  or alert events into generic reconciliation plans
- source resolvers fetch, summarize, verify, and snippet source payloads on
  demand
- graph apply code receives only ontology-validated generic mutations

The graph apply path must not branch on provider-specific compatibility bundles.

## Design Principles

1. One application graph port.
2. One query entrypoint for graph reads.
3. Graphiti is the backing graph substrate.
4. Potpie owns deterministic identity, ontology validation, and public response shapes.
5. Canonical facts must remain exact and explainable.
6. Semantic search is evidence and recall, not the only query mechanism.
7. Keep existing HTTP/CLI/MCP contracts stable while moving internals.
8. Migrate by adding a unified port first, then retiring old ports behind adapters.

## New Domain Port

Use `domain/ports/context_graph.py` as the application graph boundary.

The port should group graph operations by use, not by storage technology. The
read side already has the minimal shape:

```python
class ContextGraphPort(Protocol):
    @property
    def enabled(self) -> bool: ...

    def query(self, request: ContextGraphQuery) -> ContextGraphResult: ...
    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult: ...
```

Write unification is now the highest-priority migration step. Either extend this
port with a bounded write operation such as `apply_plan()`, or create a sibling
write port owned by the same Graphiti adapter. In both cases, application use
cases should depend on one graph abstraction rather than accepting episodic and
structural ports separately.

## Adapter Shape

Create `adapters/outbound/graphiti/context_graph.py`.

The adapter should replace the current combination of:

- `GraphitiEpisodicAdapter`
- most of `Neo4jStructuralAdapter`
- `StructuralGraphMutationApplier`
- the deleted `DefaultContextGraphWriter` wrapper

The current version internally delegates to existing code to reduce risk:

```text
GraphitiContextGraphAdapter
    |
    +-- GraphitiEpisodicAdapter for episode add/search/reset
    +-- Neo4jCanonicalGraphStore for deterministic canonical writes/reads
    +-- Neo4jCodeBridgeStore for FILE/NODE bridge reads/writes
```

After the write path is unified, the delegated classes should become adapter
internals or be deleted. They should not remain application-facing ports.

## Data Model

### Canonical Entities

Canonical entities should continue to be Graphiti-compatible `:Entity` nodes scoped by `group_id`.

Required properties:

- `group_id`
- `entity_key`
- `uuid`
- `name`
- `summary`
- `source_ref`
- `provenance_source_event`
- `provenance_episode_uuid`
- `created_at`
- `updated_at`
- `valid_from`
- `valid_to`
- `confidence`

Canonical labels still come from `domain/ontology.py`.

Examples:

- `Entity:PullRequest`
- `Entity:Decision`
- `Entity:Service`
- `Entity:Feature`
- `Entity:Incident`
- `Entity:Runbook`

### Canonical Edges

Potpie canonical relationships should be stored as first-class relationships between `:Entity` nodes.

Required relationship properties:

- `provenance_source_event`
- `provenance_episode_uuid`
- `source_ref`
- `confidence`
- `valid_from`
- `valid_to`
- `lifecycle_status`

For Graphiti semantic compatibility, the adapter may also mirror canonical typed edges into Graphiti-style `RELATES_TO` edges with `name = edge_type`, but that should be an adapter detail. The application contract should talk in canonical edge types.

### Episodes And Provenance

Episodes remain Graphiti `:Episodic` nodes.

Canonical facts should link to supporting episodes through one of these patterns:

- relationship property: `provenance_episode_uuid`
- explicit edge: `(:Entity)-[:SUPPORTED_BY]->(:Episodic)`
- explicit edge for relationships when needed: `(:FactAnchor)-[:SUPPORTED_BY]->(:Episodic)`

Start with relationship properties plus `SUPPORTED_BY` for high-value entities. Add fact anchors only if edge provenance queries become awkward.

### Code Graph Bridges

Existing `FILE` and `NODE` code graph nodes do not need to become Graphiti entities immediately. They are already a code graph maintained by Potpie.

The unified application model should treat code graph links as context graph bridge facts:

- `(:FILE)-[:TOUCHED_BY]->(:Entity:PullRequest)`
- `(:NODE)-[:MODIFIED_IN]->(:Entity:PullRequest)`
- `(:NODE)-[:HAS_DECISION]->(:Entity:Decision)`
- `(:Entity:Decision)-[:AFFECTS_CODE]->(:FILE|:NODE)` for new canonical writes

Longer term, add canonical `CodeAsset` entities as aliases for code graph nodes when agent queries need a pure Graphiti entity path:

```text
(:Entity:CodeAsset {entity_key: "code:file:repo:path"})
    -[:MATERIALIZES]->
(:FILE {repoId, file_path})
```

This keeps code graph ownership separate while making context graph traversal uniform.

## Unified Query Contract

The simplification goal is not just "put episodic and structural reads behind one adapter." The query layer should expose a minimal number of methods while still supporting precise graph traversal, fuzzy semantic retrieval, temporal reads, and task-oriented context bundles.

The target application read API should have one primary method:

```python
graph.query(request: ContextGraphQuery) -> ContextGraphResult
```

and one async variant:

```python
await graph.query_async(request: ContextGraphQuery) -> ContextGraphResult
```

Everything else should be a preset, wrapper, or response renderer.

Do not model the application query layer as many methods such as:

```python
get_change_history(...)
get_file_owners(...)
get_decisions(...)
get_pr_diff(...)
search(...)
```

Those names can remain as HTTP compatibility routes, CLI commands, or tests, but internally they should compile into `ContextGraphQuery`.

### Minimal Method Set

The graph application port should have only these read-oriented methods:

```python
class ContextGraphPort(Protocol):
    def query(self, request: ContextGraphQuery) -> ContextGraphResult: ...
    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult: ...
```

That gives the application layer one graph read concept:

- `query`: read facts/evidence/context from the graph.

### ContextGraphQuery Shape

`ContextGraphQuery` should be expressive enough to replace specialized methods:

```python
class ContextGraphQuery(BaseModel):
    pot_id: str

    # What the caller wants.
    query: str | None = None
    goal: Literal[
        "retrieve",
        "answer",
        "neighborhood",
        "timeline",
        "aggregate",
    ] = "retrieve"

    # How to retrieve.
    strategy: Literal[
        "auto",
        "semantic",
        "exact",
        "hybrid",
        "traversal",
        "temporal",
    ] = "auto"

    # What to include.
    include: list[str] = []
    exclude: list[str] = []

    # Scope filters.
    scope: ContextGraphScope = ContextGraphScope()

    # Effective query filters.
    node_labels: list[str] = []
    source_descriptions: list[str] = []
    episode_uuids: list[str] = []

    # Temporal controls.
    as_of: datetime | None = None
    include_invalidated: bool = False

    # Result budget.
    limit: int = 12
```

The query engine decides which Graphiti and canonical graph operations to run based on `goal`, `strategy`, `include`, and `scope`.

### Query Goals

Use `goal` to express the caller's intent without multiplying methods.

| Goal | Meaning | Replaces |
| --- | --- | --- |
| `retrieve` | Return matching records/evidence. | `search`, `decisions`, `project_graph`, `debugging_memory` |
| `answer` | Return an agent-ready context envelope. | `resolve_context` |
| `neighborhood` | Return a compact scoped subgraph. | `project_graph`, graph UI focused views |
| `timeline` | Return time-ordered changes/facts. | `change_history`, event history |
| `aggregate` | Return counts/health/schema summaries. | `graph_overview`, quality dashboards |

### Query Strategies

Use `strategy` to control retrieval mechanics:

| Strategy | Behavior |
| --- | --- |
| `semantic` | Graphiti hybrid/vector/keyword retrieval over episodes and entity edges. |
| `exact` | Deterministic property, label, key, and relationship matching. |
| `traversal` | Start from anchors and walk allowed relationships. |
| `temporal` | Exact or semantic reads constrained by `as_of` / validity windows. |
| `hybrid` | Exact + semantic + traversal, merged and ranked. |
| `auto` | Query planner chooses based on `goal`, `scope`, and `include`. |

This is where fuzzy semantic search fits naturally. It is not a separate API. It is `strategy="semantic"` or one leg of `strategy="hybrid"`.

### Scope Model

`ContextGraphScope` should absorb the fields currently scattered across request models:

```python
class ContextGraphScope(BaseModel):
    repo_name: str | None = None
    branch: str | None = None
    file_path: str | None = None
    function_name: str | None = None
    symbol: str | None = None
    pr_number: int | None = None
    services: list[str] = []
    features: list[str] = []
    environment: str | None = None
    ticket_ids: list[str] = []
    user: str | None = None
    source_refs: list[str] = []
```

### Presets Instead Of Methods

Specialized queries should be named presets that compile into `ContextGraphQuery`.

Examples:

```python
def preset_change_history(pot_id: str, file_path: str | None, function_name: str | None):
    return ContextGraphQuery(
        pot_id=pot_id,
        goal="timeline",
        strategy="traversal",
        include=["recent_changes", "decisions"],
        scope=ContextGraphScope(file_path=file_path, function_name=function_name),
    )
```

```python
def preset_file_owners(pot_id: str, file_path: str):
    return ContextGraphQuery(
        pot_id=pot_id,
        goal="aggregate",
        strategy="traversal",
        include=["owners"],
        scope=ContextGraphScope(file_path=file_path),
    )
```

```python
def preset_fuzzy_decisions(pot_id: str, text: str, file_path: str | None = None):
    return ContextGraphQuery(
        pot_id=pot_id,
        query=text,
        goal="retrieve",
        strategy="hybrid",
        include=["decisions", "semantic_search"],
        scope=ContextGraphScope(file_path=file_path),
        node_labels=["Decision"],
    )
```

The router should expose one read endpoint, `/query/context-graph`, and execute this request directly through `graph.query(...)`.

Target flow:

```text
HTTP / MCP / CLI
    |
    v
application/use_cases/query_context_graph.py
    |
    v
ContextGraphPort.query(...)
    |
    v
GraphitiContextGraphAdapter
```

This gives us one place for:

- `as_of`
- retrieval strategy selection
- invalidation filtering
- source reference enrichment
- provenance enrichment
- conflict annotations
- repo/pot scoping
- result limits
- fallbacks
- normalized result envelopes
- semantic/exact/traversal result merging

## Query Semantics

### Exact First, Semantic Second

For agent-facing context resolution, the resolver should prefer exact canonical reads first:

1. Determine scope.
2. Read canonical graph facts.
3. Use Graphiti semantic search for recall, evidence, and ambiguity.
4. Merge and rank into the agent envelope.

This preserves the current good behavior: semantic search enriches the answer, but exact facts like ownership, decisions, and PR links do not depend on vector recall.

### Search Over One Graph

`semantic_search` should still use Graphiti search, but post-processing should be unified:

- all rows include `source_refs`
- all rows include `episode_uuid` when known
- all rows include `source_node_uuid` / `target_node_uuid` when known
- all rows get canonical labels from endpoint nodes
- open conflicts are attached uniformly
- optional causal expansion uses canonical graph edges in the same adapter

The current `search_pot_context` behavior should move behind `ContextGraphPort.query(...)` with `strategy="semantic"` or `strategy="hybrid"`.

### Structural Reads Become Canonical Reads

Current "structural" query names can stay public, but internally they become canonical graph read patterns.

Examples:

- `change_history`: traverse code asset/file/function to PR entities and decision entities.
- `file_owners`: traverse file bridge edges to PR authors / developers / teams.
- `decisions`: query `Entity:Decision` by scope and active validity.
- `project_graph`: query canonical entity families by include categories.
- `debugging_memory`: query canonical incident/fix/investigation/signal labels.

This is mostly a rename and consolidation, not a storage rewrite.

## Write Path

### Reconciliation Apply

Current:

```python
apply_reconciliation_plan(episodic, structural, plan, ...)
```

Target:

```python
apply_reconciliation_plan(context_graph, plan, ...)
```

New behavior:

1. Validate the plan against `domain/ontology.py`.
2. Build a write command containing episode drafts, canonical mutations, invalidations, and code bridge mutations.
3. Apply it through a future unified write port.
4. Return one mutation result with the primary `episode_uuid` and `MutationSummary`.

### Merged PR Ingest

Merged PR ingestion should stop being a special two-graph path.

Target:

1. Fetch PR bundle.
2. Build a provider-specific planner input.
3. Apply the plan through `ContextGraphPort`.
4. Record ingestion ledger.

The planner must emit generic ontology mutations, not a GitHub compatibility
bundle. Deterministic PR stamping logic should be deleted or reduced to a
planner helper that produces `EntityUpsert`, `EdgeUpsert`, and invalidation
operations.

### Raw Episode Ingest

Raw ingest should keep its current event-first behavior:

1. Persist raw event.
2. Run ingestion agent.
3. Produce episode drafts and canonical mutations.
4. Apply through `ContextGraphPort`.

Only the no-Postgres development fallback should write a raw Graphiti episode directly without canonical reconciliation.

## Application Layer Changes

### New Files

Add:

- `domain/ports/context_graph.py`
- `domain/graph_query.py` update: replace the many-kind internal query model with `ContextGraphQuery`, `ContextGraphScope`, and preset builders
- `adapters/outbound/graphiti/context_graph.py`
- `application/use_cases/query_context_graph.py`

### Update Existing Use Cases

Change these to depend on `ContextGraphPort`:

- `apply_reconciliation_plan.py`
- `apply_episode_step.py`
- `reconcile_event.py`
- `replay_context_event.py`
- `run_ingestion_agent_worker.py`
- `ingest_merged_pr.py`
- `ingest_single_pr.py`
- `backfill_pot.py`
- `run_raw_episode_ingestion.py`
- `hard_reset_pot.py`
- `query_context.py`
- `resolve_context.py` / `context_resolution.py`

Do this incrementally, but do not add new provider-specific write branches while
migrating. Temporary adapters are acceptable only as internal implementation
details on the path to deleting the old application ports.

### Container Wiring

`ContextEngineContainer` should expose:

```python
context_graph: ContextGraphPort
```

Temporary migration properties can remain during migration:

```python
episodic -> context_graph.episodic_compat
structural -> context_graph.structural_compat
```

But new application code should not consume them.

## HTTP, CLI, And MCP Contracts

Keep one graph read route:

- `/query/context-graph`

Remove the old split read routes. Non-query operations remain separate:

- `/ingest`
- `/ingest-pr`
- `/record`
- `/status`
- `/reset`

For MCP, keep:

- `context_resolve`
- `context_search`
- `context_record`
- `context_status`

Do not add tools for the unified graph. The unification is internal.

## Migration Plan

### Phase 0: Freeze Public Behavior And Target Invariants

Before changing internals, add characterization tests for:

- semantic search response fields
- change history
- file owners
- decisions
- PR review context
- project graph
- graph overview
- context resolve envelope
- reset behavior
- merged PR ingest entity/edge effects

These tests should assert response shape, important fields, provenance, and the
absence of source-specific graph apply branches. Do not add new tests that bless
compatibility branches as permanent behavior.

### Phase 1: Add Unified Port And Delegating Adapter

Add `ContextGraphPort` and `GraphitiContextGraphAdapter`.

The adapter can initially delegate:

- episode methods to `GraphitiEpisodicAdapter`
- canonical methods to `Neo4jStructuralAdapter`
- mutation methods to existing structural mutation methods

No behavior change yet.

### Phase 2: Move Reconciliation Apply

Change reconciliation apply paths to receive `ContextGraphPort`.

Keep temporary methods under the adapter only while the application dependency is
being collapsed. They should not be called by application use cases once the
phase is complete.

Success criteria:

- `apply_reconciliation_plan` has one graph dependency.
- `apply_episode_step_for_event` has one graph dependency.
- mutation provenance still includes `episode_uuid`.

### Phase 3: Move Query Use Cases

Introduce `ContextGraphQuery` and preset builders.

Route the single `/query/context-graph` endpoint through `ContextGraphPort.query(...)` and remove endpoint-specific read shapes.

Success criteria:

- there is one internal read method: `query(...)`
- old query kinds are gone from the application/API surface
- temporal filtering is centralized
- conflict annotations are centralized
- source provenance enrichment is centralized
- `HybridGraphIntelligenceProvider` depends on `ContextGraphPort`, not separate episodic/structural ports
- semantic, exact, traversal, and hybrid strategies share one result normalization path

### Phase 4: Move PR Ingest

Convert merged PR ingest and backfill to produce/apply a reconciliation plan through `ContextGraphPort`.

Success criteria:

- PR ingest no longer manually calls `episodic.add_episode` and then a
  provider-specific structural stamping branch in application code.
- PR-specific planning emits generic canonical mutations.
- code bridge writes are explicit graph mutations or bridge mutations.

### Phase 5: Retire Old Ports From Application

Keep `EpisodicGraphPort` and `StructuralGraphPort` only as adapter-internal implementation details, or delete them once test coverage is stable.

Success criteria:

- `application/` imports `ContextGraphPort`, not `EpisodicGraphPort` or `StructuralGraphPort`.
- `bootstrap/container.py` wires one graph port.
- docs and tests use "context graph" terminology instead of "episodic vs structural" for application behavior.

### Phase 6: Graphiti-Native Canonical Writes

Replace direct structural Cypher writes with a Graphiti ontology adapter where Graphiti APIs are reliable enough:

- create/update `EntityNode`
- create/update `EntityEdge`
- direct triplet insertion where appropriate
- temporal invalidation through Graphiti-compatible fields

Keep controlled Cypher for performance-sensitive bridge writes if needed. The application should not care.

## Migration Strategy

Do not attempt a big-bang rewrite.

Use temporary shims only to preserve behavior while removing application
dependencies on the old ports:

```text
ContextGraphPort
    |
    +-- semantic search      -> existing GraphitiEpisodicAdapter
    +-- canonical exact read -> existing Neo4jStructuralAdapter
    +-- mutations            -> generic canonical mutation applier
```

Then pull logic inward behind the unified adapter and delete the old surfaces.

This lets the repo keep shipping while the application dependency graph changes,
but source-specific compatibility branches should shrink on every migration PR.

## Testing Strategy

Use three test layers.

### Unit Tests

Add fake `ContextGraphPort` tests for:

- reconciliation apply ordering
- provenance propagation
- query dispatch
- result normalization
- provenance response shape

### Integration Tests

Extend existing mocked-Neo4j integration tests for:

- search plus canonical labels
- conflict detection
- causal expansion
- graph overview
- code bridge writes
- PR ingest through the unified port

### Lab Harness

Update `scripts/context_engine_lab.py` so `mock-e2e` and `http-e2e` exercise:

- ingest
- search
- resolve
- record
- graph overview
- event details

through the unified port.

## Risks

### Graphiti API Coverage

Graphiti may not expose every low-level write pattern Potpie needs. Keep the adapter allowed to use controlled Neo4j writes where Graphiti APIs are insufficient.

The architecture requirement is one application graph, not "never write Cypher."

### Query Regression

Existing structural reads are exact and useful. Do not replace them with semantic search. Move them behind the unified graph port as canonical reads.

### Code Graph Ownership

The existing code graph is not the same concern as project context memory. Do not duplicate all code nodes as Graphiti entities immediately. Bridge to them, then add `CodeAsset` aliases only where query ergonomics require it.

### Ontology Drift

If Graphiti extraction writes unknown labels or generic edges, those should remain candidates/evidence until reconciliation validates or downgrades them.

### Migration Size

Changing all application use cases at once will be risky. The delegating adapter phase is required.

## Recommended Next Pull Request

The highest-risk PR compatibility write path has been removed, and ingestion
writes now go through `ContextGraphPort.apply_plan()` / `write_raw_episode()`.
Hard reset is also routed through the unified graph port, and the first
`ContextGraphQuery` presets are in place for the old exact-read shapes. The
next implementation PR should keep shrinking direct low-level graph access:

1. Replace the named query handlers with planner objects for exact, traversal,
   temporal, semantic, hybrid, and answer goals.
2. Move remaining intelligence/context consumers off direct `query_context.py`
   imports.
3. Rename or retire bridge-specific ledger fields from the old PR path.
4. Keep low-level Graphiti/Neo4j helpers as adapter internals until they can be
   deleted.

## End State

After migration, application code should read like this:

```text
events -> existing ingestion/reconciliation write path
queries -> context_graph.query(ContextGraphQuery)
resolver -> context_graph.query(...) + source resolvers
```

The user-facing behavior remains:

- context can be ingested
- context can be searched
- task context can be resolved
- exact answers still work
- provenance, temporal filtering, quality, and conflicts remain visible

The difference is that the application layer no longer asks "episodic or structural?" It asks the context graph for facts, evidence, and graph operations, and the Graphiti-backed adapter decides how to execute them.
