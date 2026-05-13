# Architecture (current state)

This describes the Context Engine **as it is today**, not as it will be. The target shape is in [`plan.md`](./plan.md). Where the current system has known drift from that target, this doc flags it inline with `> drift:` callouts that name the phase that closes the gap.

The Context Engine lives at `app/src/context-engine/` and follows a hexagonal layout: `domain/` and `domain/ports/` define types and interfaces, `application/` orchestrates them, and `adapters/inbound/` and `adapters/outbound/` plug concrete implementations into the ports. `bootstrap/` wires everything.

For the agent-facing API surface, see [`agent-contract.md`](./agent-contract.md).
For vision and principles, see [`vision.md`](./vision.md).

---

## High-level shape

```
inbound (HTTP, MCP, CLI, Hatchet worker)
       │
       ▼
application (use_cases + services)
       │
       ▼
domain ports  ◄─────────────────────  domain types
       │
       ▼
outbound adapters (Graphiti, Neo4j, Postgres, GitHub, source resolvers, …)
       │
       ▼
external systems (Neo4j, Postgres, GitHub, Linear, LLMs, Hatchet)
```

Two flows traverse this stack:

- **Write path** — events arrive (HTTP, webhook, backfill, or CLI), are admitted through `IngestionSubmissionService.submit` into a debounced batch in Postgres, the dispatcher claims due batches and runs the reconciliation agent (Phase 4), the agent emits a validated `ReconciliationPlan`, and the plan is applied through **Graphiti as the sole writer** into Neo4j (Phase 1). One pipeline, no sync shortcut.
- **Read path** — agent queries arrive via HTTP/MCP, `ContextResolutionService` plans which evidence families to gather, the `ContextReaderRegistry` routes the request to the registered readers, the source resolver enriches with live source data, and an envelope is returned (Phase 3).

Both flows share the same canonical entity model defined in [`domain/ontology.py`](../../app/src/context-engine/domain/ontology.py).

---

## Domain layer

### Types

The load-bearing domain modules:

- [`domain/ontology.py`](../../app/src/context-engine/domain/ontology.py) — canonical labels, edge types, allowed source/target pairs, required properties, lifecycle/status validation, ontology version. **The catalog is the source of truth; docs link, they do not duplicate.** Every spec declares its own metadata — `project_map_family`, `debugging_family`, `include_keys`, `fact_family`, `source_of_truth`, `freshness_ttl_hours`, `text_patterns`, `property_signatures`, `scope`, `is_activity`, `predicate_family`. Downstream modules (classifier, structural reader, hybrid graph, graph-quality, query helpers) **derive** their lookup tables from the spec at import time — there are no hardcoded label strings anywhere else in the codebase. Adding or renaming an entity is a single-file edit. See [`extending.md`](./extending.md#adding-an-ontology-entity).
- [`domain/source_references.py`](../../app/src/context-engine/domain/source_references.py) — `SourceReferenceRecord`, `FreshnessReport`, `SourceFallback`, freshness TTL policy, source-of-truth policy.
- [`domain/intelligence_models.py`](../../app/src/context-engine/domain/intelligence_models.py) — `ContextResolutionRequest`, `IntelligenceBundle`, all the per-family record types (`ChangeRecord`, `DecisionRecord`, `OwnershipRecord`, `ProjectContextRecord`, `DebuggingMemoryRecord`, `CausalChainItem`), `CoverageReport`, `CapabilitySet`, `ResolutionMeta`.
- [`domain/intelligence_policy.py`](../../app/src/context-engine/domain/intelligence_policy.py) — `EvidencePlan` plus `build_evidence_plan(request, signals, capabilities)` that turns a request into the set of leg-runs the planner will dispatch.
- [`domain/agent_context_port.py`](../../app/src/context-engine/domain/agent_context_port.py) — intent / include / record-type vocabularies, `CONTEXT_RESOLVE_RECIPES`, `bundle_to_agent_envelope()`.
- [`domain/graph_quality.py`](../../app/src/context-engine/domain/graph_quality.py) — `GraphQualityReport`, freshness/quality assessment.
- [`domain/reconciliation.py`](../../app/src/context-engine/domain/reconciliation.py) and [`domain/reconciliation_batch.py`](../../app/src/context-engine/domain/reconciliation_batch.py) — `ReconciliationPlan`, `BatchAgentContext`.
- [`domain/graph_query.py`](../../app/src/context-engine/domain/graph_query.py) — `ContextGraphQuery`, `ContextGraphResult`, the legacy preset builders.
- [`domain/context_reader.py`](../../app/src/context-engine/domain/context_reader.py) — `ReaderCapability`, `ReaderResult`, `ReaderManifestEntry` (Phase 3).
- [`domain/source_resolution.py`](../../app/src/context-engine/domain/source_resolution.py) — `SourceResolutionResult` returned by source resolvers.

### Ports

`domain/ports/` defines the interfaces the application layer talks to. The ones that matter most:

| Port | Role |
|---|---|
| `context_graph.py` | Unified read/write surface (`query`, `apply_plan`, `write_raw_episode`, `reset_pot`). |
| `intelligence_provider.py` | Multi-source evidence aggregation behind `ContextResolutionService`. |
| `reconciliation_agent.py` | Batch-driven agent execution with checkpoint recovery. |
| `ingestion_submission.py`, `ingestion_event_store.py`, `ingestion_event_queue.py` | Event capture, persistence, and debounced queueing. |
| `pot_resolution.py` | Maps repos / external IDs to a pot. |
| `source_connector.py` | One unified contract for every source backend (GitHub, Linear, Notion, …). Subsumed `SourceControlPort` / `IssueTrackerPort` / `LinearIssueFetcher` / `SourceResolverPort` in Phase 2. |
| `context_reader.py` | One unified contract for every read-side evidence family (decisions, change_history, owners, …). Subsumed `GraphQueryPlanner`'s monolithic per-leg dispatch in Phase 3. |
| `pot_source_listing.py` | Host-side per-pot source rows (last sync, sync mode, errors). Distinct from the connector registry's engine-side capability manifest. |
| `reconciliation_ledger.py` | Batch metadata + state machine. |
| `agent_checkpoint_store.py` | Mid-run agent recovery (message history, tool calls). |
| `policy.py`, `telemetry.py` | Centralized authorization (Phase 5) and cost/drift telemetry. |
| `settings.py`, `context_graph_job_queue.py` | Configuration and background-job dispatch. |

---

## Application layer

### Use cases (`application/use_cases/`)

After Phase 4 the use-case set is the canonical agent-facing verb list:

- `submit_raw_episode.py` — raw Graphiti episode submission (HTTP `/ingest`); admits through the same async pipeline when a session is present, falls back to a direct episodic write for standalone CLI use without Postgres. Authorization is the caller's responsibility — HTTP routes call `PolicyPort.authorize` once before invoking the verb.
- `record_durable_context.py` — durable record submission (HTTP `/record`, MCP `context_record`); shapes the request from a `DurableContextPayload` and submits via the ingestion service.
- `report_status.py` — status assembly (HTTP `/status`, MCP `context_status`); composes connector + reader manifests, ledger health, conflict probes, the recommended recipe.
- `resolve_context.py` — read entrypoint (HTTP `/query/context-graph`, MCP `context_resolve` / `context_search`).
- `process_batch.py` — worker entrypoint: claim batch → run agent → mark events processed.
- `dispatch_due_batches.py` — beat-driven sweep that claims due batches and runs `process_batch` per batch in its own session.
- `backfill_pot.py` — connector-driven enumerate-then-submit backfill. Walks list-capable connectors and submits one event per artifact through the standard async pipeline; the agent handles each event during batch processing.
- `hard_reset_pot.py` — operator: clear ledger rows then reset the graph.
- `context_graph_jobs.py` — thin job-runner shims (`handle_backfill_pot`, `handle_dispatch_due_batches`) called by Hatchet/Celery worker adapters; rebuild containers per session.

The canonical "submit event" inbound is the `IngestionSubmissionService.submit(IngestionSubmissionRequest)` method on the service layer (see below). HTTP `/events/reconcile`, the GitHub webhook, and `backfill_pot` all call it. There is exactly one ingestion path post-Phase-4.

### Services (`application/services/`)

- [`context_resolution.py`](../../app/src/context-engine/application/services/context_resolution.py) — fuses `IntelligenceProvider` calls, applies source policy, dedupes, ranks, filters. Builds the `IntelligenceBundle`.
- [`context_reader_registry.py`](../../app/src/context-engine/application/services/context_reader_registry.py) — Phase 3 router. Resolves the family set from `(include, intent, goal, strategy, scope)`, dispatches each family to its registered `ContextReader`, merges the results into a single `ContextGraphResult`. The application layer never branches on family names.
- [`source_connector_registry.py`](../../app/src/context-engine/application/services/source_connector_registry.py) — Phase 2 registry of `SourceConnector`s.
- [`ingestion_submission_service.py`](../../app/src/context-engine/application/services/ingestion_submission_service.py) — `DefaultIngestionSubmissionService` admits events through the debounced batch queue. Constructed with ports only (no session held); the container's `ingestion_submission(session)` factory binds session-scoped adapters and hands them to the service.
- [`event_admission.py`](../../app/src/context-engine/application/services/event_admission.py) — ledger primitive used by `DefaultIngestionSubmissionService`: append the event row, mark queued, upsert the open batch.
- [`reconciliation_validation.py`](../../app/src/context-engine/application/services/reconciliation_validation.py) — validate a `ReconciliationPlan` before deterministic apply; soft / strict ontology coercion lives here.
- [`ingestion_wait.py`](../../app/src/context-engine/application/services/ingestion_wait.py) — poll the event store until an event reaches a terminal state (used by `submit(sync=True)` and `submit(wait=True)`).
- `temporal_search.py` — helper service used by query helpers.

---

## Inbound adapters

### HTTP — `adapters/inbound/http/`

Routes under `/api/v1/context/`:

| Path | Purpose |
|---|---|
| `POST /query/context-graph` | Unified read surface. `goal` ∈ `{answer, retrieve}`, `strategy` ∈ `{hybrid, semantic, auto}`. Backs both `context_resolve` and `context_search`. |
| `POST /record` | Backs `context_record`. Calls `record_durable_context`. |
| `POST /status` | Backs `context_status`. Calls `report_status`. |
| `POST /events/reconcile` | Canonical event submission. Calls `IngestionSubmissionService.submit`. |
| `POST /ingest` | Raw episode submission. Calls `submit_raw_episode`. |
| `GET /events/{id}`, `GET /events` | Event ledger reads. |
| `POST /...` (operator-scoped) | Hard reset, conflict listing/resolve, MODIFIED-edge reclassification. |

Webhooks live under `adapters/inbound/http/webhooks/integrations/`. The GitHub webhook is a thin transport shell: signature validation, repo→pot mapping (env var `CONTEXT_ENGINE_REPO_TO_POT`), then it calls `registry.find_for_webhook("github").normalize_webhook(payload, headers)` and submits the returned `ContextEvent` through the standard async pipeline (`ingestion_submission(session).submit(req)`). Adding a new source's webhook is a connector-side change.

### MCP — `adapters/inbound/mcp/server.py`

FastMCP server exposing exactly the four tools: `context_resolve`, `context_search`, `context_record`, `context_status`. Each tool calls `PotpieContextApiClient`, which is an HTTP proxy back into the engine. The tools live behind a `assert_mcp_pot_allowed` guard.

> **drift:** MCP-as-HTTP-proxy is a deployment-shape choice that isn't load-bearing in the architecture. The contract is what matters; the transport can change.

### CLI — `adapters/inbound/cli/`

`main.py` is large (~1500 lines) and exposes ingest, query, and maintenance commands. `dispatcher_loop.py` polls batches and dispatches them to Hatchet. There's a credentials store with legacy-config migration support.

### Hatchet worker — `adapters/inbound/hatchet/worker.py`

One job: `reconciliation_context_graph_job`. Calls into `application/use_cases/context_graph_jobs.py`, which triggers `process_batch` for due batches.

---

## Outbound adapters

### Graphiti — `adapters/outbound/graphiti/`

The episodic + temporal graph layer **and** the canonical-mutation layer. After Phase 1, every write into Neo4j goes through this directory.

- `context_graph.py` — implements `ContextGraphPort`. Read calls delegate to the `ContextReaderRegistry` (Phase 3); write calls go through `apply_reconciliation_plan`. The `goal=ANSWER` path stays in this adapter because it composes `resolve_context` + the answer synthesizer rather than running readers.
- `episodic.py` — episode ingest via `Graphiti.add_episode`, provenance tracking, temporal supersede orchestration. Phase 1 added the four canonical mutation methods (`apply_entity_upserts`, `apply_edge_upserts`, `apply_edge_deletes`, `apply_invalidations`) which delegate to `canonical_writer.py` over Graphiti's driver.
- `canonical_writer.py` — the Cypher MERGE patterns for canonical entity / edge / invalidation mutations, async, run through the same Graphiti driver. This is where the old `neo4j/structural.py` writes moved.
- `query_helpers.py` — semantic search, timeline, change history, owners, decisions, PR diff/review context, project graph aggregators. Each helper is the body of one reader under `adapters/outbound/readers/`; helpers stay in this module so readers remain composable.
- `apply_plan.py`, `apply_episode_provenance.py` — write reconciliation mutations through the episodic port.
- `family_conflict_detection.py`, `temporal_supersede.py` — conflict resolution on inserts.
- `classify_modified_edges.py` — mark stale edges after mutations.

#### What Graphiti's API gets for each canonical mutation

| Canonical mutation | Lowering |
|---|---|
| `EpisodeDraft` | `Graphiti.add_episode(group_id=pot_id, ...)` (built-in extraction + embeddings) |
| `EntityUpsert` | `MERGE (e:Entity {group_id, entity_key}) SET e += $props` over `g.driver` (identity = `(group_id, entity_key)`, not Graphiti's UUID) |
| `EdgeUpsert` | `MERGE (a)-[r:TYPE]->(b) SET r += $props` over `g.driver` (typed edge labels: `:OWNS`, `:IMPLEMENTS`, …) |
| `EdgeDelete` | Hard delete with `prov_deleted_*` audit fields stamped pre-delete |
| `InvalidationOp` | `SET e.valid_to = ... ` plus optional `MERGE (new)-[:SUPERSEDES]->(old)` |
| `reset_pot` | `clear_data` sweep on Graphiti's driver via `EpisodicGraphPort.reset_pot` |

Identity stays on `(group_id, entity_key)` and edge labels stay typed because that is Potpie's ontology choice; `EntityNode.save` / `EntityEdge.save` would force a schema migration to UUID-keyed entities and `RELATES_TO`-typed edges for no Phase-1 benefit. `add_triplet` and `fact_triple` are explicitly deferred until Phase 5 telemetry says they pay for themselves.

### Neo4j (direct read) — `adapters/outbound/neo4j/`

- `structural.py` — direct Cypher **reads** against Neo4j (change history, timeline, decisions, file owners, PR review context, project graph, debugging memory, graph overview, causal expansion). Implements `StructuralReadPort`.

The mutation methods (`upsert_entities`, `upsert_edges`, `delete_edges`, `apply_invalidations`, `reset_pot`) and the `mutation_applier.py` bridge were deleted in Phase 1; this module is now read-only. Phase 3 keeps it as the structural read substrate; readers consume it through the helper functions in `graphiti/query_helpers.py`.

### Reconciliation — `adapters/outbound/reconciliation/`

- `context_graph_tools.py` — query/mutation tools the agent uses during reconciliation.
- `timeline_plan.py` — the only generic plan helper (Activity + Period subgraph). Source-agnostic; per-connector plan compilers reuse it.
- `pydantic_deep_agent.py`, `null_agent.py`, `noop_agent.py` — agent implementations.
- Generic helpers: validation, ontology mapping, agent prompt construction.

The per-source plan compilers (GitHub PR merged, Linear issue events) moved into the connector packages in Phase 2 — see "Source connectors" below.

### Source connectors — `adapters/outbound/connectors/`

Phase 2 collapsed five source-shaped surfaces (`SourceControlPort`, `IssueTrackerPort`, `LinearIssueFetcher`, `SourceResolverPort`, the per-source webhook normalizers, and the per-source plan compilers) into one `SourceConnectorPort` and a `SourceConnectorRegistry`.

```
adapters/outbound/connectors/
├── github/      # GitHubConnector + PyGithub client + PR resolver + PR plan compiler + webhook + agent tools + review-thread grouper
├── linear/      # LinearConnector + LinearIssueResolver + Linear plan compiler + webhook + Linear event types + LinearIssueFetcher protocol
└── notion/      # NotionConnector — Phase 2 third-source smoke test (passive, fetch + propose_plan only)
```

Every connector implements the same five verbs:

- `kind()` — stable identifier.
- `capabilities()` — `(provider, source_kind, policies)` × per-verb flags (`fetch_capable`, `list_capable`, `webhook_capable`, `plan_capable`, `sync_capable`).
- `list_artifacts(scope)` — enumerate refs in scope (used by backfill).
- `normalize_webhook(payload, headers)` — turn a raw webhook into a `ContextEvent` (or raise on signature failure).
- `fetch(refs, policy, …)` — resolve refs to summaries / snippets / verifications.
- `propose_plan(event, context_graph)` — produce a deterministic `ReconciliationPlan`, or `None` for passive connectors that only contribute resolution.

The application layer never imports a concrete connector. Webhook routing, source resolution, plan proposal, and the per-pot status manifest all dispatch through `application/services/source_connector_registry.py`. Adding a new source means writing one connector module and registering it in `bootstrap/container.py` — see [`extending.md`](./extending.md).

### Context readers — `adapters/outbound/readers/`

Phase 3 collapsed `GraphQueryPlanner`'s monolithic per-leg dispatch into a `ContextReader` registry. Each evidence family is one module that implements `ContextReaderPort` and is registered with the `ContextReaderRegistry` at container build time.

```
adapters/outbound/readers/
├── semantic_search.py      # vector search across the pot's episodic memory
├── change_history.py       # file/function/PR-anchored temporal change rows
├── timeline.py             # actor/feature/branch pulse over a time window
├── owners.py               # inferred reviewers/owners for a file
├── decisions.py            # durable decisions captured against this pot
├── pr_review_context.py    # PR title + summary + grouped review threads
├── pr_diff.py              # PR diff rows (compat — full diffs go through source resolvers)
├── project_graph.py        # bounded neighbourhood traversal anchored on scope
├── graph_overview.py       # schema/edge coverage and ontology drift signal
└── release_notes.py        # Phase 3 third-reader smoke test (release-relevant PRs)
```

Every reader implements the same three verbs:

- `family()` — stable evidence-family key (e.g. `"decisions"`).
- `capability()` — `ReaderCapability` describing intents, required scope fields, cost label, backend, and whether the reader is `compat`-only.
- `read(request)` — execute and return a `ReaderResult` with `result`, `count`, optional `error` / `fallback_reason` / `compat`.

Routing is declarative. The registry consults `capability()` for required scope and intents; the request's `(goal, strategy, include, query)` resolve to a family set; missing readers and missing scope become `RouterFallback` entries surfaced in `meta.fallbacks`. The application layer never branches on family names. Adding a new evidence family means writing one reader file and adding one `register()` call — see [`extending.md`](./extending.md).

The reader manifest is surfaced through `context_status.readers` so agents can ask "what evidence families does this pot expose."

### Intelligence — `adapters/outbound/intelligence/`

`hybrid_graph.py` — `HybridGraphIntelligenceProvider`. Fuses Graphiti episodic queries with Neo4j structural reads behind one `IntelligenceProvider`. Returns the per-family records that go into `IntelligenceBundle`. `mock.py` is the deterministic provider used by the lab and benchmarks.

### Postgres — `adapters/outbound/postgres/`

Thin SQLAlchemy wrappers for the ledgers: `context_events`, reconciliation runs, batches, event store, agent checkpoints, work events.

### Other outbound

- `synthesis/` — LLM answer synthesizer (PydanticAI) plus a `NullAnswerSynthesizer` fallback.
- `agent_tools/` — generic tool builders (sandbox, …); source-specific agent tools live under `connectors/<source>/agent_tools.py`.
- `policy/` — `DefaultPolicyAdapter`: in-process implementation of `PolicyPort` (Phase 5).
- `http/` — outbound HTTP client used by MCP tools and benchmarks to call back into the engine.
- `hatchet/` — Hatchet client for job dispatch.

---

## Bootstrap and wiring

`bootstrap/container.py` defines `ContextEngineContainer`, a dataclass that holds every wired adapter (episodic, structural, intelligence provider, resolution service, reconciliation agent, pots, source resolver, source control, settings, …). `build_container()` constructs the full graph; lazy factory methods (`ingestion_submission()`, `ledger()`, …) build per-request adapters with a database session.

A separate `bootstrap/standalone_container.py` wires a CLI/standalone variant (no FastAPI, no DB session injection).

`_build_answer_synthesizer()` conditionally wires a PydanticAI synthesizer based on env vars. `_attach_reconciliation_context()` couples the reconciliation agent to its tool surface.

---

## The ingestion path

Phase 4 collapsed the dual sync / async pipeline into one. Every event —
HTTP `/events/reconcile`, the GitHub webhook, `record_durable_context`,
`backfill_pot`, `submit_raw_episode` — flows through the same debounced
batch queue and the same reconciliation agent.

```
caller (HTTP / webhook / CLI / backfill)
   │
   ▼
IngestionSubmissionService.submit(IngestionSubmissionRequest)
   │
   ▼
event_admission.admit_event()  →  ContextEvent row in Postgres
                                  +  open debounced batch updated
   │
   ▼   [debounce window elapses]
   │
   ▼
dispatch_due_batches  (Celery beat tick, or Hatchet, or CLI dispatcher)
   │   claims due batches with FOR UPDATE SKIP LOCKED
   ▼
process_batch  (one session per batch)
   │   loads events + checkpoint  →  reconciliation agent  →  validated plan
   ▼
reconciliation_validation.validate_reconciliation_plan
   │   ontology validation, soft / strict downgrade
   ▼
ContextGraphPort.apply_plan
   │
   ▼
adapters/outbound/graphiti/apply_plan.py
   │
   ▼
Neo4j (via Graphiti)
```

Connector-proposed plans are produced by the agent's tool surface during
batch processing — `SourceConnector.propose_plan(event, context_graph)`
is callable from inside the reconciliation agent for events whose source
ships a deterministic compiler (e.g. GitHub merged-PR, Linear issue).
Inbound submission never calls `propose_plan`; the agent decides whether
to defer to a deterministic plan or reason from scratch.

---

## The read path

```
caller → POST /api/v1/context/query/context-graph
       → application/use_cases/resolve_context.py
       → application/services/context_resolution.py (ContextResolutionService.resolve)
       → domain/intelligence_policy.py (build_evidence_plan)
       → adapters/outbound/intelligence/hybrid_graph.py (HybridGraphIntelligenceProvider)
       → in parallel:
            • Graphiti episodic — adapters/outbound/graphiti/query_helpers.py legs
            • Structural — adapters/outbound/neo4j/structural.py Cypher reads
       → application/services/source_connector_registry.py (per source-policy fan-out)
       → adapters/outbound/synthesis (LLM answer synthesizer, optional)
       → bundle_to_agent_envelope() → response
```

For the read primitives (any non-`ANSWER` `ContextGraphQuery`), the path collapses to:

```
caller → POST /api/v1/context/query/context-graph
       → adapters/outbound/graphiti/context_graph.py::query
       → application/services/context_reader_registry.py::execute
       → registered ContextReader.read() per resolved family
       → merged ContextGraphResult
```

The reader catalogue is open: `semantic_search`, `change_history`, `timeline`, `owners`, `decisions`, `pr_review_context`, `pr_diff`, `project_graph`, `graph_overview`, `release_notes` (Phase 3 smoke test). Adding a new family means writing one reader file and adding one `register()` call — see [`extending.md`](./extending.md).

---

## What Graphiti owns vs what Potpie owns

This split is the load-bearing reason Graphiti is a substrate, not a black box.

**Graphiti owns:**
- Episode persistence and episode-to-entity provenance.
- Temporal edge semantics (`valid_at` / `invalid_at` event time, `created_at` / `expired_at` system time).
- Embeddings + semantic retrieval.
- Generic entity / fact primitives (`EntityNode`, `EntityEdge`, `add_episode`, `add_triplet`).
- Namespace isolation via `group_id` (Potpie maps `group_id == pot_id`).
- The Neo4j connection — there is exactly one driver, owned by the Graphiti client, and every Context Engine write is run through it (Phase 1).

**Potpie owns:**
- Deterministic `entity_key` generation per canonical type.
- Label and edge validation against the ontology catalog.
- Allowed-edge / required-property enforcement.
- Canonical mutation lowering (`adapters/outbound/graphiti/canonical_writer.py`) — the Cypher patterns that MERGE on `(group_id, entity_key)` and on typed edge labels.
- Conflict and supersession handling at the canonical layer.
- Exact-read helpers that return canonical shapes (not raw nodes).
- Source-reference-first storage discipline.
- The four-tool agent contract.

If Graphiti cannot express a canonical mutation, the gap is named explicitly and lowered through Graphiti's driver inside `adapters/outbound/graphiti/`. We do not work around it with a parallel store. The previous `neo4j/structural.py` writer was the parallel-store anti-pattern; Phase 1 retired it.

---

## Operability and observability

Phase 5 introduced one centralized `PolicyPort`, structured cost telemetry across every engine LLM call site, and per-pot drift signals surfaced in the resolve envelope.

### PolicyPort

[`domain/ports/policy.py`](../../app/src/context-engine/domain/ports/policy.py) defines the single decision call site. `PolicyPort.authorize(actor, resource, action, context) -> PolicyDecision` is enforced at two boundaries:

- **Every HTTP route** in [`adapters/inbound/http/api/v1/context/router.py`](../../app/src/context-engine/adapters/inbound/http/api/v1/context/router.py) calls `_enforce(...)` before doing any work. MCP and CLI talk to the engine through HTTP, so HTTP `_enforce` covers them transitively. The MCP server adds a pre-flight `assert_mcp_pot_allowed` allowlist as a client-side tenant boundary; it is not a substitute for the server-side policy decision.
- **The apply-write boundary** in [`application/use_cases/process_batch.py`](../../app/src/context-engine/application/use_cases/process_batch.py) authorizes `apply.write` once per batch before invoking the reconciliation agent — one decision covers every mutation the agent issues in that batch.

The default in-process implementation lives in [`adapters/outbound/policy/default.py`](../../app/src/context-engine/adapters/outbound/policy/default.py) and folds in:

- the engine on/off settings gate (`CONTEXT_GRAPH_ENABLED`),
- the reconciliation feature flags (`CONTEXT_ENGINE_RECONCILIATION_ENABLED`, `CONTEXT_ENGINE_AGENT_PLANNER_ENABLED`),
- pot-access resolution via `PotResolutionPort`,
- maintenance-write flags (`CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES`, `CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE`).

`PolicyDecision` carries a stable `reason` taxonomy (`context_graph_disabled`, `reconciliation_disabled`, `agent_planner_disabled`, `unknown_pot`, `forbidden`, `maintenance_write_disabled`, …) and an HTTP `status_code` so transports translate without secondary lookups. Routes call `_enforce(container, ...)` once per request; the prior chains of `is_enabled() / agent_planner_enabled() / pots.resolve_pot()` are gone.

Resource taxonomy: `pot.{read, submit_event, record, ingest_episode, reset, resolve_conflict, maintenance}`, `connector.{list, fetch}`, `apply.write`. New gates are one branch in the default adapter — no route or use-case edit needed.

### Cost telemetry

[`domain/ports/telemetry.py`](../../app/src/context-engine/domain/ports/telemetry.py) defines `TelemetryPort` plus two payloads: `CostEvent` (per LLM call) and `DriftSnapshot` (per resolve). `NoOpTelemetry` is the default; the Postgres adapter at [`adapters/outbound/postgres/telemetry.py`](../../app/src/context-engine/adapters/outbound/postgres/telemetry.py) writes to two append-only tables:

- `context_engine_cost_events` — one row per LLM call: `pot_id, kind, model, input_tokens, output_tokens, total_tokens, latency_ms, batch_id?, event_id?, occurred_at`. `kind` is one of `agent | synthesis | graphiti_extract | connector`.
- `context_engine_drift_snapshots` — one row per resolve: `pot_id, status, source_ref_count, stale_ref_count, needs_verification_ref_count, verification_failed_ref_count, source_access_gap_count, missing_coverage_count, fallback_count, open_conflicts_count, captured_at`.

Cost events are emitted from:

- `PydanticDeepReconciliationAgent.run_batch` after the agent loop (uses `result.usage()` from pydantic-ai),
- `PydanticAIAnswerSynthesizer.synthesize` after the answer call.

Drift snapshots are emitted from `ContextResolutionService.resolve` after the post-fetch quality assessment, using the metrics already computed by `assess_graph_quality` plus `len(open_conflicts)`. The adapter is best-effort: failures log at WARNING and never propagate; telemetry never fails the request.

`DATABASE_URL` controls auto-wiring. With it set, `bootstrap.container.build_container` plugs in `SqlAlchemyTelemetry`; without it, `NoOpTelemetry`. `CONTEXT_ENGINE_TELEMETRY=0` disables emission at runtime.

### Resolve envelope cost + drift

The agent envelope (`bundle_to_agent_envelope`) carries:

- `meta.cost.resolve_ms` — total resolve wall time.
- `meta.cost.per_call_latency_ms` — already-existing per-leg latency map.
- `meta.cost.synthesis` — `{model, input_tokens, output_tokens, total_tokens, latency_ms}` when the synthesizer ran.
- `quality.drift` — `{status, signals: {stale_refs, needs_verification_refs, verification_failed_refs, source_access_gaps, missing_coverage, fallbacks, open_conflicts}, thresholds}`.

The `quality.drift.status` taxonomy is the same `good | watch | degraded | unknown` produced by `assess_graph_quality`, surfaced under one explicit "drift" key so an agent can decide "verify before acting" from a single field.

### Per-pot dashboards

Both telemetry tables are indexed `(pot_id, ...)` and timestamped, so pot-level cost and drift queries are direct SQL:

```sql
SELECT pot_id, kind, sum(total_tokens), avg(latency_ms)
FROM context_engine_cost_events
WHERE occurred_at > now() - interval '1 day'
GROUP BY pot_id, kind;

SELECT pot_id, status, max(captured_at), max(stale_ref_count)
FROM context_engine_drift_snapshots
WHERE captured_at > now() - interval '7 days'
GROUP BY pot_id, status;
```

These tables are the queryable surface; richer dashboards are downstream of them.

---

## Dev-only directories

Not part of the runtime architecture; listed here so they aren't invisible to a reviewer:

- `benchmarks/` — benchmark dataset, runner, evaluator. Used by `scripts/benchmark_context_engine.py`.
- `scripts/` — `benchmark_context_engine.py`, `context_engine_lab.py`, mock data. CLI utilities for development.
- `tests/` — `unit/`, `integration/`, `fixtures/`. Pytest suite.

---

## How this doc stays correct

Every later phase in `plan.md` ends with "docs to update," and `architecture.md` is on every list. If a section here describes something the code no longer does, the failing phase is the one that should have updated it. Treat doc rot as a phase regression, not a separate task.
