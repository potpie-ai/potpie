# Context-engine: features and capabilities (inventory)

This document summarizes what lives under `app/src/context-engine` today: behaviors, surfaces, and how they fit together. It is meant for orientation and roadmap sanity checks—not as a substitute for the operational details in [`README.md`](README.md) or Potpie host wiring in [`app/modules/context_graph/README.md`](../../modules/context_graph/README.md).

---

## 1. What problem it solves

The **context graph** ties together:

- **Episodic memory** (Graphiti on Neo4j): semantic chunks, episodes, time-bounded validity, search.
- **Structural code graph** (Neo4j): PR-linked entities, files, decisions, diff excerpts, ownership-style hints.
- **Durable ingestion** (Postgres): ledgers, normalized `context_events`, reconciliation runs, per-episode apply steps, and optional plan artifacts.

Consumers (humans, IDE agents, HTTP clients) get **scoped** access by **pot** (`pot_id` / Graphiti `group_id`), with Potpie enforcing **tenancy** for API-key and session flows.

---

## 2. Architecture: hexagonal package

| Layer | Role |
|--------|------|
| **`domain/`** | Pure types, errors, `Protocol` ports. No FastAPI, Typer, SQLAlchemy, Neo4j, or Graphiti imports. |
| **`application/`** | Use cases and orchestration (`services/`). Depends only on domain ports. |
| **`adapters/`** | **Inbound:** HTTP, CLI, MCP, Hatchet worker. **Outbound:** Postgres, Neo4j, Graphiti, GitHub, HTTP client to Potpie, reconciliation agent impls, optional Hatchet queue. |
| **`bootstrap/`** | `ContextEngineContainer` wiring, optional standalone stack, queue factory (`celery` / `hatchet` / `noop`), env-based pot maps. |

**Dependency rule:** inbound → application → domain ← outbound adapters.

---

## 3. Storage and external systems

| System | Use |
|--------|-----|
| **Neo4j** | Structural queries and mutations; also the database Graphiti uses for episodic data (config can point context-engine at a dedicated Neo4j via `CONTEXT_ENGINE_NEO4J_*`). |
| **Graphiti** (`graphiti-core`) | Episodic adapter: add/search episodes, pot-scoped semantic search, invalidation / `as_of` style queries where supported. |
| **Postgres** | Ingestion ledger, `context_events`, reconciliation ledger, plan store, ingestion event store, step status—powers **async** pipelines and observability. |
| **GitHub** (`PyGithub`) | Fetch PRs, diffs, metadata for backfill and merged-PR ingest (`SourceControlPort`). |

Without Postgres, some paths degrade to **legacy** behavior (e.g. raw episode → direct Graphiti write when `sync=true` only).

---

## 4. Pot resolution and tenancy

- **Standalone / env:** `CONTEXT_ENGINE_POTS` (pot → repos), `CONTEXT_ENGINE_REPO_TO_POT` (webhook mapping), etc.
- **Potpie (production):** `PotResolutionPort` is implemented in `app/modules/context_graph/wiring.py` using DB-backed **context pots**, **members**, and **repositories**—independent of “parsing projects” for scope. See the host README for table names and routes.

**CLI/MCP** resolve `pot_id` via stored `pot use`, env maps, or git `origin` → Potpie API (`git_project.py`, `cli_pot_resolution.py`).

---

## 5. How the software is run

### 5.1 Embedded in Potpie (default)

Same FastAPI app; routes mounted from `app/modules/context_graph/context_engine_http.py` and `app/api/router.py`:

- **`/api/v1/context/*`** — session/Firebase auth (Potpie’s existing auth).
- **`/api/v2/context/*`** — **`X-API-Key`** for CLI, MCP, and automation.

The **router implementation** is `adapters/inbound/http/api/v1/context/router.py` (`create_context_router`); paths are the suffix after the mount prefix.

Long-running mutations (sync, ingest-pr when queued) go through **`ContextGraphJobQueuePort`**: **Celery** (`context-graph-etl` queue) by default, or **Hatchet** when configured.

### 5.2 Optional standalone HTTP process

`python -m adapters.inbound.http` with env for Neo4j, Postgres, GitHub token, pot maps, optional `CONTEXT_ENGINE_API_KEY`. Mutations like sync/ingest-pr can run **inline** (no Celery) in this mode—see package README.

### 5.3 CLI (`potpie`)

Entry: `adapters/inbound/cli/main.py` (Typer).

| Command | Purpose |
|---------|---------|
| `login` / `logout` | Store or clear Potpie base URL + API key for v2 calls. |
| `doctor` | Credentials, config snapshot, `GET /health` probe. |
| `init-agent` | Install `AGENTS.md` plus context-engine agent skills, including `potpie-agent-context` recipes over the minimal context port. |
| `pot use` / `unset` / `list` / `pots` / `alias` / `clear-local` | Local pot scope and Potpie-backed pot listing/creation helpers. |
| `pot repo list` / `pot repo add` | Repositories attached to a pot (Potpie API). |
| `pot hard-reset` | Calls reset API for the active/scoped pot. |
| `add` | Register repo ↔ pot mapping context (see CLI README). |
| `search` | POST unified `query/context-graph` to Potpie. |
| `ingest` | Raw episode ingest via Potpie (async default when server has DB; `--sync` for inline). |

Global flags: `--json`, `--verbose`, `--source` (default source label for ingest/search).

### 5.4 MCP (`potpie-mcp`)

Stdio MCP server (`adapters/inbound/mcp/server.py`): tools call **Potpie `POST /api/v2/context`** with the stored API key. **Pot allowlisting** via `CONTEXT_ENGINE_MCP_ALLOWED_POTS` or dev-only `CONTEXT_ENGINE_MCP_TRUST_ALL_POTS`.

Public tools: `context_resolve`, `context_search`, `context_record`, and `context_status`.

`context_resolve` is the primary task context wrap. Feature, debugging, review, operations, docs, and onboarding workflows are expressed through `intent`, `scope`, `include`, `exclude`, `mode`, `source_policy`, `budget`, and `as_of`, not through separate MCP tools per context type. `context_status` returns readiness plus the recommended recipe, `context_search` is for narrow follow-up lookup, and `context_record` captures durable project learnings.

---

## 6. HTTP API surface (router)

All routes below are **relative to the mounted prefix** (`/api/v1/context` or `/api/v2/context` on Potpie).

### 6.1 Mutations

| Method | Path | Behavior |
|--------|------|----------|
| POST | `/sync` | Backfill pots (inline in standalone; **enqueue** when Potpie injects `mutation_handlers`). |
| POST | `/ingest-pr` | Merged PR ingest via `IngestionSubmissionService` (persist + reconcile or queue). |
| POST | `/ingest` | Raw Graphiti episode: event store + **202 queued** vs **200 applied** (`sync` / `X-Context-Ingest-Sync`). |
| POST | `/reset` | Hard reset: Graphiti + structural Neo4j + optional Postgres ledgers. |
| POST | `/events/reconcile` (alias `/events/ingest`) | Normalize **context event** → agent reconciliation (feature-flagged; requires reconciliation agent on container). |
| POST | `/events/replay` | Re-run reconciliation for an existing `context_events` row. |

### 6.2 Event observability

| Method | Path | Behavior |
|--------|------|----------|
| GET | `/events/{event_id}` | Event payload + reconciliation step rows + agent work events for each reconciliation run. |
| GET | `/pots/{pot_id}/events` | Cursor-paginated list with filters (status, `ingestion_kind`). |

### 6.3 Query (`/query/context-graph`)

| Path | Backing | Purpose |
|------|---------|---------|
| `/query/context-graph` | Unified Graphiti-backed context graph | One graph read endpoint for semantic search, exact scoped reads, timelines, neighborhoods, aggregates, and agent-ready context answers. |

Potpie can set **`enforce_pot_access`** so unknown pots return a clear 404 for the caller’s tenancy.

### 6.4 Standalone webhook

`adapters/inbound/http/webhooks/integrations/github.py`: `pull_request` **closed** + **merged** → map repo → pot → submit merged-PR ingestion (requires DB + mapping env).

---

## 7. Ingestion kinds and pipelines

The **`DefaultIngestionSubmissionService`** (`application/services/ingestion_submission_service.py`) is the single front door for persisted ingestion:

1. **`raw_episode`** — raw notes/links from CLI/MCP/HTTP/UI are normalized as raw events, then routed through the Ingestion Agent for plan generation and durable episode steps. The no-Postgres development fallback can still write directly to Graphiti when `sync=true`.
2. **`github_merged_pr`** — wraps `ingest_single_pr` / merged PR core: GitHub fetch, episodic + structural updates, dedupe by event identity.
3. **`agent_reconciliation`** (and related) — `record_context_event` + optional **`run_ingestion_agent_for_event`**: build request from event → **reconciliation agent** produces a **plan** → validated → split into **durable steps** → **`JobEnqueuePort.enqueue_episode_apply`** per step.

**Workers** (Celery/Hatchet) call into `application/use_cases/context_graph_jobs.py`: backfill, ingest-pr handler, ingestion agent runner, episode step applier.

Supporting use cases include: `wait_ingestion_event`, `reconcile_event`, `apply_reconciliation_plan`, `build_reconciliation_request`, `reconciliation_validation`, `split_reconciliation_plan`, `reconciliation_plan_codec`, `context_event_mapping`, `replay_context_event`, `event_reconciliation` (HTTP payload shaping), and the legacy no-DB raw fallback.

---

## 8. Intelligence and “resolve”

- **`HybridGraphIntelligenceProvider`** combines episodic + structural ports for tools used in higher-level reasoning.
- **`ContextResolutionService`** + **`resolve_context`** use case: async HTTP handler returns a **bundle** (coverage, errors, meta) for a natural-language query with optional artifact/scope hints (`domain/intelligence_models.py`).

This is the main **“agent-facing”** aggregated API beyond raw search.

---

## 9. Reconciliation agent (optional extra)

- Extra: `potpie[reconciliation-agent]` → **pydantic-deep** based adapter under `adapters/outbound/reconciliation/` (factory, plan schema, null agent for tests).
- Feature flags: `domain/reconciliation_flags.py` (`CONTEXT_ENGINE_RECONCILIATION_ENABLED`, `CONTEXT_ENGINE_AGENT_PLANNER_ENABLED`, etc.).
- When disabled or agent is `None`, `/events/reconcile` returns **503** with an explicit message.

---

## 10. Job queue abstraction

- **Port:** `domain/ports/context_graph_job_queue.py` + `domain/ports/jobs.py` (`JobEnqueuePort`: backfill, ingest PR, run agent for event, enqueue episode apply).
- **Celery adapter:** lives under Potpie `app/modules/context_graph/celery_job_queue.py` (implements the port).
- **Hatchet:** `adapters/outbound/hatchet/hatchet_job_queue.py`, worker bootstrap `adapters/inbound/hatchet/worker.py`, `bootstrap/queue_factory.py` selects backend via env.

---

## 11. Testing and quality

- All unit tests for this package: `app/src/context-engine/tests/unit/` (pytest).
- Do **not** reintroduce duplicate `tests/unit/context_graph/` at repo root; host README documents this policy.

---

## 12. Are we on the right track? (short opinion)

**Strengths**

- Clear **hexagonal** boundaries: portable core with swappable auth, queue, and pot resolution—appropriate for “library inside Potpie” plus optional standalone.
- **Three-layer graph story** (episodic + structural + Postgres workflow) matches real product needs: search, code-linked history, and **auditable** async ingestion.
- **Single submission service** + **ingestion kinds** reduces drift between HTTP, jobs, and webhooks.
- **v1 session vs v2 API key** split is a pragmatic way to serve humans and automation without duplicating business logic.

**Risks / things to keep watching**

- **Operational complexity:** Graphiti + Neo4j + Postgres + optional Hatchet + feature flags is a lot; teams need a “happy path” doc and minimal env for dev.
- **Two speed modes:** legacy direct Graphiti vs event-store-first async must stay obvious in APIs and CLI output (you already surface `queued` vs `applied` vs `legacy_direct`).
- **Reconciliation agent:** powerful but optional; product should define when it is required vs PR/episodic-only mode.

Overall, the codebase reads as a **coherent platform** for pot-scoped context: ingestion, query, reset, background processing, and agent-oriented resolution—rather than a one-off script. If your roadmap doubles down on **event-sourced ingestion** and **multi-tenant pots**, the current split between `potpie` and Potpie `context_graph` wiring is the right direction.

---

*Generated as a snapshot of the repository layout and primary modules; when behavior changes, update this file or replace it with generated docs from OpenAPI/CLI help.*
