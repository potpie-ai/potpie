# Integrations Architecture — Design & Implementation Plan

**Vision:** Every external tool a team uses — code hosts, issue trackers, design tools, communication platforms — feeds into a unified, correlated context graph per project. Potpie's intelligence layer draws on all of them to answer questions, not just one GitHub repo.

**OSS boundary:** GitHub stays in the open repo. All other provider implementations ship as closed-source plugins. The public repo defines **all interfaces, schemas, orchestration, and the generic UI shell**. A closed package supplies concrete adapters (Linear, Jira, Slack, etc.) and registers them at startup.

---

## 1. Architecture Principles

These are not aspirational — they are load-bearing decisions that every implementation choice must satisfy.

1. **Connections ≠ Sources.** A _connection_ is a user-level OAuth credential to a provider. A _source_ is a project-level attachment that says "this project draws context from this scope of this provider." One connection can back many sources across many projects. Detaching a source does not revoke the connection.

2. **The provider is a plugin, not a conditional branch.** Adding a new provider should never require touching orchestration code, Celery task definitions, or the Sources UI. The registry + port interface + episode contract is the full integration surface. If you have to add an `if provider == "linear"` anywhere in the shared codebase, the abstraction is wrong.

3. **Webhook-first, backfill as recovery.** Every provider should be designed for webhook-driven incremental sync from day one. Backfill (poll + cursor) exists as the catch-up mechanism for initial onboarding and missed webhooks — not as the primary sync mode.

4. **The episode body is the primary contract to the knowledge graph.** Graphiti extracts entities and relationships from narrative text. A provider's job is to produce rich `episode_body` markdown. The shared entity schema steers extraction but does not need per-provider customization for common concepts.

5. **Credentials are encrypted, refreshed, and revocable.** No in-memory token stores. No plaintext JSONB. Application-level envelope encryption from the start. Refresh and revocation are part of the connection lifecycle, not a follow-up.

6. **Cross-provider correlation is a first-class concern.** When a Linear issue references a GitHub PR, the graph should contain an explicit edge — not rely on Graphiti's LLM to notice the mention. Entity keys are namespaced (`github:pr:owner/repo:1234`, `linear:issue:ENG-5678`) and cross-reference edges are written during ingestion.

7. **Context ranking is source-aware.** When a project has 5 sources, `resolve_context` must be able to weight or filter by source type. Code questions prefer code-host episodes; product-decision questions prefer issue-tracker episodes. Freshness matters.

---

## 2. Conceptual Model

### 2.1 Entities

**`Connection`** (existing `integrations` table, tightened + encrypted)

```
connection_id    PK
provider         github | linear | jira | slack | …
user_id          FK → users.uid
auth_data        JSONB (encrypted envelope: access_token, refresh_token, scopes, expires_at)
scope_data       JSONB (org/workspace identifiers granted at OAuth time)
status           active | token_expired | revoked | error
webhook_secret   TEXT (per-connection, used for webhook signature verification)
created_at / updated_at
```

**Note on GitHub:** Today GitHub OAuth lives in `user_auth_providers`, and GitHub App install/fallback is separate. Long-term, GitHub should be migrated into `connections` like every other provider. Until then, the `connections` API returns an **aggregated view** over both tables. The aggregation adapter is an explicit, tested component — not a hidden join.

**`ProjectSource`** (new table: `project_sources`)

```
id               PK (UUID)
project_id       FK → projects.id
connection_id    FK → connections.connection_id (nullable for legacy GitHub rows derived from Project.repo_name)
provider         github | linear | …
source_kind      repository | issue_tracker_project | channel | …
scope_json       JSONB (provider-specific: repo_name, team_ids, labels, channel_id, …)
sync_enabled     BOOLEAN default true
sync_mode        webhook | poll | hybrid (default: hybrid)
webhook_status   pending_setup | active | failed | not_applicable
last_sync_at     TIMESTAMP
last_error       TEXT
health_score     INT (0–100, computed from consecutive success/failure)
created_at / updated_at
```

**Unique constraint:** `(project_id, provider, source_kind, scope_hash)` where `scope_hash` is a deterministic hash of `scope_json` — prevents duplicate attachments.

**`ProviderDefinition`** (runtime registry, not DB)

```python
@dataclass
class ProviderDefinition:
    id: str                          # "github", "linear"
    display_name: str
    capabilities: list[str]          # ["code_host"], ["issue_tracker"], ["code_host", "issue_tracker"]
    source_kinds: list[str]          # ["repository"], ["issue_tracker_project"]
    port_type: type                  # SourceControlPort, IssueTrackerPort, etc.
    adapter_factory: Callable        # (connection) → port instance
    episode_formatter: Callable      # raw data → episode dict
    ingest_use_case: Callable        # (ledger, episodic, structural, scope, data) → IngestionResult
    backfill_handler: Callable       # (source, ledger, episodic, structural, settings) → result
    webhook_handler: Callable | None # (payload, headers) → list of ingest tasks
    oauth_config: OAuthConfig | None
    refresh_interval: timedelta | None
    oss_available: bool
```

### 2.2 Provider Registry

```python
class ProviderRegistry:
    _providers: dict[str, ProviderDefinition]

    def register(self, defn: ProviderDefinition) -> None: ...
    def get(self, provider_id: str) -> ProviderDefinition | None: ...
    def list_available(self) -> list[ProviderDefinition]: ...
    def get_backfill_handler(self, provider: str) -> Callable: ...
    def get_webhook_handler(self, provider: str) -> Callable | None: ...
    def get_adapter(self, provider: str, connection) -> Any: ...
```

OSS registers `github`. The closed package calls `registry.register(linear_definition)` at import time. A single bootstrap entry point (`load_providers()`) is called from **both** API startup and Celery worker startup — this is not optional, it is required for correctness.

### 2.3 Port hierarchy

```
SourceControlPort           (existing — GitHub)
    get_pull_request, iter_closed_pulls, get_issue, …

IssueTrackerPort            (new — Linear, Jira)
    iter_issues, get_issue, get_comments, get_state_history, …

CommunicationPort           (future — Slack, Discord)
    iter_messages, get_thread, …

DesignToolPort              (future — Figma, etc.)
    ...
```

Each port is a Protocol. Providers implement exactly one. The `ProviderDefinition.port_type` field identifies which. Orchestration dispatches by port type, not by provider name.

---

## 3. Sync Architecture

### 3.1 Overview

```
┌──────────────────────────────────┐
│         Webhook Router           │  POST /api/v1/sources/webhooks/{provider}
│  (signature verify → identify    │
│   ProjectSource → enqueue task)  │
└──────────┬───────────────────────┘
           │                         ┌─────────────────────────────┐
           │  incremental            │    Periodic Backfill        │
           ▼                         │  (Celery beat → per-source  │
┌──────────────────────────┐        │   cursor-based catch-up)    │
│    Sync Orchestrator     │◄───────┘
│                          │
│  For each ProjectSource: │
│   1. registry.get(provider)
│   2. adapter = registry.get_adapter(provider, connection)
│   3. handler = registry.get_backfill_handler(provider)
│      OR ingest specific artifact (webhook path)
│   4. handler(adapter, ledger, episodic, structural, ...)
│                          │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│    Ingestion Pipeline    │
│                          │
│  episode_formatter(raw)  │ → episode dict (5 keys)
│  episodic.add_episode()  │ → Graphiti (LLM extraction → entities/edges)
│  structural.stamp_*()    │ → Neo4j structural layer (optional per provider)
│  cross_ref.link()        │ → cross-provider correlation edges
│  ledger.append()         │ → Postgres: raw_events + ingestion_log + bridge_status
│                          │
└──────────────────────────┘
```

### 3.2 Celery queue design

| Queue | Purpose | Concurrency |
|-------|---------|-------------|
| `context-graph-etl` | GitHub backfill/ingest (existing) | Default |
| `context-graph-{provider}` | Per-provider queues for commercial providers | Rate-limited per provider |
| `context-graph-webhooks` | Webhook-triggered incremental ingestion (all providers) | High priority, fast |
| `context-graph-maintenance` | Token refresh, health checks, stale-source cleanup | Low priority, periodic |

Workers subscribe to queues based on what providers they have registered. OSS workers only see `context-graph-etl` + `context-graph-webhooks`. Commercial workers add provider-specific queues.

### 3.3 Backpressure

- Monitor Graphiti adapter latency (LLM extraction is the bottleneck).
- If queue depth for any provider exceeds a configurable threshold, pause that provider's backfill tasks (skip-and-requeue with delay).
- Webhook-triggered ingest always proceeds (it's one artifact, not a batch).

---

## 4. Graphiti Integration Contract

### 4.1 Episode contract

Every provider must produce a `dict[str, Any]` with exactly these keys:

| Key | Type | Purpose |
|-----|------|---------|
| `name` | `str` | Unique identifier for the episode (e.g., `pr_1234_merged`, `linear_issue_ENG-5678_updated`) |
| `episode_body` | `str` | Rich narrative markdown — this is what Graphiti's LLM extracts entities/edges from |
| `source_description` | `str` | Human-readable source label (e.g., `GitHub Pull Request #1234`) |
| `source_id` | `str` | Stable dedup key for the ledger |
| `reference_time` | `datetime` | When the artifact was last meaningful (merged_at, updatedAt, etc.) |

### 4.2 Entity schema governance

The shared schema in `entity_schema.py` defines what Graphiti looks for:

- **Core types (provider-agnostic, do not duplicate per provider):** `PullRequest`, `Commit`, `Issue`, `Feature`, `Decision`, `Developer`
- **Edge types:** `Modified`, `Fixes`, `PartOfFeature`, `MadeIn`, `AuthoredBy`, `Owns`
- **Extension rules:**
  - New types (e.g., `Sprint`, `Deployment`, `Incident`) require measured evidence that extraction quality improves.
  - Plugins can register additional types via `registry.register_entity_types(...)` — merged at startup, validated for no conflicts.
  - Core types are versioned. Breaking changes (rename/remove) require a graph migration plan.

### 4.3 Cross-provider correlation

During ingestion, each `ingest_*` use case should:

1. Extract cross-references from the episode body (regex for GitHub PR URLs in Linear issues, Linear issue IDs in PR descriptions).
2. If a referenced entity's `entity_key` exists in the structural graph, create a `REFERENCES` edge.
3. Use namespaced entity keys: `github:pr:{owner}/{repo}:{number}`, `linear:issue:{identifier}`, etc.

This is deterministic (no LLM needed) and more reliable than hoping Graphiti's extraction catches cross-provider mentions.

### 4.4 Context ranking

`EpisodicGraphPort.search` and `resolve_context` need:

- `source_types: list[str] | None` filter — pass through to Graphiti or post-filter.
- Freshness signal — `ProjectSource.last_sync_at` gates whether episodes from that source are considered stale.
- The intelligence bundle's family resolver can hint which source types are relevant per query category (code questions → `code_host`; product questions → `issue_tracker`).

---

## 5. Credential Lifecycle

### 5.1 Storage

`auth_data` in the `connections` table uses **application-level envelope encryption**:

```python
{
    "encrypted_blob": "base64...",
    "key_id": "kms-key-2024-01",
    "algorithm": "AES-256-GCM"
}
```

A `CredentialVault` service handles encrypt/decrypt. The KMS key can be a local symmetric key for self-hosted or a cloud KMS reference for SaaS.

### 5.2 Refresh

- Each `ProviderDefinition` declares `refresh_interval` (e.g., Linear tokens expire in 10 hours).
- A Celery beat task (`refresh_expiring_tokens`) runs periodically, queries connections where `token_expires_at < now() + buffer`, and calls the provider's refresh flow.
- On refresh failure after N retries: set `connection.status = token_expired`, disable all linked `ProjectSource` rows, notify user.

### 5.3 Revocation

When a user disconnects (DELETE `/api/v1/sources/connections/{id}`):
1. Call provider's token revocation API if available.
2. Set `connection.status = revoked`.
3. Cascade: all `ProjectSource` rows with this `connection_id` → `sync_enabled = false`, `last_error = "connection revoked"`.
4. Purge encrypted `auth_data`.

### 5.4 Scope escalation

If a provider requires broader scopes (e.g., write access added later):
- The OAuth flow is re-initiated with the new scope set.
- On success, `auth_data` is replaced (not appended) and `scope_data` is updated.
- Existing `ProjectSource` rows are unaffected — the connection's credentials just got broader.

---

## 6. Implementation Phases

These are ordered by architectural dependency, not by "MVP-ness." Each phase builds permanent infrastructure that all subsequent work depends on. Nothing here is throwaway.

### Phase 1 — Provider registry + port interfaces (OSS)

**What:** Build the `ProviderRegistry`, `ProviderDefinition`, `IssueTrackerPort`, and the bootstrap loading mechanism. Register GitHub as the first provider. No behavioral change yet — this is wiring.

**Why first:** Every other phase depends on the registry existing. If you build schema, APIs, and sync first and then try to retrofit a registry, you'll have conditionals everywhere.

**Deliverables:**
- `app/src/integrations/integrations/domain/provider_registry.py` — `ProviderRegistry` class
- `app/src/context-engine/domain/ports/issue_tracker.py` — `IssueTrackerPort` Protocol
- `ProviderDefinition` dataclass with all fields from §2.1
- GitHub registered as a provider: wraps existing `SourceControlPort` / `CodeProviderFactory` / `build_pr_episode` / `ingest_merged_pull_request` / `backfill_pot_context`
- Bootstrap: `load_providers()` called in API startup (`main.py`) and Celery worker startup. Commercial package hook: `try: import potpie_integrations_commercial; ...`
- Tests: registry CRUD, GitHub provider resolves correctly, mock second provider registers and is discoverable

**Acceptance:** `ProviderRegistry.list_available()` returns `[github]`. A mock `linear` provider can be registered in tests. Zero behavioral change to existing sync.

---

### Phase 2 — Credential vault + connections schema (OSS)

**What:** Build `CredentialVault` (encrypt/decrypt), tighten the `integrations` table (rename to `connections` or add a migration that adds the encryption/webhook columns), and build the GitHub aggregation adapter that presents `user_auth_providers` as a connection.

**Why second:** Tokens are the foundation of everything. Building sources, sync, or APIs without proper credential handling means rework.

**Deliverables:**
- `CredentialVault` service: encrypt/decrypt `auth_data` with configurable key backend (local symmetric for self-hosted, KMS for cloud)
- Migration: add `webhook_secret`, ensure `auth_data` column can hold encrypted blobs; add `status` enum values (`token_expired`, `revoked`)
- GitHub connection aggregation adapter: reads `user_auth_providers` + GitHub App state → returns generic `Connection` view
- Token refresh Celery beat task (skeleton — GitHub tokens from Firebase don't expire the same way, but the mechanism must exist for Linear)
- Tests: encrypt/decrypt round-trip, aggregation adapter returns correct state for GitHub users

**Acceptance:** `auth_data` for new connections is encrypted. GitHub connections appear in the aggregated view. Existing GitHub auth is not disrupted.

---

### Phase 3 — `project_sources` schema + multi-source runtime (OSS)

**What:** Create the `project_sources` table, backfill existing GitHub projects, and refactor the runtime (`tasks.py`, `wiring.py`, `backfill_pot.py`) to iterate over sources from the DB instead of assuming `primary_repo()`.

**Why third:** This is the biggest structural change. It needs the registry (Phase 1) to dispatch per-provider and the credential layer (Phase 2) to resolve tokens for each source.

**Deliverables:**
- Alembic migration: `project_sources` table with all columns from §2.1, indexes, unique constraint
- Backfill script: for each `Project` with `repo_name` + `status=ready`, insert a `ProjectSource` row (`provider=github`, `source_kind=repository`, `scope_json={"repo_name": ...}`)
- Refactor `SqlalchemyPotResolution.resolve_pot()` to populate `ResolvedPot.repos` from `project_sources` rows. Fall back to `Project.repo_name` when no rows exist (backward compatibility).
- Refactor `context_graph_backfill_pot` to iterate `resolved.repos` and dispatch to the registry's backfill handler per repo's provider. No more `primary_repo()`.
- Refactor `context_graph_ingest_pr` to accept `repo_name` in the task payload and resolve the matching repo from `resolved.repos`.
- On project create/ready: auto-create `ProjectSource` for GitHub (idempotent).
- Tests: backfill visits all repos when `ResolvedPot.repos` has >1 entry; single-GitHub-repo behavior unchanged; project creation triggers `ProjectSource` row

**Acceptance:** Existing GitHub sync works exactly as before (now via `project_sources` rows). The runtime can handle multiple sources per project. New projects get explicit `ProjectSource` rows.

---

### Phase 4 — Sources API + webhook router (OSS)

**What:** Build the full API surface and the generic webhook router. Everything the UI and external clients need.

**Deliverables:**

API endpoints:
- `GET /api/v1/sources/providers` — catalog from registry (capabilities, availability)
- `GET /api/v1/sources/connections` — aggregated view (GitHub from aggregation adapter + others from `connections`)
- `POST /api/v1/sources/connections/{provider}/oauth/start` — initiate OAuth (returns redirect URL)
- `POST /api/v1/sources/connections/{provider}/oauth/callback` — handle OAuth callback, store encrypted tokens
- `DELETE /api/v1/sources/connections/{connection_id}` — disconnect (revoke, cascade-disable sources)
- `GET /api/v1/projects/{project_id}/sources` — list attached sources with sync health
- `POST /api/v1/projects/{project_id}/sources` — attach a source (validate scope against connection)
- `DELETE /api/v1/projects/{project_id}/sources/{source_id}` — detach
- `POST /api/v1/projects/{project_id}/sources/{source_id}/sync` — sync specific source
- `POST /api/v1/projects/{project_id}/sources/sync` — sync all sources for project
- `POST /api/v1/context/sync` — compatibility endpoint, delegates to orchestrator

Webhook router:
- `POST /api/v1/sources/webhooks/{provider}` — generic entry point. Validates signature using `connection.webhook_secret`, identifies `ProjectSource` by scope, enqueues provider-specific Celery task via registry.

**Acceptance:** Full API surface is live. GitHub: connect, attach, sync, detach all work. Webhook router dispatches correctly (tested with mock provider). Compatibility with existing UI sync preserved.

---

### Phase 5 — Sources UI (OSS)

**What:** Refactor the Sources page to be provider-agnostic, driven entirely by the API.

**Deliverables:**
- Refactor `potpie-ui/potpie-ui/app/(main)/sources/page.tsx`: fetch `providers`, `connections`, `sources` for selected project. Render a card per provider. Connect/disconnect flows. Per-source sync + health display.
- Fix or redirect `/integrations` sidebar link (currently 404).
- GitHub card: connect/install flow using existing patterns (`useGithubAppPopup`). Sync button per source.
- Provider cards are **data-driven**: if the API returns `linear` as available, the UI renders a Linear card with connect flow. No `if (provider === 'linear')` in the UI.
- Sync health: show `last_sync_at`, `health_score`, `last_error` per source. Allow manual re-sync.

**Acceptance:** Sources page works for GitHub. If a commercial provider is registered, its card appears automatically. No hardcoded provider logic in the UI.

---

### Phase 6 — Linear provider (closed source)

**What:** Full Linear integration as the first commercial provider plugin.

**Deliverables (closed repo):**
- `LinearIssueTrackerAdapter` implementing `IssueTrackerPort`: GraphQL client, `iter_issues`, `get_issue`, `get_comments`, `get_state_history`
- `build_linear_issue_episode()`: rich narrative markdown (title, description, assignee, status transitions, comments, linked PRs, labels, project/cycle context)
- `ingest_linear_issue()`: ledger dedup, `episodic.add_episode`, cross-reference extraction (PR URLs → structural edges), ledger append. `LedgerScope.provider="linear"`, `provider_host="linear.app"`, `repo_name` = team/workspace identifier.
- `backfill_linear_context()`: paginate issues by `updatedAt`, cursor-based, same ledger + rate-limit pattern
- Linear webhook handler: validate HMAC signature, parse event type (issue created/updated, comment added), enqueue targeted ingest task
- `register_linear()`: registers `ProviderDefinition` with all handlers, OAuth config, refresh interval
- Remove `linear_oauth.py` from public repo (or replace with stub)

**Deliverables (OSS repo — if needed):**
- Stub/gated routes for `/api/v1/sources/connections/linear/*` that return 403 when the commercial package is not loaded

**Acceptance:** Connect Linear, pick workspace/team, attach to project, sync (backfill + webhook), see Linear data in context-graph queries. Cross-references to GitHub PRs create structural edges.

---

### Phase 7 — Production hardening

**What:** Everything needed for this to be reliable at scale.

**Deliverables:**
- Per-provider Celery queues (`context-graph-linear`, etc.) with configurable rate limits and concurrency
- Backpressure: monitor Graphiti adapter latency + queue depth; pause provider backfill when overloaded
- Token refresh: Celery beat task proactively refreshes tokens nearing expiry; auto-disable connections on repeated failure
- Provider health tracking: `health_score` on `ProjectSource` computed from consecutive success/failure; auto-disable after threshold; UI surfaces health state
- Structured observability: logs tagged with `project_id`, `source_id`, `provider`; metrics for sync latency, episode count, error rate per provider
- Source-type filtering in `resolve_context` / `EpisodicGraphPort.search`: `source_types` parameter, freshness decay for stale sources
- Admin tools: re-run sync, clear error state, force-refresh token, manually trigger backfill

---

## 7. OSS / Closed-Source Boundary

| In the public repo | In the closed package |
|--------------------|-----------------------|
| `ProviderRegistry`, `ProviderDefinition`, all port Protocols | Concrete adapters: `LinearIssueTrackerAdapter`, `JiraAdapter`, … |
| `project_sources` schema, Alembic migrations | Provider-specific OAuth configs (client IDs, secrets) |
| `CredentialVault` (encryption logic) | |
| Sources API (all endpoints) | Provider-specific webhook signature validation |
| Sources UI (generic, data-driven) | Optional branded connect-flow components |
| `entity_schema.py` (core types) | Provider-specific entity type extensions |
| `episode_formatters.py` (GitHub only) | `build_linear_issue_episode`, etc. |
| `ingest_merged_pull_request` (GitHub) | `ingest_linear_issue`, etc. |
| `backfill_pot_context` (GitHub) | `backfill_linear_context`, etc. |
| Webhook router (generic dispatch) | Webhook handlers (per-provider payload parsing) |
| Bootstrap hook: `try: import potpie_integrations_commercial` | `register()` function that registers all commercial providers |
| Mock provider + testing harness | |

**Rule:** No `if provider == "linear"` in the public repo. No client IDs, secrets, or proprietary API logic. OSS defines the interfaces; the closed package fills them in.

---

## 8. File / Module Placement

Integrations code lives in the **`potpie-integrations`** editable package (`app/src/integrations/`), hexagonal layout like `context-engine`. Import root is **`integrations`** (not `app.modules.integrations`).

| Component | Location |
|-----------|----------|
| `ProviderRegistry` | `app/src/integrations/integrations/domain/provider_registry.py` |
| `ProviderDefinition` | `app/src/integrations/integrations/domain/provider_definitions.py` |
| Integration schemas (Pydantic) | `app/src/integrations/integrations/domain/integrations_schema.py` |
| `CredentialVault` | (planned — not implemented yet) |
| `Integration` / `ProjectSource` ORM | `app/src/integrations/integrations/adapters/outbound/postgres/` |
| Sources API | `app/src/integrations/integrations/adapters/inbound/http/sources_router.py` (mounted in `main.py`) |
| Integrations OAuth API | `app/src/integrations/integrations/adapters/inbound/http/integrations_router.py` |
| Linear webhook (current) | `POST /api/v1/sources/webhooks/linear` in `sources_router.py`; generic `webhooks/{provider}` (planned) |
| GitHub connection aggregation | (planned dedicated adapter) — today aggregated inline in `sources_router` `list_connections` |
| `IssueTrackerPort` | `app/src/context-engine/domain/ports/issue_tracker.py` |
| GitHub / Linear provider registration | `app/src/integrations/integrations/adapters/outbound/providers/` |
| Bootstrap (`load_providers`) | `app/src/integrations/integrations/application/bootstrap.py` (called from `main.py` + Celery worker) |
| Root dependency | `pyproject.toml` → `potpie-integrations` → path `app/src/integrations` |
| Alembic migrations | `app/alembic/versions/` |
| Closed package | `potpie-integrations-commercial/` (separate repo/package) |

---

## 9. API Surface

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/sources/providers` | Provider catalog (capabilities, availability) |
| GET | `/api/v1/sources/connections` | User's connected accounts (aggregated) |
| POST | `/api/v1/sources/connections/{provider}/oauth/start` | Start OAuth flow |
| POST | `/api/v1/sources/connections/{provider}/oauth/callback` | Complete OAuth |
| DELETE | `/api/v1/sources/connections/{id}` | Disconnect (revoke + cascade) |
| GET | `/api/v1/projects/{id}/sources` | Attached sources + health |
| POST | `/api/v1/projects/{id}/sources` | Attach source |
| DELETE | `/api/v1/projects/{id}/sources/{sid}` | Detach |
| POST | `/api/v1/projects/{id}/sources/{sid}/sync` | Sync one source |
| POST | `/api/v1/projects/{id}/sources/sync` | Sync all sources |
| POST | `/api/v1/sources/webhooks/{provider}` | Inbound webhook |
| POST | `/api/v1/context/sync` | Compatibility (delegates to orchestrator) |

---

## 10. Risks

| Risk | Mitigation |
|------|------------|
| GitHub connections split across `user_auth_providers` and `integrations` | Explicit aggregation adapter; long-term migrate GitHub into `connections` |
| Existing `linear_oauth.py` in OSS contains commercial code | Remove or stub in Phase 6 |
| Entity schema bloat | Governance: core types generic, new types need extraction-quality evidence, plugin extension API |
| Noisy multi-source context dilutes answer quality | Source-type filtering + freshness decay in `resolve_context` |
| Slow Graphiti extraction blocks sync pipeline | Per-provider queues + backpressure monitoring |
| Webhook delivery gaps | Periodic backfill as safety net; `webhook_status` tracking |
| Token leaks from `auth_data` | Envelope encryption from day one (Phase 2) |
| Provider API breaking changes | Version field in `ProviderDefinition`; adapters can coexist |
| Plugin authors breaking core schema | Extension API validates at startup; core types immutable |
| Celery workers missing provider registrations | Single `load_providers()` bootstrap in both API and worker; fail-loud if expected provider missing |

---

## 11. Definition of Done

### Phase 1 — Registry
- [ ] `ProviderRegistry` with register/get/list
- [ ] `IssueTrackerPort` Protocol defined
- [ ] GitHub registered as provider (wrapping existing code)
- [ ] Bootstrap loads providers in API + Celery worker
- [ ] Mock provider registers and resolves in tests

### Phase 2 — Credentials
- [ ] `CredentialVault` encrypts/decrypts `auth_data`
- [ ] Migration: encryption + webhook columns on connections table
- [ ] GitHub aggregation adapter returns connections view
- [ ] Token refresh beat task (skeleton)
- [ ] Revocation cascade implemented

### Phase 3 — Multi-source runtime
- [ ] `project_sources` table + backfill from `Project.repo_name`
- [ ] `resolve_pot()` reads from `project_sources` (fallback to `repo_name`)
- [ ] Backfill + PR ingest iterate `resolved.repos` via registry dispatch
- [ ] Auto-create `ProjectSource` on project ready
- [ ] Tests: multi-repo backfill, single-repo backward compat

### Phase 4 — APIs + webhooks
- [ ] All endpoints from §9 implemented and tested
- [ ] Webhook router dispatches to registered provider handlers
- [ ] Compatibility with existing `/api/v1/context/sync` preserved
- [ ] OAuth flow works for GitHub (aggregation) and is ready for commercial providers

### Phase 5 — UI
- [ ] Sources page renders from API (no hardcoded providers)
- [ ] GitHub connect/sync works
- [ ] Per-source health display
- [ ] `/integrations` sidebar link resolved

### Phase 6 — Linear
- [ ] Full Linear lifecycle: connect → attach → backfill + webhook sync → query
- [ ] Cross-provider correlation edges (Linear issue ↔ GitHub PR)
- [ ] `linear_oauth.py` removed from OSS
- [ ] Episode quality validated with golden queries

### Phase 7 — Production
- [ ] Per-provider queues + backpressure
- [ ] Token refresh + auto-disable on failure
- [ ] Source-type filtering in `resolve_context`
- [ ] Observability: structured logs + metrics per provider
- [ ] Health tracking + admin tools

---

## 12. References

**Existing code (context-engine):** `domain/ports/pot_resolution.py` (`ResolvedPot`, `ResolvedPotRepo`, `single_github_repo_pot`), `domain/ports/episodic_graph.py`, `domain/ports/ingestion_ledger.py`, `domain/ports/source_control.py`, `domain/entity_schema.py`, `domain/episode_formatters.py`, `application/use_cases/ingest_merged_pr.py`, `application/use_cases/backfill_pot.py`, `adapters/outbound/graphiti/episodic.py` (paths under `app/src/context-engine/`)

**Potpie wiring:** `app/modules/context_graph/wiring.py` (SqlalchemyPotResolution, build_container_for_session), `app/modules/context_graph/tasks.py` (Celery tasks), `app/celery/worker.py`

**Auth / integrations:** `app/modules/auth/auth_provider_model.py` (`UserAuthProvider`); `app/src/integrations/integrations/adapters/outbound/postgres/integration_model.py`, `app/src/integrations/integrations/adapters/outbound/oauth/linear_oauth.py`, `app/src/integrations/integrations/adapters/inbound/http/integrations_router.py`

**UI:** `potpie-ui/potpie-ui/app/(main)/sources/page.tsx`, `potpie-ui/potpie-ui/lib/Constants.tsx`
