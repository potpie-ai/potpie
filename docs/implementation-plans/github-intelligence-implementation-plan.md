# GitHub Intelligence Layer — Implementation Plan

Branch: `feat/context-engine`

Architecture reference: [github-intelligence-layer-architecture.md](./github-intelligence-layer-architecture.md)

---

## Current state on `feat/context-engine`

| Area | Status |
|------|--------|
| `app/modules/context_graph/` | Only `context_graph_router.py` (partial — imports missing modules). No other files. |
| Graphiti integration | Not installed. No `graphiti-core` in dependencies. |
| Entity schemas | Do not exist. |
| Episode formatters | Do not exist. |
| Ingestion service / tasks | Do not exist. |
| Bridge writer | Does not exist. |
| Postgres models / migrations | No `context_sync_state`, `context_ingestion_log`, or `raw_events` tables. |
| Agent tools | No `context_tools/` directory. |
| Config | No `get_context_graph_config()` on `ConfigProvider`. No `CONTEXT_GRAPH_ENABLED` env var handling. |
| Router mounting | `context_graph_router` not included in `app/main.py`. |
| Celery queues | `scripts/start.sh` does not consume `context-graph-etl` or `external-event`. |
| GitHub provider | `get_pull_request(include_diff=True)` exists. **No** methods for PR commits, review comments, or issue comments. |
| Webhook handler | GitHub webhooks reach Celery (`external-event` queue) but handler is a generic echo — no PR merge processing. |
| Code graph | NODE labels: FILE, CLASS, FUNCTION, INTERFACE with `file_path`, `start_line`, `end_line`, `repoId`. Relationship types: CONTAINS, REFERENCES. |

---

## Milestone 0: Foundation wiring (must be done first)

**Goal:** Make the module importable, configurable, and reachable. Nothing intelligence-related yet — just plumbing.

### 0.1 — Config provider

**File:** `app/core/config_provider.py`

- Add `get_context_graph_config()` method to `ConfigProvider`.
- Reads `CONTEXT_GRAPH_ENABLED` (default `false`), `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` from env.
- Returns dict: `{ enabled: bool, neo4j_uri, neo4j_user, neo4j_password }`.

### 0.2 — Package init and router mount

**Files:** `app/modules/context_graph/__init__.py`, `app/main.py`

- Create `__init__.py` (can be empty or docstring).
- Add `include_router(context_graph_router.router)` in `app/main.py`.

### 0.3 — Postgres models

**File:** `app/modules/context_graph/models.py`

Three SQLAlchemy models:

| Model | Table | Purpose |
|-------|-------|---------|
| `ContextSyncState` | `context_sync_state` | ETL cursor per (project_id, source_type): `last_synced_at`, `status`, `error`. Unique on `(project_id, source_type)`. |
| `ContextIngestionLog` | `context_ingestion_log` | Dedup log: `(project_id, source_type, source_id)` → `graphiti_episode_uuid`, `bridge_written`. Unique index on the triple. |
| `RawEvent` | `raw_events` | Immutable event store: `(project_id, source_type, source_id)` → `payload` (JSONB), `received_at`, `processed_at`. Unique on triple. |

### 0.4 — Alembic migration

**File:** New migration under `app/alembic/versions/`

- Creates all three tables from 0.3.

### 0.5 — Dependency: install `graphiti-core`

- Add `graphiti-core` to `pyproject.toml` dependencies.
- Run `uv add graphiti-core` (or equivalent).
- Verify import works: `from graphiti_core import Graphiti`.

### 0.6 — Celery queue wiring

**Files:** `scripts/start.sh`, `app/celery/celery_app.py`

- In `scripts/start.sh`: when `CONTEXT_GRAPH_ENABLED=true`, add `context-graph-etl` to worker `-Q` list (same pattern as existing conditional).
- Verify `external-event` queue is consumed: either add to main worker or document that `scripts/start_event_worker.sh` must run.

### 0.7 — `.env.template` update

- Add `CONTEXT_GRAPH_ENABLED=false` with comment.

**Deliverable:** `POST /api/v1/context-graph/sync-all` returns 503 correctly when disabled; module imports without errors; tables exist after migration.

---

## Milestone 1: Graphiti client + entity schemas

**Goal:** Define the type system and get a working Graphiti wrapper.

### 1.1 — Entity schema

**New file:** `app/modules/context_graph/entity_schema.py`

Pydantic models (all fields Optional):

- `PullRequest` — pr_number, title, why_summary, change_type, feature_area, author, merged_at, files_changed.
- `Commit` — sha, message, author, branch.
- `Issue` — issue_number, title, problem_statement.
- `Feature` — name, description.
- `Decision` — decision_made, alternatives_rejected, rationale.
- `Developer` — github_login, display_name, expertise_areas.

Edge models:

- `Modified`, `Fixes`, `PartOfFeature`, `MadeIn`, `AuthoredBy`, `Owns`.

Constants:

- `ENTITY_TYPES` dict mapping name → model.
- `EDGE_TYPES` dict mapping name → model.
- `EDGE_TYPE_MAP` dict mapping (source_type, target_type) → list of edge type names.

### 1.2 — Graphiti client wrapper

**New file (or rewrite):** `app/modules/context_graph/graphiti_client.py`

- `ContextGraphClient.__init__`: reads config, initializes `Graphiti(uri, user, password)`.
- `add_episode(project_id, name, episode_body, source_description, reference_time)`: calls `graphiti.add_episode(...)` with `entity_types`, `edge_types`, `edge_type_map` from 1.1, `group_id=project_id`. Returns episode UUID.
- `search(project_id, query, limit, node_labels=None)`: calls `graphiti.search(...)` with optional `SearchFilters(node_labels=...)`.
- Graceful fallback if `graphiti-core` not installed or `enabled=false`.

**Deliverable:** Can add an episode and search within a project namespace. Entity types are passed on every ingest.

---

## Milestone 2: Deterministic extractors + review thread grouping

**Goal:** Pure functions, no I/O, fully testable.

### 2.1 — Deterministic extractors

**New file:** `app/modules/context_graph/deterministic_extractors.py`

- `extract_issue_refs(text) → list[int]`: Regex for "Fixes/Closes/Resolves #N" in PR body and commit messages.
- `extract_ticket_from_branch(branch_name) → str | None`: Regex for `PROJ-123` style tokens.
- `extract_feature_from_labels(labels, milestone) → str | None`: Milestone first, then first non-bug label.
- `parse_diff_hunks(patch) → list[tuple[int, int]]`: Regex for `@@ -n,m +n,m @@` → list of (start_line, end_line) in the new file.

### 2.2 — Review thread grouper

**New file:** `app/modules/context_graph/review_thread_grouper.py`

- `group_review_threads(flat_comments) → list[dict]`: Groups by `in_reply_to_id`. Each thread has `path`, `line`, `diff_hunk`, `comments: [{author, body, created_at}]`.

### 2.3 — Unit tests

**New file:** `tests/unit/context_graph/test_extractors.py`

- Test each extractor with known inputs and edge cases.
- Test review thread grouping with mock comment lists.

**Deliverable:** All parsers work and are tested before touching Graphiti or Neo4j.

---

## Milestone 3: GitHub provider extensions

**Goal:** Expose PR commits, review comments, and issue comments from GitHub API.

### 3.1 — Add to `ICodeProvider` interface

**File:** `app/modules/code_provider/base/code_provider_interface.py`

New abstract methods (with default empty-list implementations so other providers don't break):

- `get_pull_request_commits(repo_name, pr_number) → list[dict]`
- `get_pull_request_review_comments(repo_name, pr_number) → list[dict]`
- `get_pull_request_issue_comments(repo_name, pr_number) → list[dict]`

### 3.2 — Implement on `GitHubProvider`

**File:** `app/modules/code_provider/github/github_provider.py`

- `get_pull_request_commits`: via `pr.get_commits()` (PyGithub). Returns list of `{sha, message, author, committed_at}`.
- `get_pull_request_review_comments`: via `pr.get_review_comments()`. Returns list of `{id, body, user: {login}, path, line, in_reply_to_id, diff_hunk, created_at}`.
- `get_pull_request_issue_comments`: via `pr.get_issue_comments()`. Returns list of `{id, body, user: {login}, created_at}`.

These are **read-only** methods (no new writes). Cap each at a sensible limit (50 issue comments, 100 review comments) to avoid huge payloads.

**Deliverable:** All raw GitHub data needed for rich episodes is accessible from the provider.

---

## Milestone 4: Episode builder (rich format)

**Goal:** Build the "episode body" text that Graphiti's LLM will extract entities from.

### 4.1 — PR episode builder

**New file (or rewrite):** `app/modules/context_graph/episode_formatters.py`

`build_pr_episode(pr_data, commits, review_threads, linked_issues) → dict`

Episode body structure (sections in this order):
1. PR # + title
2. Author, branches, merge date
3. Files changed (comma-separated)
4. "WHY THIS CHANGE WAS MADE" — PR body (fall back to linked issue body if empty)
5. "RELATED ISSUES" — issue number + title + body excerpt per linked issue
6. "COMMITS" — sha + first-line message per commit
7. "REVIEW DISCUSSIONS" — per thread: file + line context, diff hunk excerpt, back-and-forth comments
8. Labels and milestone/feature

Returns: `{ name, episode_body, source_description, source_id, reference_time }`.

### 4.2 — Commit episode builder (standalone commits only)

Same file. `build_commit_episode(commit_data, branch) → dict`.

For direct pushes to default branch that didn't go through a PR.

**Deliverable:** Episode text is rich enough for Graphiti to extract PullRequest, Issue, Decision, Feature, Developer, Commit entities.

---

## Milestone 5: Ingestion service + backfill task

**Goal:** Wire the full pipeline: fetch → dedup → episode → Graphiti → log.

### 5.1 — Ingestion service

**New file (or rewrite):** `app/modules/context_graph/ingestion_service.py`

`ingest_pr(db, project_id, pr_data, commits, review_threads, linked_issues) → Optional[str]`

Steps:
1. Build `source_id` (e.g. `pr_42_merged`).
2. Check `context_ingestion_log` — skip if exists.
3. Save raw payload to `raw_events`.
4. Build episode via `build_pr_episode()`.
5. Call `ContextGraphClient.add_episode()` with entity types.
6. Log to `context_ingestion_log` with episode UUID.
7. Return UUID.

### 5.2 — PR data fetcher helper

**New file:** `app/modules/context_graph/github_pr_fetcher.py`

`fetch_full_pr(github_provider, repo_name, pr_number) → dict`

Combines: `get_pull_request(include_diff=True)` + `get_pull_request_commits()` + `get_pull_request_review_comments()` → `group_review_threads()` + `get_pull_request_issue_comments()` + fetch linked issues from `extract_issue_refs(pr.body)` via `get_issue()`.

Returns a bundle ready for `ingest_pr()`.

### 5.3 — Backfill Celery task

**New file (or rewrite):** `app/modules/context_graph/tasks.py`

`context_graph_backfill_project(project_id)`

Steps:
1. Get project's `repo_name` from Postgres.
2. Authenticate `GitHubProvider`.
3. Read `context_sync_state` for cursor (`last_synced_at`).
4. Paginate merged PRs (oldest first, capped at 100 per run).
5. For each PR: `fetch_full_pr()` → `ingest_pr()`.
6. Update `context_sync_state.last_synced_at` after each batch.
7. Rate limit: 0.5s delay between PR fetches.

Queue: `context-graph-etl`.

### 5.4 — Fix `context_graph_router.py`

Existing router imports from `tasks` which now exists. Verify it works with the selective sync (`project_ids` filter) we already added.

**Deliverable:** Backfill from UI works end-to-end: Sources → Sync now → Celery task → GitHub API → episodes in Graphiti → logged in Postgres.

---

## Milestone 6: Bridge writer

**Goal:** Connect Graphiti entities back to code graph nodes.

### 6.1 — Bridge writer service

**New file:** `app/modules/context_graph/bridge_writer.py`

`write_bridges(neo4j_driver, project_id, pr_entity_name, pr_number, files_with_patches, review_threads, merged_at, is_live=False)`

Steps:
1. For each file in the PR:
   a. Write `(FILE)-[:TOUCHED_BY {pr_number}]->(Entity:PullRequest)`.
   b. If `is_live` (webhook, not backfill): parse diff hunks → line ranges → match FUNCTION/CLASS nodes by overlap → write `(NODE)-[:MODIFIED_IN {pr_number, merged_at}]->(Entity:PullRequest)`.
2. For each review thread that has a `path` and `line`: match the FUNCTION/CLASS node whose `(file_path, start_line, end_line)` overlaps that line → find the `Decision` entity Graphiti created from that thread (by name/content match within `group_id`) → write `(NODE)-[:HAS_DECISION]->(Entity:Decision)`. This links code nodes directly to design decisions from review.
3. Update `context_ingestion_log.bridge_written = True`.

### 6.2 — Neo4j indexes

Add in the Alembic migration or a startup script:
- `(n:NODE)` on `(n.file_path, n.repoId)` — for bridge queries.
- `(e:Entity)` on `(e.name, e.group_id)` — for finding PR entities.

### 6.3 — Integrate into ingestion

After `ingest_pr()` succeeds (episode UUID returned), call `write_bridges()`.

**Deliverable:** After sync, code graph nodes are linked to PR entities. Cypher traversals from FUNCTION → PullRequest work.

---

## Milestone 7: Agent tools

**Goal:** Agents can query the intelligence layer.

### 7.1 — `get_change_history` tool

**New file:** `app/modules/intelligence/tools/context_tools/get_change_history_tool.py`

Input: `project_id`, optional `function_name` or `file_path`, `limit`.

Implementation: Neo4j Cypher — traverse `(NODE)-[:MODIFIED_IN]->(Entity:PullRequest)`, optionally join to `Issue` and `Decision` entities.

Returns: list of `{ pr_number, title, why_summary, change_type, feature_area, fixed_issues, decisions }`.

### 7.2 — `get_file_owner` tool

**New file:** `app/modules/intelligence/tools/context_tools/get_file_owner_tool.py`

Input: `project_id`, `file_path`, `limit`.

Implementation: Neo4j Cypher — traverse `(FILE)-[:TOUCHED_BY]->(Entity:PullRequest)-[:AuthoredBy]->(Entity:Developer)`, aggregate by developer, order by recency and PR count.

Returns: list of `{ github_login, pr_count, last_touched }`.

### 7.3 — `get_decisions` tool

**New file:** `app/modules/intelligence/tools/context_tools/get_decisions_tool.py`

Input: `project_id`, optional `file_path` or `function_name`.

Implementation: Neo4j Cypher — traverse `(NODE)-[:MODIFIED_IN|HAS_DECISION]->(Entity:Decision)`, return decision text, rejected alternatives, and rationale.

Returns: list of `{ decision_made, alternatives_rejected, rationale, pr_number }`.

### 7.4 — Upgrade `get_project_context` tool

**New file:** `app/modules/intelligence/tools/context_tools/get_project_context_tool.py`

Same as existing concept but passes `SearchFilters(node_labels=[...])` when searching Graphiti for typed results.

### 7.5 — Register tools

**File:** `app/modules/intelligence/tools/tool_service.py` (or equivalent registry)

Register `get_change_history`, `get_file_owner`, `get_decisions`, and `get_project_context` so agents can use them.

**Deliverable:** Agents can query change history, file ownership, design decisions, and broad project context.

---

## Milestone 8: Live ingestion (webhook path)

**Goal:** When a PR is merged on GitHub, it automatically enters the intelligence graph.

### 8.1 — GitHub webhook handler

**File:** `app/modules/event_bus/handlers/webhook_handler.py`

Add a `github` branch in `process_event()`:
- If event is `pull_request` with `action: closed` and `merged: true`:
  - Extract `repo_name`, `pr_number` from payload.
  - Look up project by `repo_name`.
  - Enqueue `context_graph_ingest_pr.delay(project_id, pr_number)`.

### 8.2 — Single-PR ingest Celery task

**File:** `app/modules/context_graph/tasks.py`

`context_graph_ingest_pr(project_id, pr_number)`

Same as backfill but for one PR. Uses `is_live=True` in bridge writer for function-level matching.

Queue: `context-graph-etl` (or `external-event` — decide based on worker setup).

### 8.3 — Ensure `external-event` queue is consumed

**File:** `scripts/start.sh`

Either add `external-event` to the main worker's `-Q` list, or document that `scripts/start_event_worker.sh` must be running.

**Deliverable:** Merge a PR on GitHub → webhook → Celery → episode + bridge → queryable by agents within seconds.

---

## Milestone 9: Parsing pipeline hook

**Goal:** When a new repo is parsed, automatically trigger context graph backfill.

### 9.1 — Post-parse hook

**File:** `app/celery/tasks/parsing_tasks.py`

After `process_parsing` succeeds:
- If `CONTEXT_GRAPH_ENABLED=true`: enqueue `context_graph_backfill_project.delay(project_id)`.

**Deliverable:** First-time repo parse automatically backfills PR history.

---

## Build order summary

```
M0  Foundation wiring              ← config, models, migration, deps, queues
 │
M1  Graphiti client + schemas      ← entity types, client wrapper
 │
M2  Deterministic extractors       ← regex parsers, thread grouper, unit tests
 │
M3  GitHub provider extensions     ← PR commits, review comments, issue comments
 │
M4  Episode builder                ← rich PR episode format
 │
M5  Ingestion service + backfill   ← fetch → dedup → episode → Graphiti → log
 │
M6  Bridge writer                  ← diff hunks → code graph → PullRequest links
 │
M7  Agent tools                    ← get_change_history, get_project_context
 │
M8  Live ingestion (webhooks)      ← PR merge → auto-ingest
 │
M9  Parsing pipeline hook          ← parse repo → auto-backfill
```

M0–M2 have no external dependencies — can start immediately.
M3 is independent of M1–M2 (can be parallel).
M4 depends on M2 (extractors) + M3 (provider methods).
M5 depends on M1 (Graphiti client) + M4 (episode builder).
M6 depends on M5 (needs ingested episodes to bridge).
M7 depends on M6 (needs bridges for graph traversal queries).
M8 depends on M5 (ingestion service exists).
M9 depends on M5 (backfill task exists).

**Parallelization:** M1, M2, M3 can all be built simultaneously.

---

## Testing strategy

| Milestone | Test type | What to verify |
|-----------|-----------|----------------|
| M0 | Smoke | Migration runs, config loads, router returns 503 when disabled |
| M1 | Integration | Graphiti client can add/search episodes against local Neo4j |
| M2 | Unit | All extractors + thread grouper with known inputs |
| M3 | Integration | Provider methods return expected shapes from a test repo |
| M4 | Unit | Episode body contains all expected sections |
| M5 | Integration | Full pipeline: fetch → ingest → verify Episodic nodes in Neo4j |
| M6 | Integration | Bridge edges exist: `(FUNCTION)-[:MODIFIED_IN]->(Entity:PullRequest)` |
| M7 | Integration | Agent tool returns structured results for known function |
| M8 | Integration | Simulated webhook payload → episode + bridge created |
| M9 | Integration | Parse task completion → backfill task enqueued |

---

## Files to create (full list)

| File | Milestone |
|------|-----------|
| `app/modules/context_graph/__init__.py` | M0 |
| `app/modules/context_graph/models.py` | M0 |
| `app/alembic/versions/..._context_graph_tables.py` | M0 |
| `app/modules/context_graph/entity_schema.py` | M1 |
| `app/modules/context_graph/graphiti_client.py` | M1 |
| `app/modules/context_graph/deterministic_extractors.py` | M2 |
| `app/modules/context_graph/review_thread_grouper.py` | M2 |
| `tests/unit/context_graph/test_extractors.py` | M2 |
| `app/modules/context_graph/episode_formatters.py` | M4 |
| `app/modules/context_graph/github_pr_fetcher.py` | M5 |
| `app/modules/context_graph/ingestion_service.py` | M5 |
| `app/modules/context_graph/tasks.py` | M5 |
| `app/modules/context_graph/bridge_writer.py` | M6 |
| `app/modules/intelligence/tools/context_tools/__init__.py` | M7 |
| `app/modules/intelligence/tools/context_tools/get_change_history_tool.py` | M7 |
| `app/modules/intelligence/tools/context_tools/get_file_owner_tool.py` | M7 |
| `app/modules/intelligence/tools/context_tools/get_decisions_tool.py` | M7 |
| `app/modules/intelligence/tools/context_tools/get_project_context_tool.py` | M7 |

## Files to modify (full list)

| File | Milestone | Change |
|------|-----------|--------|
| `app/core/config_provider.py` | M0 | Add `get_context_graph_config()` |
| `app/main.py` | M0 | Mount context graph router |
| `pyproject.toml` | M0 | Add `graphiti-core` dependency |
| `scripts/start.sh` | M0, M8 | Add `context-graph-etl` and `external-event` queues |
| `.env.template` | M0 | Add `CONTEXT_GRAPH_ENABLED` |
| `app/modules/context_graph/context_graph_router.py` | M5 | Fix imports (now modules exist) |
| `app/modules/code_provider/base/code_provider_interface.py` | M3 | Add PR commit/comment abstract methods |
| `app/modules/code_provider/github/github_provider.py` | M3 | Implement PR commit/comment methods |
| `app/modules/event_bus/handlers/webhook_handler.py` | M8 | Add GitHub PR merge handling |
| `app/celery/tasks/parsing_tasks.py` | M9 | Post-parse context graph hook |
| `app/modules/intelligence/tools/tool_service.py` | M7 | Register new tools |
