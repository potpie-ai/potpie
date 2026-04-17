# Potpie Agent Guide

## Scope

This repository contains the Potpie API and an embedded `context-engine` package. When a task mentions the `context-engine` CLI, work from the package sources under [`app/src/context-engine`](app/src/context-engine).

## Context-Engine CLI

- Entry point: [`app/src/context-engine/adapters/inbound/cli/main.py`](app/src/context-engine/adapters/inbound/cli/main.py)
- CLI docs: [`app/src/context-engine/adapters/inbound/cli/README.md`](app/src/context-engine/adapters/inbound/cli/README.md)
- Package overview: [`app/src/context-engine/README.md`](app/src/context-engine/README.md)

Use the repo-local skills under `.agents/skills/` when the task is about:

- gathering project context through `context_resolve` recipes
- running or explaining CLI commands
- resolving pot scope from git remotes or env maps
- debugging `doctor`, `search`, `ingest`, or Potpie API setup

### Context-graph job queue (Celery default, optional Hatchet)

**Default:** **`CONTEXT_GRAPH_JOB_QUEUE_BACKEND=celery`** (or unset → Celery). Context-graph tasks use the **`context-graph-etl`** Celery queue; **[`scripts/start.sh`](scripts/start.sh)** starts a worker that includes it when `CONTEXT_GRAPH_ENABLED` is on.

**Optional Hatchet:** set **`CONTEXT_GRAPH_JOB_QUEUE_BACKEND=hatchet`**, self-host Hatchet per upstream docs, set **`HATCHET_CLIENT_*`**, and run **`python -m app.modules.context_graph.hatchet_worker`**. See **[`docs/hatchet-local.md`](docs/hatchet-local.md)**. The Hatchet adapter code lives under `app/src/context-engine/adapters/outbound/hatchet/` and `adapters/inbound/hatchet/`.

- **Hatchet CLI agent skills** (optional): [`.agents/skills/hatchet-cli/`](.agents/skills/hatchet-cli/SKILL.md)

## Working Rules

- Prefer `uv run context-engine ...` from [`app/src/context-engine`](app/src/context-engine).
- For pot inference, follow the code path in [`app/src/context-engine/adapters/inbound/cli/git_project.py`](app/src/context-engine/adapters/inbound/cli/git_project.py): active pot from `pot use`, then env maps, then git `origin`, else fail.
- **`search`**, **`ingest`**, and **`pot hard-reset`** call Potpie **`POST /api/v2/context/*`** with **`X-API-Key`** (same auth as [`app/api/router.py`](app/api/router.py)). Set **`POTPIE_API_URL`** / **`POTPIE_BASE_URL`** and **`POTPIE_API_KEY`**, or run **`context-engine login`**. The machine running the CLI does **not** need local Neo4j/Graphiti. Async work runs on the **server** (Celery/Hatchet per host config).
- **`doctor`** checks stored credentials, probes **`GET /health`**, and runs an authenticated **`GET /api/v2/context/pots`** probe when URL and key resolve (API readiness, not graph depth).
- **Potpie server** still mounts **`/api/v1/context`** (Firebase) and **`/api/v2/context`** (API key); CLI/MCP use **v2** only.
- Agents should use the minimal context port when MCP is available:
  - **`context_resolve`**: primary task context wrap with `intent`, `scope`, `include`, `exclude`, `mode`, `source_policy`, `budget`, and `as_of`.
  - **`context_search`**: narrow follow-up memory lookup after `context_resolve`.
  - **`context_record`**: durable decisions, fixes, preferences, workflows, feature notes, incident summaries, and doc references.
  - **`context_status`**: cheap pot/scope readiness, freshness, known gaps, and recommended recipe.
- Do not add new agent tools for each context type; express feature, debugging, review, operations, docs, and onboarding workflows as `context_resolve` parameter recipes.
- Start with `mode=fast` and `source_policy=references_only`. Escalate to `source_policy=summary`, `verify`, `snippets`, or `mode=deep` only when coverage, freshness, source verification, or task risk requires it.
- Always inspect `coverage`, `freshness`, `quality`, `fallbacks`, `open_conflicts`, and `source_refs` before relying on graph memory.
- If `quality.status` is `watch` or `degraded`, verify relevant facts against source truth before high-impact work and follow any `quality.recommended_maintenance` jobs.
- Use `context_record` after discovering reusable project memory, especially fixes, bug patterns, decisions, preferences, runbook/workflow notes, feature notes, and incident summaries.
- Keep changes aligned with Typer patterns already used in the CLI and update the CLI README when behavior changes.

## Context Recipes

Feature work:

```json
{"intent":"feature","include":["purpose","feature_map","service_map","docs","tickets","decisions","recent_changes","owners","preferences","source_status"],"mode":"fast","source_policy":"references_only"}
```

Debugging:

```json
{"intent":"debugging","include":["prior_fixes","diagnostic_signals","incidents","alerts","recent_changes","config","deployments","owners","source_status"],"mode":"fast","source_policy":"references_only"}
```

Review:

```json
{"intent":"review","include":["artifact","discussions","owners","recent_changes","decisions","preferences","source_status"],"mode":"balanced","source_policy":"summary"}
```

Operations:

```json
{"intent":"operations","include":["deployments","runbooks","alerts","incidents","scripts","config","owners","source_status"],"mode":"balanced","source_policy":"summary"}
```

## Validation

- Context graph / context-engine tests live **only** under [`app/src/context-engine/tests`](app/src/context-engine/tests). There is no separate `tests/unit/context_graph/` tree—avoid duplicating domain tests at the repo root.
- Potpie-only wiring for the embedded API is under [`app/modules/context_graph`](app/modules/context_graph); see [`app/modules/context_graph/README.md`](app/modules/context_graph/README.md) for what belongs there vs in the package.
- Common verification:
  - `uv run pytest app/src/context-engine/tests/unit/`
  - `uv run context-engine doctor`
