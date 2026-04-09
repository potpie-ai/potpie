# Potpie Agent Guide

## Scope

This repository contains the Potpie API and an embedded `context-engine` package. When a task mentions the `context-engine` CLI, work from the package sources under [`app/src/context-engine`](app/src/context-engine).

## Context-Engine CLI

- Entry point: [`app/src/context-engine/adapters/inbound/cli/main.py`](app/src/context-engine/adapters/inbound/cli/main.py)
- CLI docs: [`app/src/context-engine/adapters/inbound/cli/README.md`](app/src/context-engine/adapters/inbound/cli/README.md)
- Package overview: [`app/src/context-engine/README.md`](app/src/context-engine/README.md)

Use the repo-local skills under `.agents/skills/` when the task is about:

- running or explaining CLI commands
- resolving pot scope from git remotes or env maps
- debugging `doctor`, `search`, `ingest`, or Neo4j/Graphiti setup

### Context-graph job queue (Celery default, optional Hatchet)

**Default:** **`CONTEXT_GRAPH_JOB_QUEUE_BACKEND=celery`** (or unset → Celery). Context-graph tasks use the **`context-graph-etl`** Celery queue; **[`scripts/start.sh`](scripts/start.sh)** starts a worker that includes it when `CONTEXT_GRAPH_ENABLED` is on.

**Optional Hatchet:** set **`CONTEXT_GRAPH_JOB_QUEUE_BACKEND=hatchet`**, self-host Hatchet per upstream docs, set **`HATCHET_CLIENT_*`**, and run **`python -m app.modules.context_graph.hatchet_worker`**. See **[`docs/hatchet-local.md`](docs/hatchet-local.md)**. The Hatchet adapter code lives under `app/src/context-engine/adapters/outbound/hatchet/` and `adapters/inbound/hatchet/`.

- **Hatchet CLI agent skills** (optional): [`.agents/skills/hatchet-cli/`](.agents/skills/hatchet-cli/SKILL.md)

## Working Rules

- Prefer `uv run context-engine ...` from [`app/src/context-engine`](app/src/context-engine).
- For pot inference, follow the code path in [`app/src/context-engine/adapters/inbound/cli/git_project.py`](app/src/context-engine/adapters/inbound/cli/git_project.py): active pot from `pot use`, then env maps, then git `origin`, else fail.
- **`search`**, **`ingest`**, and **`pot hard-reset`** call Potpie **`POST /api/v2/context/*`** with **`X-API-Key`** (same auth as [`app/api/router.py`](app/api/router.py)). Set **`POTPIE_API_URL`** / **`POTPIE_BASE_URL`** and **`POTPIE_API_KEY`**, or run **`context-engine login`**. The machine running the CLI does **not** need local Neo4j/Graphiti. Async work runs on the **server** (Celery/Hatchet per host config).
- **`doctor`** checks stored credentials and probes **`GET /health`** on the configured base URL (connectivity, not graph depth).
- **Potpie server** still mounts **`/api/v1/context`** (Firebase) and **`/api/v2/context`** (API key); CLI/MCP use **v2** only.
- Keep changes aligned with Typer patterns already used in the CLI and update the CLI README when behavior changes.

## Validation

- Context graph / context-engine tests live **only** under [`app/src/context-engine/tests`](app/src/context-engine/tests). There is no separate `tests/unit/context_graph/` tree—avoid duplicating domain tests at the repo root.
- Potpie-only wiring for the embedded API is under [`app/modules/context_graph`](app/modules/context_graph); see [`app/modules/context_graph/README.md`](app/modules/context_graph/README.md) for what belongs there vs in the package.
- Common verification:
  - `uv run pytest app/src/context-engine/tests/unit/`
  - `uv run context-engine doctor`
