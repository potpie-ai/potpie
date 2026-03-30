# context-engine

**Context graph** logic and adapters: ingestion, ledger, Neo4j/Graphiti queries, optional HTTP/MCP/CLI entrypoints. Built with **hexagonal (ports and adapters)** so the core stays independent of FastAPI, SQLAlchemy, Graphiti, and Neo4j drivers at the domain layer.

## Production: embedded in Potpie (default)

In production, **context-engine runs inside the Potpie API process**. You mount routes on the **same FastAPI app** and use the **same deployment** as Potpie—**no separate context-engine HTTP service** and **no extra API key** for these routes.

### How to use the HTTP API

1. Run Potpie as usual (`uvicorn`, etc.).
2. Authenticate like any other Potpie `/api/v1/*` route (same session / bearer / headers your app already uses for `AuthService.check_auth`).
3. Call the context routes on your Potpie base URL:

| Method | Path | What it does |
|--------|------|----------------|
| `POST` | `/api/v1/context/sync` | Enqueues Celery backfill per project (optional body: `{ "project_ids": ["..."] }`; omit to sync all **your** eligible projects) |
| `POST` | `/api/v1/context/ingest-pr` | Enqueues single-PR ingest (`project_id`, `pr_number`, optional `is_live_bridge`) |
| `POST` | `/api/v1/context/query/change-history` | Neo4j change history |
| `POST` | `/api/v1/context/query/file-owners` | File owner hints from PR history |
| `POST` | `/api/v1/context/query/decisions` | Linked decisions |
| `POST` | `/api/v1/context/query/search` | Graphiti semantic search |

**Project scope:** Requests only apply to projects **owned by the authenticated user** (enforced in Potpie’s wiring).

**Long-running work:** Sync and ingest **enqueue Celery** tasks (`context-graph-etl` queue). Run a Potpie Celery worker that consumes that queue so jobs actually execute.

### Configuration (Potpie)

Context graph reads **Potpie’s config** (via `config_provider`), not a parallel auth layer:

| Concern | Notes |
|---------|--------|
| Feature flag | `CONTEXT_GRAPH_ENABLED=true` (and related Potpie env you already use) |
| Neo4j / Graphiti | Same Neo4j settings as the rest of Potpie (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, etc., as configured in Potpie) |
| Ledger tables | Potpie’s Postgres / Alembic migrations; same DB session as the app |
| GitHub | Potpie’s code provider (`CodeProviderFactory` / tokens your deployment already sets) |

No `CONTEXT_ENGINE_API_KEY`, `CONTEXT_ENGINE_PROJECTS` JSON map, or standalone `DATABASE_URL` is required **for the embedded API**—those exist for the **optional standalone** HTTP server below.

### Using the library from Python (Potpie)

Potpie wires ports in `app/modules/context_graph/wiring.py` (`PotpieContextEngineSettings`, `build_container_for_session`, Celery tasks calling `application.use_cases`, intelligence tools, etc.). Import **`application`**, **`domain`**, **`bootstrap`** from the installed `context-engine` package like any other dependency.

---

## Optional: standalone HTTP service

Use this when you want a **separate** process (e.g. integrating non-Potpie agents via a dedicated host). Not required when Potpie is your only API.

```bash
cd app/src/context-engine
uv sync --all-extras
uv run python -m adapters.inbound.http
```

### Standalone HTTP env

| Variable | Purpose |
|----------|---------|
| `CONTEXT_GRAPH_ENABLED` | `true` to enable graph clients |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | Neo4j (structural + Graphiti) |
| `DATABASE_URL` or `POSTGRES_URL` | Ledger tables (`context_*`, `raw_events`) |
| `CONTEXT_ENGINE_GITHUB_TOKEN` or `GITHUB_TOKEN` | GitHub API |
| `CONTEXT_ENGINE_PROJECTS` | JSON map `{"project-uuid":"owner/repo"}` for sync |
| `CONTEXT_ENGINE_API_KEY` | **Standalone only:** optional; if set, required as header `X-API-Key` |
| `CONTEXT_ENGINE_REPO_TO_PROJECT` | JSON map `{"owner/repo":"project-uuid"}` for GitHub webhook |
| `GITHUB_WEBHOOK_SECRET` | Optional HMAC verification for webhooks |

Routes on the standalone app:

- `GET /api/v1/health`
- `POST /api/v1/context/sync` — inline backfill (not Celery)
- `POST /api/v1/context/ingest-pr` — single PR (inline)
- `POST /api/v1/context/query/*`
- `POST /webhooks/integrations/github`

---

## CLI and MCP (optional integrations)

```bash
cd app/src/context-engine
uv sync --all-extras

# CLI
uv run context-engine doctor

# MCP (stdio) — for external agents, not the Potpie HTTP server
uv run context-engine-mcp
```

### MCP (project scope)

MCP tools accept a `project_id` string. **By default, access is denied** until you configure one of:

| Variable | Purpose |
|----------|---------|
| `CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS` | JSON array of allowed project IDs, e.g. `["uuid-1","uuid-2"]` |
| `CONTEXT_ENGINE_MCP_TRUST_ALL_PROJECTS` | Set to `true` for **development only** — any `project_id` is accepted |

Omit both → tools raise a clear error (no implicit multi-tenant trust).

---

## Package layout

Source is **flat under this directory** as four top-level packages:

- `domain` — pure logic, errors, ports (`Protocol`s)
- `application` — use cases and orchestration services
- `adapters` — inbound (HTTP, CLI, MCP) and outbound (Postgres, Neo4j, Graphiti, GitHub)
- `bootstrap` — `ContextEngineContainer` wiring

Example:

```python
from bootstrap.container import build_container_with_github_token
from bootstrap.http_projects import ExplicitProjectResolution
```

**Note:** Names like `domain` and `adapters` are generic. If this package is installed alongside another project that defines the same top-level names, use a dedicated virtualenv or install only one such package in the environment.

## Dependency rule (hexagonal)

```text
Inbound adapters  →  Application (use cases)  →  Domain (ports + model)
       ↓                         ↓
Outbound adapters  ←  (implements domain ports)
```

- **Domain** does not import FastAPI, Typer, MCP, SQLAlchemy, Graphiti, or Neo4j drivers.
- **Application** depends only on **domain ports**, not concrete adapters.
- **Inbound adapters** map HTTP/CLI/MCP to use cases.
- **Outbound adapters** implement ports; **bootstrap** wires them into `ContextEngineContainer`.

## Tests

```bash
cd app/src/context-engine
uv sync --all-extras --extra dev
uv run pytest
```

From the monorepo root:

```bash
uv run pytest app/src/context-engine/tests/unit/
```

## Packaging

- PyPI name: **context-engine**
- Import roots: **`domain`**, **`application`**, **`adapters`**, **`bootstrap`**

Potpie depends on `context-engine[all]` (path / editable via `[tool.uv.sources]` in the monorepo).

Optional extras: `postgres`, `graph`, `github`, `all`, `dev`.
