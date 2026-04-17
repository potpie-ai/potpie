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
| `POST` | `/api/v1/context/sync` | Enqueues backfill via the configured job queue (optional body: `{ "pot_ids": ["..."] }`; omit to sync all **your** eligible pots) |
| `POST` | `/api/v1/context/ingest-pr` | Enqueues single-PR ingest (`pot_id`, `pr_number`, optional `is_live_bridge`) |
| `POST` | `/api/v1/context/reset` | Hard-delete all context-graph data for a pot (`pot_id` in JSON body; Graphiti + structural Neo4j + Postgres ledger when DB is configured) |
| `POST` | `/api/v1/context/query/change-history` | Neo4j change history |
| `POST` | `/api/v1/context/query/file-owners` | File owner hints from PR history |
| `POST` | `/api/v1/context/query/decisions` | Linked decisions |
| `POST` | `/api/v1/context/query/search` | Graphiti semantic search |

**Pot scope:** Requests only apply to pots the authenticated user may access (enforced in Potpie’s wiring).

**Long-running work:** Sync and ingest use **`ContextGraphJobQueuePort`** (**default: Celery** on the `context-graph-etl` queue). Optional **Hatchet** (`CONTEXT_GRAPH_JOB_QUEUE_BACKEND=hatchet`): install **`hatchet-sdk`**, set **`HATCHET_CLIENT_*`**, self-host Hatchet per **[`docs/hatchet-local.md`](../../../docs/hatchet-local.md)**, and run `python -m app.modules.context_graph.hatchet_worker`. See `bootstrap/queue_factory.py`.

### Configuration (Potpie)

Context graph reads **Potpie’s config** (via `config_provider`), not a parallel auth layer:

| Concern | Notes |
|---------|--------|
| Feature flag | `CONTEXT_GRAPH_ENABLED=true` (and related Potpie env you already use) |
| Neo4j / Graphiti | Default: same as Potpie (`NEO4J_*` via config). Set **`CONTEXT_ENGINE_NEO4J_URI`** (and username/password) to use a **dedicated** Neo4j for Graphiti + context-engine structural queries instead of the code graph DB. |
| Ledger tables | Potpie’s Postgres / Alembic migrations; same DB session as the app |
| GitHub | Potpie’s code provider (`CodeProviderFactory` / tokens your deployment already sets) |

Standalone-only configuration (`CONTEXT_ENGINE_API_KEY`, `CONTEXT_ENGINE_POTS` JSON map, or a separate `DATABASE_URL`) is **not** required **for the embedded API**—those exist for the **optional standalone** HTTP server below.

### Using the library from Python (Potpie)

Potpie wires ports in `app/modules/context_graph/wiring.py` (`PotpieContextEngineSettings`, `build_container_for_session`, job queue + Celery tasks calling `application.use_cases`, intelligence tools, etc.). Import **`application`**, **`domain`**, **`bootstrap`** from the installed `context-engine` package like any other dependency.

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
| `DATABASE_URL` or `POSTGRES_URL` | Ledger tables (`context_events`, `context_sync_state`, …) |
| `CONTEXT_ENGINE_GITHUB_TOKEN` or `GITHUB_TOKEN` | GitHub API |
| `CONTEXT_ENGINE_POTS` | JSON map `{"pot-uuid":"owner/repo"}` for sync |
| `CONTEXT_ENGINE_API_KEY` | **Standalone only:** optional; if set, required as header `X-API-Key` |
| `CONTEXT_ENGINE_REPO_TO_POT` | JSON map `{"owner/repo":"pot-uuid"}` for GitHub webhook |
| `GITHUB_WEBHOOK_SECRET` | Optional HMAC verification for webhooks |

Routes on the standalone app:

- `GET /api/v1/health`
- `POST /api/v1/context/sync` — inline backfill (not Celery)
- `POST /api/v1/context/ingest-pr` — single PR (inline)
- `POST /api/v1/context/reset` — hard-delete graph + ledger for a pot
- `POST /api/v1/context/query/*`
- `POST /webhooks/integrations/github`

**Errors:** `HTTPException` with a string `detail` is returned as JSON `{"error":{"code":"http_error","message":"..."}}`. Unhandled server errors return `500` with `{"error":{"code":"internal_error","message":"Internal server error"}}` (details are logged server-side).

---

## CLI and MCP (optional integrations)

```bash
cd app/src/context-engine
uv sync --all-extras

# CLI
uv run potpie doctor

# MCP (stdio) — for external agents, not the Potpie HTTP server
uv run potpie-mcp
```

### MCP (pot scope)

MCP tools accept a `pot_id` string. **By default, access is denied** until you configure one of:

| Variable | Purpose |
|----------|---------|
| `CONTEXT_ENGINE_MCP_ALLOWED_POTS` | JSON array of allowed pot IDs, e.g. `["uuid-1","uuid-2"]` |
| `CONTEXT_ENGINE_MCP_TRUST_ALL_POTS` | Set to `true` for **development only** — any `pot_id` is accepted |

Omit both → tools raise a clear error (no implicit multi-tenant trust).

### Raw episode ingest (HTTP / CLI / MCP)

All three use **`application.use_cases.run_raw_episode_ingestion`**: persist **`context_events`** when Postgres is configured, then **enqueue** apply by default (**async**). Use **`sync=true`** (HTTP query), **`--sync`** (CLI), or **`sync=true`** on the MCP tool for inline apply after persist; without Postgres, **sync** is required and performs a **legacy** direct Graphiti write (no event row).

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
from bootstrap.http_projects import ExplicitPotResolution
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

All unit tests for this package live under **`tests/`** here. Do **not** add a separate `tests/unit/context_graph/` tree at the Potpie repo root—that duplicated coverage and is removed.

```bash
cd app/src/context-engine
uv sync --all-extras --extra dev
uv run pytest
```

From the monorepo root:

```bash
uv run pytest app/src/context-engine/tests/unit/
```

## Local lab / smoke testing

Use the bundled lab harness for quick mock, in-process HTTP, and live API checks:

```bash
uv run python app/src/context-engine/scripts/context_engine_lab.py mock-e2e
uv run python app/src/context-engine/scripts/context_engine_lab.py http-e2e
uv run python app/src/context-engine/scripts/context_engine_lab.py api-smoke
uv run python app/src/context-engine/scripts/context_engine_lab.py api-smoke --write
```

The mock mode runs context resolution directly with deterministic data and no external services. The HTTP mode mounts the context API router in-process with in-memory adapters, so it exercises status, ingest, search, resolve, record normalization, and reset without a Potpie server or API key. The live API mode uses the same Potpie URL/key resolution as the CLI and writes a report under `app/src/context-engine/.tmp/`. Findings and open bugs are tracked in [`docs/context-graph/testing-and-bugs.md`](../../../docs/context-graph/testing-and-bugs.md).

## Packaging

- PyPI name: **context-engine**
- Import roots: **`domain`**, **`application`**, **`adapters`**, **`bootstrap`**

Potpie depends on `potpie[all]` (path / editable via `[tool.uv.sources]` in the monorepo).

Optional extras: `postgres`, `graph`, `github`, `all`, `dev`.
