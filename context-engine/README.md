# context-engine

**Context graph** service: HTTP API (sync, ingest, queries), GitHub webhooks, CLI, and MCP. Uses **hexagonal (ports and adapters)** so the core stays independent of FastAPI, Typer, MCP, SQLAlchemy, Graphiti, and Neo4j.

## Package layout

Source is **flat under `src/`** as four top-level packages (no extra nesting):

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

## Quick start

```bash
cd context-engine
uv sync --all-extras

# HTTP API
uv run python -m adapters.inbound.http

# CLI
uv run context-engine doctor

# MCP (stdio)
uv run context-engine-mcp
```

### Standalone HTTP env

| Variable | Purpose |
|----------|---------|
| `CONTEXT_GRAPH_ENABLED` | `true` to enable graph clients |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | Neo4j (structural + Graphiti) |
| `DATABASE_URL` or `POSTGRES_URL` | Ledger tables (`context_*`, `raw_events`) |
| `CONTEXT_ENGINE_GITHUB_TOKEN` or `GITHUB_TOKEN` | GitHub API |
| `CONTEXT_ENGINE_PROJECTS` | JSON map `{"project-uuid":"owner/repo"}` for sync |
| `CONTEXT_ENGINE_API_KEY` | Optional; if set, required as header `X-API-Key` |
| `CONTEXT_ENGINE_REPO_TO_PROJECT` | JSON map `{"owner/repo":"project-uuid"}` for GitHub webhook |
| `GITHUB_WEBHOOK_SECRET` | Optional HMAC verification for webhooks |

Routes:

- `GET /api/v1/health`
- `POST /api/v1/context/sync` — backfill (uses `CONTEXT_ENGINE_PROJECTS`)
- `POST /api/v1/context/ingest-pr` — single PR
- `POST /api/v1/context/query/*` — change history, file owners, decisions, Graphiti search
- `POST /webhooks/integrations/github` — merged PR webhook

## Potpie integration

Potpie depends on `context-engine[all]` (path / editable via `[tool.uv.sources]`). Host-specific wiring lives in `app/modules/context_graph/wiring.py`. Celery tasks delegate to `application.use_cases`.

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
cd context-engine
uv sync --all-extras --extra dev
uv run pytest
```

From the monorepo root:

```bash
uv run pytest context-engine/tests/unit/
```

## Packaging

- PyPI name: **context-engine**
- Import roots: **`domain`**, **`application`**, **`adapters`**, **`bootstrap`** (under `src/`)

Optional extras: `postgres`, `graph`, `github`, `all`, `dev`.
