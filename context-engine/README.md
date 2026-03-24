# context-engine

Scaffold for a **context graph** service: HTTP API (context + webhooks), CLI, and MCP servers. The codebase follows **hexagonal (ports and adapters)** so the core stays independent of FastAPI, Typer, MCP, databases, and external integrations.

This document is the **placement guide** for future work: put each kind of code in the layer that owns it, and keep dependencies pointing **inward** (adapters → application → domain).

## Layout

Source lives under **`src/`** as four top-level Python packages (no extra nesting):

- `domain/` — core vocabulary and ports
- `application/` — use cases
- `adapters/` — inbound (HTTP, CLI, MCP) and outbound (clients, DB, …)
- `bootstrap/` — wiring / composition

Imports look like `from adapters.inbound.http.app import create_app` or `from domain.ports.context_graph import ContextGraphPort`.

These names are short on purpose. If you ever publish this as a library on PyPI, consider prefixing packages (e.g. `ce_domain`) to avoid clashing with other projects in the same environment.

## Quick start

```bash
cd context-engine
uv sync

# HTTP API (defaults: 127.0.0.1:8000)
uv run python -m adapters.inbound.http

# CLI
uv run context-engine --help

# MCP server (stdio)
uv run context-engine-mcp
```

Optional environment variables for the HTTP server: `CONTEXT_ENGINE_HOST`, `CONTEXT_ENGINE_PORT`, `CONTEXT_ENGINE_RELOAD`.

## Dependency rule (hexagonal)

```text
Inbound adapters  →  Application (use cases)  →  Domain (ports + model)
       ↓                         ↓
Outbound adapters  ←  (implements domain ports)
```

- **Domain** must not import FastAPI, Typer, MCP, `httpx`, or framework types.
- **Application** depends only on **domain ports** (protocols), not on concrete adapters.
- **Inbound adapters** translate HTTP / CLI / MCP into calls to use cases (orchestration lives in application).
- **Outbound adapters** implement ports (storage, external APIs, message buses) and are wired in **bootstrap**.

## Where code goes

| You are adding… | Put it here |
|-----------------|-------------|
| Business vocabulary: entities, value objects, domain errors, invariants | `src/domain/` (new modules as needed; today only `ports/` is populated) |
| **Interfaces** the core needs from the outside world (e.g. “load graph slice”, “persist event”) | `src/domain/ports/` — one focused module per capability (e.g. `context_graph.py`) |
| Orchestration: “fetch context for query”, “apply webhook to graph” | `src/application/use_cases/` — functions or small classes that take port implementations (injected), not concrete DB/API clients |
| REST routes for **clients** (query context, admin, etc.) | `src/adapters/inbound/http/api/` — prefer `api/v1/<feature>/router.py`; aggregate mounts in `api/router.py` |
| **Webhook** endpoints per integration (Slack, GitHub, …) | `src/adapters/inbound/http/webhooks/` — shared webhook wiring in `webhooks/router.py`; per-integration routers under `webhooks/integrations/` |
| Typer commands | `src/adapters/inbound/cli/` |
| MCP tools / resources | `src/adapters/inbound/mcp/tools/` (and register from `adapters/inbound/mcp/server.py` or a dedicated registrar module) |
| HTTP client to **this** service (CLI/MCP calling the API) | `src/adapters/outbound/api_client/` |
| Database, graph DB, queues, third-party SDKs | `src/adapters/outbound/<name>/` — new package per integration or store |
| Composition: construct implementations, bind ports, single place for “the app” wiring | `src/bootstrap/container.py` (grow into a small DI/wiring module as needed) |

## HTTP layout (current)

- **API base:** `/api/v1` — public/context APIs.
- **Health:** `GET /api/v1/health`.
- **Webhooks:** `/webhooks` — integration ingress; keep payloads thin here and delegate to use cases.

## Tests

```bash
cd context-engine
uv sync --extra dev
uv run pytest
```

Add tests under `tests/`, mirroring layers where it helps: unit tests for domain/application, adapter tests with mocks or HTTP client against `create_app()`.

## Packaging

- Installable name: **context-engine** (`pyproject.toml`).
- Import roots: **`domain`**, **`application`**, **`adapters`**, **`bootstrap`** under `src/`.

When in doubt: **domain** = what must stay stable; **application** = how you coordinate it; **adapters** = how the world talks to the core or how the core talks to the world.
