# context-engine CLI

Optional command-line entrypoint for the context graph. It reads the same **environment variables** as the standalone HTTP service and MCP (Neo4j / Graphiti, feature flags, etc.). Use it for quick checks, scripted search, or one-off episodic ingestion without starting the API.

The console script is registered as **`context-engine`** (see `pyproject.toml` → `[project.scripts]`).

## Setup

From `app/src/context-engine` (the package root):

```bash
uv sync --all-extras
```

Graphiti/Neo4j extras are required for **`search`** and **`ingest`** (same as production).

## Global options

| Option | Description |
|--------|-------------|
| `--version` | Print package version and exit. |

## Commands

### `login` / `logout`

Persist a **Potpie API key** (create one in the app) for **`GET /api/v2/projects/list`** and any future Potpie HTTP calls from the CLI. The file is written under **`$XDG_CONFIG_HOME/context-engine/credentials.json`**, or **`~/.config/context-engine/credentials.json`** when `XDG_CONFIG_HOME` is unset, with mode **600** (user read/write only).

| Command | Description |
|---------|-------------|
| `context-engine login TOKEN` | Save the token. |
| `context-engine login TOKEN --url http://127.0.0.1:8001` | Save token and default Potpie base URL (no trailing slash). |
| `context-engine logout` | Delete the stored credentials file. |

**Precedence:** `POTPIE_API_KEY` in the environment **overrides** the stored token (useful for CI). For base URL: **`POTPIE_API_URL` / `POTPIE_BASE_URL`** override a stored URL; otherwise the stored URL from `login --url` is used before `POTPIE_PORT` and localhost guesses.

### `doctor`

Prints a short snapshot of whether common env vars look set (feature flag, Neo4j URI, Potpie API token source, GitHub token). Does not validate connectivity to Neo4j or Graphiti.

```bash
uv run context-engine doctor
```

### `search`

Semantic search over **Graphiti episodic** entities for a project. Matches **`POST /api/v1/context/query/search`** (and **`/query/get-project-context`**) and the MCP tools `context_search` / `get_project_context`.

**Arguments**

| Form | Meaning |
|------|---------|
| One argument (`FIRST`) | `FIRST` is the **query**; the project UUID is **inferred** from the current directory’s git `origin` (see below). |
| Two arguments (`FIRST` + `SECOND`) | `FIRST` is **project UUID**, `SECOND` is the **query** (same as before). |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` / `-n` | `8` | Max results (clamped 1–50 in the use case). |
| `--node-labels` | (none) | Comma-separated label filters, e.g. `PullRequest,Decision`. |

**Inferring project from git (one-argument form, and optional `ingest` project)**

1. Run inside a git checkout with **`origin`** set (e.g. `git@github.com:owner/repo.git`).
2. Resolve `owner/repo` → project UUID in order:
   - **`CONTEXT_ENGINE_REPO_TO_PROJECT`** — JSON `{"owner/repo":"project-uuid"}` (standalone / webhooks), or
   - **`CONTEXT_ENGINE_PROJECTS`** — JSON `{"project-uuid":"owner/repo"}` (inverse map; case-insensitive match on value).
   - **Potpie API** (if both are unset or no match): the CLI loads the nearest **`.env`** (for env vars), then calls **`GET /api/v2/projects/list`** with **`X-API-Key`** to match `repo_name` to `git`’s `owner/repo`.
     - **Auth:** **`POTPIE_API_KEY`** in the environment, or a token saved with **`context-engine login`** (see above). The key identifies **one** user; no extra headers.
     - **Base URL:** **`POTPIE_API_URL`** / **`POTPIE_BASE_URL`** if set; else URL from **`login --url`**; else **`POTPIE_PORT`**; else **`http://127.0.0.1:8000`** then **`http://127.0.0.1:8001`**.

If `origin` cannot be read or no mapping/API match exists, the command exits with code `1` and an error message.

**Output:** JSON array of objects with `uuid`, `name`, `summary`, `fact`.

**Exit codes:** `0` on success; `1` if the context graph is disabled (`CONTEXT_GRAPH_ENABLED`), project inference failed, or Graphiti is unavailable.

Examples:

```bash
# One-time: save API key (from Potpie app) and optional base URL
context-engine login "$POTPIE_TOKEN" --url http://127.0.0.1:8001

# From a repo directory with env mapping or Potpie auth (see above)
uv run context-engine search "how is auth handled?" -n 10

# Explicit project UUID
uv run context-engine search "00000000-0000-0000-0000-000000000000" "how is auth handled?" -n 10
```

### `ingest`

Adds a **raw episode** to the episodic graph via Graphiti. Matches **`POST /api/v1/context/ingest`** on the HTTP API.

**Arguments**

| Argument | Description |
|----------|-------------|
| `PROJECT_ID` | Optional. Project UUID (Graphiti `group_id`). Omit to infer from **git `origin`** using the same resolution order as **`search`** (env maps, then Potpie API with `POTPIE_API_KEY` or **`context-engine login`**). |

**Options**

| Option | Required | Description |
|--------|----------|-------------|
| `--name` / `-n` | Yes | Episode title. |
| `--episode-body` / `-b` | Yes | Main episode body text. |
| `--source` / `-s` | Yes | Short source description / label. |
| `--reference-time` / `-t` | No | ISO 8601 timestamp; defaults to **UTC now**. `Z` suffix is accepted. |

**Output:** JSON object `{"episode_uuid": "<uuid>"}`.

**Exit codes:** `0` on success; `1` if the context graph is disabled, reference time is invalid, project inference failed, or Graphiti returns no episode UUID.

**Note:** The HTTP API validates `project_id` against configured project resolution (e.g. `CONTEXT_ENGINE_PROJECTS` in standalone, or user-scoped projects in Potpie). The CLI infers from git + env only when `PROJECT_ID` is omitted; use an explicit UUID when you are not in a mapped repo.

Examples:

```bash
# Infer project from current git repo + env
uv run context-engine ingest \
  --name "Design note" \
  --episode-body "We chose Postgres for the ledger." \
  --source "cli" \
  --reference-time "2025-03-27T12:00:00Z"

# Explicit project UUID
uv run context-engine ingest "00000000-0000-0000-0000-000000000000" \
  --name "Design note" \
  --episode-body "We chose Postgres for the ledger." \
  --source "cli" \
  --reference-time "2025-03-27T12:00:00Z"
```

## Related

- Full service and env tables: repository root `app/src/context-engine/README.md`.
- HTTP routes live under `/api/v1/context/` (e.g. `/ingest`, `/query/search`).
- MCP entrypoint: `context-engine-mcp` (stdio server).
