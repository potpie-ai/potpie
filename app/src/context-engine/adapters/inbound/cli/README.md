# context-engine CLI

Optional command-line entrypoint for the context graph. It reads the same **environment variables** as the standalone HTTP service and MCP (Neo4j / Graphiti, feature flags, etc.). Use it for quick checks, scripted search, or one-off episodic ingestion without starting the API.

The console script is registered as **`context-engine`** (see `pyproject.toml` → `[project.scripts]`).

## Setup

From `app/src/context-engine` (the package root):

```bash
uv sync --all-extras
```

Graphiti/Neo4j extras are required for **`search`** and **`ingest`** (same as production).

**Dedicated Neo4j for context-engine (recommended):** set **`CONTEXT_ENGINE_NEO4J_URI`**, **`CONTEXT_ENGINE_NEO4J_USERNAME`**, **`CONTEXT_ENGINE_NEO4J_PASSWORD`** so Graphiti uses a separate database/cluster from any other Neo4j usage. If unset, the CLI falls back to **`NEO4J_*`**.

## Global options

These apply to all subcommands (place them **before** the command name):

| Option | Description |
|--------|-------------|
| `--version` | Print package version and exit. |
| `--json` | Print machine-readable JSON on stdout (for scripting and pipes). Human-friendly tables/panels are the default. |
| `--verbose` / `-v` | Raise Neo4j driver log verbosity (use with `CONTEXT_ENGINE_VERBOSE_NEO4J=1` for persistent debug). By default the CLI quiets noisy Neo4j notification logs. |

Examples:

```bash
context-engine --json doctor
context-engine -v search "query here"
```

## Commands

### `login` / `logout`

Optional: persist a **Potpie API key** for future HTTP integrations. **Pot scope for `search` / `ingest` does not use the Potpie projects API** — use env maps, **`context-engine pot use`**, or an explicit pot UUID. Credentials are stored under **`$XDG_CONFIG_HOME/context-engine/credentials.json`**, or **`~/.config/context-engine/credentials.json`** when `XDG_CONFIG_HOME` is unset, mode **600**.

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

Semantic search over **Graphiti episodic** entities for a pot. Matches **`POST /api/v1/context/query/search`** and the MCP tool **`context_search`**.

**Arguments**

| Form | Meaning |
|------|---------|
| One argument (`FIRST`) | `FIRST` is the **query**; the pot scope UUID is **inferred** from the current directory’s git `origin` (see below). |
| Two arguments (`FIRST` + `SECOND`) | `FIRST` is **pot UUID**, `SECOND` is the **query** (same as before). |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` / `-n` | `8` | Max results (clamped 1–50 in the use case). |
| `--node-labels` | (none) | Comma-separated label filters, e.g. `PullRequest,Decision`. |

**Inferring pot scope from git (one-argument form, and optional `ingest` pot)**

1. Run inside a git checkout with **`origin`** set (e.g. `git@github.com:owner/repo.git`).
2. Resolve `owner/repo` → pot UUID in order:
   - **`CONTEXT_ENGINE_REPO_TO_POT`** — JSON `{"owner/repo":"pot-uuid"}`, or
   - **`CONTEXT_ENGINE_POTS`** — JSON `{"pot-uuid":"owner/repo"}` (case-insensitive match on value).
   - Else **`context-engine pot use <uuid>`** (stored default pot for this machine).
   - Else exit with code `1` — pass an explicit pot UUID, or set maps / `pot use`.

Pot scope is **not** tied to Potpie projects in the CLI; the app/agent resolves repo → pot separately when using tools.

**Output:** JSON array of objects with `uuid`, `name`, `summary`, `fact`.

**Exit codes:** `0` on success; `1` if the context graph is disabled (`CONTEXT_GRAPH_ENABLED`), pot inference failed, or Graphiti is unavailable.

Examples:

```bash
# Default pot for this machine (optional)
context-engine pot use 00000000-0000-0000-0000-000000000000

# From a repo directory with env mapping or active pot (see above)
uv run context-engine search "how is auth handled?" -n 10

# Explicit pot UUID
uv run context-engine search "00000000-0000-0000-0000-000000000000" "how is auth handled?" -n 10
```

### `ingest`

Adds a **raw episode** to the episodic graph via Graphiti. Matches **`POST /api/v1/context/ingest`** on the HTTP API.

**Arguments**

| Argument | Description |
|----------|-------------|
| `POT_ID` | Optional. Pot scope UUID (Graphiti `group_id`). Omit to infer from **git `origin`** using the same resolution order as **`search`** (env maps, then **`pot use`**). |

**Options**

| Option | Required | Description |
|--------|----------|-------------|
| `--name` / `-n` | Yes | Episode title. |
| `--episode-body` / `-b` | Yes | Main episode body text. |
| `--source` / `-s` | Yes | Short source description / label. |
| `--reference-time` / `-t` | No | ISO 8601 timestamp; defaults to **UTC now**. `Z` suffix is accepted. |

**Output:** JSON object `{"episode_uuid": "<uuid>"}`.

**Exit codes:** `0` on success; `1` if the context graph is disabled, reference time is invalid, pot inference failed, or Graphiti returns no episode UUID.

**Note:** The HTTP API may validate `pot_id` against host-specific resolution. The CLI infers from git + env + `pot use` when `POT_ID` is omitted; use an explicit UUID when needed.

Examples:

```bash
# Infer pot from current git repo + env
uv run context-engine ingest \
  --name "Design note" \
  --episode-body "We chose Postgres for the ledger." \
  --source "cli" \
  --reference-time "2025-03-27T12:00:00Z"

# Explicit pot UUID
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
