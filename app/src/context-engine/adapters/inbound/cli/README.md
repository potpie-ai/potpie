# Potpie CLI

Optional command-line entrypoint for the context graph. **`search`**, **`ingest`**, and **`pot hard-reset`** call your **Potpie** server at **`POST /api/v2/context/*`** with **`X-API-Key`** (configure **`POTPIE_API_URL`** + **`POTPIE_API_KEY`** or **`potpie login`**). You do **not** need local Neo4j/Graphiti for those commands. Local env maps and **`pot use`** still resolve which `pot_id` to send.

The console script is registered as **`potpie`** (see `pyproject.toml` → `[project.scripts]`).

## Setup

From `app/src/context-engine` (the package root):

```bash
uv sync --all-extras
```

Install **`httpx`** (included in package dependencies). Graphiti/Neo4j extras are **not** required on the machine running the CLI for **`search`** / **`ingest`** / **`pot hard-reset`** (the server holds Neo4j/Graphiti). Optional: install **`--extra graph`** only if you run standalone context-engine HTTP or other local tools that touch Graphiti directly.

## Global options

These apply to all subcommands (place them **before** the command name):

| Option | Description |
|--------|-------------|
| `--version` | Print package version and exit. |
| `--json` | Print machine-readable JSON on stdout (for scripting and pipes). Human-friendly tables/panels are the default. |
| `--verbose` / `-v` | Verbose errors (tracebacks on API failures). |

When `--json` is enabled, command results stay on stdout and errors are emitted as JSON objects on stderr.

Examples:

```bash
potpie --json doctor
potpie -v search "query here"
```

## Commands

### `init-agent`

Install the packaged agent bundle into the current repository. This writes a top-level `AGENTS.md` plus the repo-local `.agents/skills/potpie-*` files used by Codex-style agent workflows.

The bundle includes a dedicated `potpie-agent-context` skill for MCP workflows. That skill keeps the public agent surface to `context_resolve`, `context_search`, `context_record`, and `context_status`, and encodes feature, debugging, review, operations, docs, and onboarding workflows as `context_resolve` parameter recipes.

| Command | Description |
|---------|-------------|
| `potpie init-agent` | Install into the current repo root (auto-detected from the current directory). |
| `potpie init-agent path/to/repo` | Install into a specific repository or a subdirectory inside it. |
| `potpie init-agent --force` | Overwrite existing agent files when local contents differ. |

By default, the command is safe: existing files with different contents are skipped and reported. Use `--json` for automation or `--force` to refresh local copies after upgrading the CLI.

### `login` / `logout`

Persist a **Potpie API key** and optional base URL — **required** for **`search`**, **`ingest`**, and **`pot hard-reset`** unless you set **`POTPIE_API_KEY`** / URL in the environment. **Pot scope** still comes from env maps, **`potpie pot use`**, or an explicit pot UUID. Credentials live under **`$XDG_CONFIG_HOME/potpie/credentials.json`**, or **`~/.config/potpie/credentials.json`**, mode **600**.

| Command | Description |
|---------|-------------|
| `potpie login TOKEN` | Save the token. |
| `potpie login TOKEN --url http://127.0.0.1:8001` | Save token and default Potpie base URL (no trailing slash). |
| `potpie logout` | Delete the stored credentials file (API key, base URL, active pot, aliases). |
| `potpie pot clear-local` | Clear only active pot + `pot_aliases`; keep API key / URL. |
| `potpie pot create <slug>` | Check slug availability, then `POST /api/v2/context/pots` + local alias. |
| `potpie pot slug-available <slug>` | Check whether a pot slug is available on the server. |
| `potpie pot pots` | List context pots (`GET /api/v2/context/pots`). |
| `potpie pot repo list` / `pot repo add` | List or attach repositories on a pot (`GET`/`POST` `/api/v2/context/pots/.../repositories`). |
| `potpie event list` / `event show` / `event wait` | Inspect recent ingestion events, fetch one event, or wait for a queued ingest to finish. |

**Precedence:** `POTPIE_API_KEY` in the environment **overrides** the stored token (useful for CI). For base URL: **`POTPIE_API_URL` / `POTPIE_BASE_URL`** override a stored URL; otherwise the stored URL from `login --url` is used before `POTPIE_PORT` and localhost guesses.

### `doctor`

Shows Potpie API URL/key presence, local pot maps, **`GET /health`** on the Potpie host, and an authenticated **`GET /api/v2/context/pots`** probe when URL+key are set. Local Neo4j/Graphiti lines are informational only for other workflows.

```bash
uv run potpie doctor
```

### `pot hard-reset`

**Destructive:** calls Potpie **`POST /api/v2/context/reset`** (same behavior as v1 reset on the server). **`--skip-ledger`** sets `skip_ledger: true` in the JSON body.

| Option | Description |
|--------|-------------|
| `POT_ID` | Optional. Omit to infer scope like **`search`** / **`ingest`** (`pot use`, env maps, git `origin` under `--cwd`). |
| `--skip-ledger` | Do not delete Postgres ledger rows for the pot (Neo4j is still cleared). |
| `--cwd` | Git tree used when inferring pot (default: current directory). |

```bash
uv run potpie pot hard-reset
uv run potpie pot hard-reset --skip-ledger
uv run potpie --json pot hard-reset 00000000-0000-0000-0000-000000000000
```

### `search`

Semantic search over **Graphiti episodic** entities for a pot via **`POST /api/v2/context/query/search`**. This CLI command returns raw search rows; the MCP **`context_search`** tool wraps those rows in the common agent context envelope.

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

**Inferring pot scope (one-argument form, and optional `ingest` pot)**

1. Use the active pot from **`potpie pot use <slug-or-uuid>`** when set.
2. Otherwise, run inside a git checkout with **`origin`** set (e.g. `git@github.com:owner/repo.git`) and resolve `owner/repo` → pot UUID from:
   - **`CONTEXT_ENGINE_REPO_TO_POT`** — JSON `{"owner/repo":"pot-uuid"}`, or
   - **`CONTEXT_ENGINE_POTS`** — JSON `{"pot-uuid":"owner/repo"}` (case-insensitive match on value).
3. Else exit with code `1` — pass an explicit pot UUID / registered name, set maps, or run **`potpie pot use`**.

Pot scope is chosen explicitly (**`potpie pot use`**, maps, or a UUID argument); repo → pot mapping uses your server pots and attached repositories.

**Output:** JSON array of objects with `uuid`, `name`, `summary`, `fact`.

**Exit codes:** `0` on success; `1` if Potpie API is not configured, request failed (401/404/503, etc.), or pot inference failed.

Examples:

```bash
# Context pot (recommended): created on the server
potpie pot create my-workspace
potpie pot use my-workspace

# Or list existing context pots and pick an id
potpie pot pots
potpie pot use my-workspace
potpie pot use 00000000-0000-0000-0000-000000000000

# Optional: short name → id (stored in credentials.json)
potpie pot alias my-alias 00000000-0000-0000-0000-000000000000

# From a repo directory with env mapping or active pot (see above)
uv run potpie search "how is auth handled?" -n 10

# Explicit pot UUID
uv run potpie search "00000000-0000-0000-0000-000000000000" "how is auth handled?" -n 10
```

### `ingest`

Sends **`POST /api/v2/context/ingest`** with your API key. The **server** persists events, enqueues Celery/Hatchet when configured, or applies inline when you pass **`--sync`** (query `sync=true`, same as browser/API clients on v1/v2).

**Arguments**

| Argument | Description |
|----------|-------------|
| `POT_ID` | Optional. Pot scope UUID (Graphiti `group_id`). Omit to infer using the same resolution order as **`search`** (**`potpie pot use`**, then env maps from git `origin`). |

**Options**

| Option | Required | Description |
|--------|----------|-------------|
| `--name` / `-n` | Yes | Episode title. |
| `--episode-body` / `-b` | Yes | Main episode body text. |
| `--file` / `-f` | No | Read episode body from this path (UTF-8). Cannot be combined with `-b` or inline episode text (except an optional leading pot UUID). |
| `--source` / `-s` | Yes | Short source description / label. |
| `--reference-time` / `-t` | No | ISO 8601 timestamp; defaults to **UTC now**. `Z` suffix is accepted. |
| `--sync` | No | Maps to HTTP **`sync=true`** (inline apply on the server when supported). |
| `--idempotency-key` | No | Optional dedupe key (forwarded to the API). |

**Output (JSON):** includes `status` (`queued` \| `applied` \| `legacy_direct`), `episode_uuid` (when known), and `event_id` / `job_id` when the event store path is used.

When async ingest returns `queued`, use `potpie event wait <event_id>` to block until the server reports `done` or `error`, or `potpie event show <event_id>` to inspect the current event state.

**Exit codes:** `0` on success; `1` if API misconfigured, reference time is invalid, pot inference failed, duplicate ingest (409), or server error.

**Note:** The HTTP API may validate `pot_id` against host-specific resolution. The CLI infers from git + env + `pot use` when `POT_ID` is omitted; use an explicit UUID when needed.

Examples:

```bash
# Infer pot from current git repo + env
uv run potpie ingest \
  --name "Design note" \
  --episode-body "We chose Postgres for the ledger." \
  --source "cli" \
  --reference-time "2025-03-27T12:00:00Z"

# Explicit pot UUID
uv run potpie ingest "00000000-0000-0000-0000-000000000000" \
  --name "Design note" \
  --episode-body "We chose Postgres for the ledger." \
  --source "cli" \
  --reference-time "2025-03-27T12:00:00Z"

# Episode body from a file (paths relative to the shell cwd)
uv run potpie ingest --file ./notes/design.md --name "Design note" --source "cli"
```

### `event`

Inspect ingestion events created by async ingest and reconciliation flows.

| Command | Description |
|---------|-------------|
| `potpie event show EVENT_ID` | Fetch one persisted ingestion event, including episode steps when available. |
| `potpie event wait EVENT_ID` | Poll until the event reaches `done` or `error`; exits `1` on timeout or error. |
| `potpie event list` | List recent events for the active/inferred pot. |

Options:

| Option | Command | Description |
|--------|---------|-------------|
| `--timeout` | `event wait` | Max seconds to wait (default `60`). |
| `--interval` | `event wait` | Poll interval in seconds (default `2`). |
| `--limit` / `-n` | `event list` | Max events to show (default `20`). |
| `--status` | `event list` | Filter by `queued`, `processing`, `done`, or `error`. |
| `--kind` | `event list` | Filter by ingestion kind, e.g. `raw_episode`. |

Examples:

```bash
uv run potpie ingest "Design note" --name "Design note" --source cli
uv run potpie event wait 00000000-0000-0000-0000-000000000000
uv run potpie event list --status done -n 10
```

## Related

- Full service and env tables: repository root `app/src/context-engine/README.md`.
- CLI targets **`/api/v2/context/`** with **`X-API-Key`**. **`/api/v1/context/`** remains for Firebase-authenticated clients.
- MCP entrypoint: `potpie-mcp` (stdio server).

### MCP agent context port

MCP exposes the minimal context port:

| Tool | Use |
|------|-----|
| `context_resolve` | Primary bounded context wrap for a task. |
| `context_status` | Cheap pot/scope readiness, freshness gaps, and recommended recipe. |
| `context_search` | Narrow follow-up lookup after `context_resolve`. |
| `context_record` | Durable learnings: decisions, fixes, preferences, workflows, feature notes, doc refs, and incident summaries. |

Use `context_resolve` recipes instead of adding separate tools per context type:

```json
{"intent":"feature","include":["purpose","feature_map","service_map","docs","tickets","decisions","recent_changes","owners","preferences","source_status"],"mode":"fast","source_policy":"references_only"}
```

```json
{"intent":"debugging","include":["prior_fixes","diagnostic_signals","incidents","alerts","recent_changes","config","deployments","owners","source_status"],"mode":"fast","source_policy":"references_only"}
```

```json
{"intent":"review","include":["artifact","discussions","owners","recent_changes","decisions","preferences","source_status"],"mode":"balanced","source_policy":"summary"}
```

```json
{"intent":"operations","include":["deployments","runbooks","alerts","incidents","scripts","config","owners","source_status"],"mode":"balanced","source_policy":"summary"}
```

Agents should inspect `coverage`, `freshness`, `quality`, `fallbacks`, `open_conflicts`, and `source_refs` before relying on graph memory. Start with `mode=fast` and `source_policy=references_only`; escalate to `summary`, `verify`, `snippets`, or `deep` only when risk, missing coverage, or degraded quality calls for it.
