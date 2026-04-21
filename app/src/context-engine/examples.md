# context-engine — command examples

Run these from the **package root** so `uv` resolves the editable package and extras:

```bash
cd app/src/context-engine
uv sync --all-extras
```

Prefer **`uv run potpie …`** (or activate the venv and run `potpie` directly). Graphiti/Neo4j-related commands need the same env as production (`CONTEXT_GRAPH_ENABLED`, Neo4j, etc.); see [`README.md`](README.md).

---

## Global options

Place **before** the subcommand:

| Flag | Purpose |
|------|---------|
| `--version` | Print package version and exit. |
| `--json` | Machine-readable JSON on stdout (scripting). |
| `--verbose` / `-v` | Verbose logging (e.g. Neo4j driver). |
| `--source` / `-s` | Default source label for ingest (and optional search filter). Subcommand `--source` wins if both are set. |

```bash
uv run potpie --version
uv run potpie --json doctor
uv run potpie -v search "example query"
```

Built-in help:

```bash
uv run potpie --help
uv run potpie search --help
uv run potpie pot --help
```

---

## `doctor`

Shows whether common **environment variables look set** (feature flag, Neo4j, Potpie API token source, GitHub token). It does **not** test connectivity to Neo4j or Graphiti.

```bash
uv run potpie doctor
uv run potpie --json doctor
```

---

## `login` / `logout`

Optional: store a Potpie API key (and optional base URL) for future HTTP use. CLI **pot scope** for `search` / `ingest` still uses **`pot use`**, env maps, or git `origin` — not the projects API.

```bash
uv run potpie login YOUR_API_KEY
uv run potpie login YOUR_API_KEY --url http://127.0.0.1:8001
uv run potpie logout
```

`POTPIE_API_KEY` in the environment overrides the stored token.

---

## `init-agent`

Installs `AGENTS.md` and repo-local `.agents/skills/potpie-*` into a repository (safe by default; skips differing files unless `--force`).
The installed `potpie-agent-context` skill teaches agents to use the four-tool context port and `context_resolve` recipes instead of one-off tools for every context type.

```bash
uv run potpie init-agent
uv run potpie init-agent /path/to/repo
uv run potpie init-agent --force
uv run potpie --json init-agent .
```

---

## `pot`

### `pot use` / `pot unset`

Remember or clear the **default pot UUID** for this machine (stored with credentials). **`pot use` takes precedence** over `CONTEXT_ENGINE_REPO_TO_POT` / `CONTEXT_ENGINE_POTS` when inferring scope.

```bash
uv run potpie pot use 00000000-0000-0000-0000-000000000000
uv run potpie pot unset
```

### `pot list`

Prints `CONTEXT_ENGINE_POTS` (from env) and the active pot id (`--json` for raw object).

```bash
uv run potpie pot list
uv run potpie --json pot list
```

### `pot create` / `pot pots` / `pot repo`

Create and list **context pots** (`POST` / `GET /api/v2/context/pots`), then attach repos as needed.

```bash
uv run potpie pot slug-available myscope
uv run potpie pot create myscope
uv run potpie pot use myscope
uv run potpie pot pots
uv run potpie pot repo list
uv run potpie pot repo add owner/repo
```

### `pot hard-reset`

**Destructive:** deletes Graphiti episodic data, structural Neo4j entities for that pot, and (unless `--skip-ledger`) Postgres ledger rows when a DB URL is set. Omit `POT_ID` to infer like `search` / `ingest`.

```bash
uv run potpie pot hard-reset
uv run potpie pot hard-reset --skip-ledger
uv run potpie pot hard-reset --cwd /path/to/git/repo
uv run potpie --json pot hard-reset 00000000-0000-0000-0000-000000000000
```

---

## `add`

Inspect **git `origin`** and print provider-scoped repo identity (helps you wire `CONTEXT_ENGINE_REPO_TO_POT` or `pot use`).

```bash
uv run potpie add
uv run potpie add /path/to/clone
uv run potpie --json add .
```

---

## `search`

Semantic search over **Graphiti episodic** data for a pot.

- **One argument:** `QUERY` — pot is inferred from `--cwd`’s git `origin` + env maps + `pot use`.
- **Two arguments:** `POT_UUID` then `QUERY`.

```bash
# Infer pot (maps + pot use + git under current dir)
uv run potpie search "how is auth handled?"
uv run potpie search "how is auth handled?" -n 10
uv run potpie search "how is auth handled?" --node-labels PullRequest,Decision

# Explicit pot
uv run potpie search 00000000-0000-0000-0000-000000000000 "how is auth handled?"

# Optional filters / output
uv run potpie search "query" --repo owner/repo --source cli
uv run potpie search "query" --as-of 2026-04-01T12:00:00Z
uv run potpie search "query" --include-invalidated
uv run potpie search "query" --with-temporal
uv run potpie search "query" --cwd /path/to/git/repo

uv run potpie --json search "query"
```

Requires `CONTEXT_GRAPH_ENABLED` and working Graphiti/Neo4j configuration.

---

## `ingest`

Adds a **raw episode** through the same pipeline as HTTP ingest: with Postgres, default is **async** (event row + Ingestion Agent planning + queued apply); **`--sync`** runs agent planning and apply inline, or uses a legacy direct Graphiti write when no DB is configured.

**Pot scope:** omit positional pot id to infer from git + env + `pot use` (via `--cwd`).  
**Body:** positional text, or `--episode-body` / `-b`, or `--file` / `-f` (UTF-8). Quick form: `potpie ingest "Your episode text"`.

```bash
# Short form: body as single positional argument (pot inferred)
uv run potpie ingest "We chose Postgres for the ledger." --source cli

# Named options
uv run potpie ingest \
  --name "Design note" \
  --episode-body "We chose Postgres for the ledger." \
  --source cli \
  --reference-time "2026-04-06T12:00:00Z"

# Explicit pot UUID (with body flag or second positional)
uv run potpie ingest 00000000-0000-0000-0000-000000000000 \
  --name "Design note" \
  --episode-body "Note body." \
  --source cli

# Body from file
uv run potpie ingest --file ./notes/design.md --name "Design note" --source cli

# Inline apply (or required when no Postgres for durable events)
uv run potpie ingest "Note" --source cli --sync

# Dedupe when using the event store
uv run potpie ingest "Note" --source cli --idempotency-key my-key-1

uv run potpie --json ingest "Note" --source cli
```

---

## MCP server (optional)

Stdio MCP for external agents (separate from the Potpie HTTP app):

```bash
cd app/src/context-engine
uv sync --all-extras
uv run potpie-mcp
```

Pot access control: `CONTEXT_ENGINE_MCP_ALLOWED_POTS` or (dev only) `CONTEXT_ENGINE_MCP_TRUST_ALL_POTS=true`. See [`README.md`](README.md).

Default context gathering should use `context_resolve` with a recipe:

```json
{"intent":"feature","include":["purpose","feature_map","service_map","docs","tickets","decisions","recent_changes","owners","preferences","source_status"],"mode":"fast","source_policy":"references_only"}
```

Use `context_status` for cheap readiness and recipe hints, `context_search` only for narrow follow-up lookup, and `context_record` for durable project learnings discovered during the work.

---

## More detail

- CLI flags and tables: [`adapters/inbound/cli/README.md`](adapters/inbound/cli/README.md)
- HTTP routes, standalone server, env vars: [`README.md`](README.md)
- Tests: `uv run pytest` from this directory, or from repo root: `uv run pytest app/src/context-engine/tests/unit/`
