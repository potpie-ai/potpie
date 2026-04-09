# context-engine — command examples

Run these from the **package root** so `uv` resolves the editable package and extras:

```bash
cd app/src/context-engine
uv sync --all-extras
```

Prefer **`uv run context-engine …`** (or activate the venv and run `context-engine` directly). Graphiti/Neo4j-related commands need the same env as production (`CONTEXT_GRAPH_ENABLED`, Neo4j, etc.); see [`README.md`](README.md).

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
uv run context-engine --version
uv run context-engine --json doctor
uv run context-engine -v search "example query"
```

Built-in help:

```bash
uv run context-engine --help
uv run context-engine search --help
uv run context-engine pot --help
```

---

## `doctor`

Shows whether common **environment variables look set** (feature flag, Neo4j, Potpie API token source, GitHub token). It does **not** test connectivity to Neo4j or Graphiti.

```bash
uv run context-engine doctor
uv run context-engine --json doctor
```

---

## `login` / `logout`

Optional: store a Potpie API key (and optional base URL) for future HTTP use. CLI **pot scope** for `search` / `ingest` still uses **`pot use`**, env maps, or git `origin` — not the projects API.

```bash
uv run context-engine login YOUR_API_KEY
uv run context-engine login YOUR_API_KEY --url http://127.0.0.1:8001
uv run context-engine logout
```

`POTPIE_API_KEY` in the environment overrides the stored token.

---

## `init-agent`

Installs `AGENTS.md` and repo-local `.agents/skills/context-engine-*` into a repository (safe by default; skips differing files unless `--force`).

```bash
uv run context-engine init-agent
uv run context-engine init-agent /path/to/repo
uv run context-engine init-agent --force
uv run context-engine --json init-agent .
```

---

## `pot`

### `pot use` / `pot unset`

Remember or clear the **default pot UUID** for this machine (stored with credentials). **`pot use` takes precedence** over `CONTEXT_ENGINE_REPO_TO_POT` / `CONTEXT_ENGINE_POTS` when inferring scope.

```bash
uv run context-engine pot use 00000000-0000-0000-0000-000000000000
uv run context-engine pot unset
```

### `pot list`

Prints `CONTEXT_ENGINE_POTS` (from env) and the active pot id (`--json` for raw object).

```bash
uv run context-engine pot list
uv run context-engine --json pot list
```

### `pot create` / `pot pots` / `pot repo`

Create and list **context pots** (`POST` / `GET /api/v2/context/pots`), then attach repos as needed.

```bash
uv run context-engine pot create "myscope"
uv run context-engine pot use myscope
uv run context-engine pot pots
uv run context-engine pot repo list
uv run context-engine pot repo add owner/repo
```

### `pot hard-reset`

**Destructive:** deletes Graphiti episodic data, structural Neo4j entities for that pot, and (unless `--skip-ledger`) Postgres ledger rows when a DB URL is set. Omit `POT_ID` to infer like `search` / `ingest`.

```bash
uv run context-engine pot hard-reset
uv run context-engine pot hard-reset --skip-ledger
uv run context-engine pot hard-reset --cwd /path/to/git/repo
uv run context-engine --json pot hard-reset 00000000-0000-0000-0000-000000000000
```

---

## `add`

Inspect **git `origin`** and print provider-scoped repo identity (helps you wire `CONTEXT_ENGINE_REPO_TO_POT` or `pot use`).

```bash
uv run context-engine add
uv run context-engine add /path/to/clone
uv run context-engine --json add .
```

---

## `search`

Semantic search over **Graphiti episodic** data for a pot.

- **One argument:** `QUERY` — pot is inferred from `--cwd`’s git `origin` + env maps + `pot use`.
- **Two arguments:** `POT_UUID` then `QUERY`.

```bash
# Infer pot (maps + pot use + git under current dir)
uv run context-engine search "how is auth handled?"
uv run context-engine search "how is auth handled?" -n 10
uv run context-engine search "how is auth handled?" --node-labels PullRequest,Decision

# Explicit pot
uv run context-engine search 00000000-0000-0000-0000-000000000000 "how is auth handled?"

# Optional filters / output
uv run context-engine search "query" --repo owner/repo --source cli
uv run context-engine search "query" --as-of 2026-04-01T12:00:00Z
uv run context-engine search "query" --include-invalidated
uv run context-engine search "query" --with-temporal
uv run context-engine search "query" --cwd /path/to/git/repo

uv run context-engine --json search "query"
```

Requires `CONTEXT_GRAPH_ENABLED` and working Graphiti/Neo4j configuration.

---

## `ingest`

Adds a **raw episode** through the same pipeline as HTTP ingest: with Postgres, default is **async** (event row + queue); **`--sync`** applies inline or uses a legacy direct Graphiti write when no DB is configured.

**Pot scope:** omit positional pot id to infer from git + env + `pot use` (via `--cwd`).  
**Body:** positional text, or `--episode-body` / `-b`, or `--file` / `-f` (UTF-8). Quick form: `context-engine ingest "Your episode text"`.

```bash
# Short form: body as single positional argument (pot inferred)
uv run context-engine ingest "We chose Postgres for the ledger." --source cli

# Named options
uv run context-engine ingest \
  --name "Design note" \
  --episode-body "We chose Postgres for the ledger." \
  --source cli \
  --reference-time "2026-04-06T12:00:00Z"

# Explicit pot UUID (with body flag or second positional)
uv run context-engine ingest 00000000-0000-0000-0000-000000000000 \
  --name "Design note" \
  --episode-body "Note body." \
  --source cli

# Body from file
uv run context-engine ingest --file ./notes/design.md --name "Design note" --source cli

# Inline apply (or required when no Postgres for durable events)
uv run context-engine ingest "Note" --source cli --sync

# Dedupe when using the event store
uv run context-engine ingest "Note" --source cli --idempotency-key my-key-1

uv run context-engine --json ingest "Note" --source cli
```

---

## MCP server (optional)

Stdio MCP for external agents (separate from the Potpie HTTP app):

```bash
cd app/src/context-engine
uv sync --all-extras
uv run context-engine-mcp
```

Pot access control: `CONTEXT_ENGINE_MCP_ALLOWED_POTS` or (dev only) `CONTEXT_ENGINE_MCP_TRUST_ALL_POTS=true`. See [`README.md`](README.md).

---

## More detail

- CLI flags and tables: [`adapters/inbound/cli/README.md`](adapters/inbound/cli/README.md)
- HTTP routes, standalone server, env vars: [`README.md`](README.md)
- Tests: `uv run pytest` from this directory, or from repo root: `uv run pytest app/src/context-engine/tests/unit/`
