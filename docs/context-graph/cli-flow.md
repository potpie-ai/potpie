# Potpie CLI Flow And Command Contract

Last reviewed: 2026-06-08.

This is the target product contract for the `potpie` CLI. Graph V1 keeps the
existing legacy wrappers while moving them onto V2-aligned graph internals.
Graph V2 later exposes the explicit `potpie graph ...` workbench over
the same internals. The same command language should work across local OSS and
managed backends. The active pot decides where the CLI routes the request: local
pots route to the local daemon, and managed pots route to the authenticated
managed backend.

```mermaid
flowchart TB
  cg_user["user or agent"]
  cg_cli["potpie CLI"]
  cg_local_profile["local profile<br/>daemon hosts same services<br/>local stores"]
  cg_managed_profile["managed backend<br/>API hosts same services<br/>hosted stores"]
  cg_ledger["Event Ledger<br/>managed or self-hosted"]

  cg_user --> cg_cli
  cg_cli --> cg_local_profile
  cg_cli -. "login / managed pot / cloud sync" .-> cg_managed_profile
  cg_cli -. "explicit ledger config/pull" .-> cg_ledger
  cg_local_profile -. "pull events" .-> cg_ledger
  cg_managed_profile -. "consume events" .-> cg_ledger
```

The CLI is the user/agent command surface for setup, login, pots, source
registration, queued connector/source-event ingestion, graph reads/writes,
graph/backend admin, skills, and explicit cloud sync. For setup, the CLI owns
flags, validation, output, and daemon bootstrap; the daemon-hosted
`SetupOrchestrator` owns local dependency setup. Commands route by the active or
selected pot. Selecting a managed pot after `potpie login` is the explicit remote
boundary. Explicit ledger commands may call a managed or self-hosted Event
Ledger without changing where graph state is stored.

## Journey

```mermaid
flowchart LR
  cg_setup["setup or login"]
  cg_status["graph status"]
  cg_read["catalog/read"]
  cg_work["work"]
  cg_write["propose/commit"]
  cg_ingest["connector ingest/sync"]

  cg_setup --> cg_status --> cg_read --> cg_work --> cg_write
  cg_ingest --> cg_status
  cg_write --> cg_read
```

Local first run:

```bash
pip install potpie
potpie setup --repo . --agent claude
potpie status
```

Managed backend use is explicit:

```bash
potpie login
potpie pot list --managed
potpie use <managed-pot-name> --managed
potpie status
```

Managed ledger use from a local graph is also explicit:

```bash
potpie login
potpie ledger use managed
potpie ledger pull --source <id>
```

## Profiles

| Profile | Routes to | Storage | Lifecycle behavior |
|---|---|---|---|
| Local | Local daemon hosting Pot Management, Graph Service, and Skill Manager | Local state DB, embedded GraphBackend, local skill cache | `potpie setup` installs/starts daemon; the daemon provisions dependencies, creates active `default` pot, registers repo, and optionally installs skills. |
| Managed | Managed backend API hosting the same services | Hosted operational DB, hosted graph/search, hosted skill/catalog stores | `potpie login` authenticates to `cloud.backend_url`; managed pots become available through the same pot commands. Cloud push/pull/sync remain explicit. |

Commands default to the active pot. Before login, that is normally the local
`default` pot. After login, `potpie use <name>` may select either a local or
managed pot. `--local` and `--managed` filter lists and disambiguate names;
`--pot <id-or-name>` scopes commands without changing the active pot.

Managed backend URL is configuration, not a graph backend profile:

```bash
potpie config get cloud.backend_url
potpie config set cloud.backend_url https://potpie.example.com
potpie login [--backend-url <url>]
```

An Event Ledger binding is separate from the active pot and backend:

| Ledger binding | Routes to | Effect |
|---|---|---|
| Managed | Potpie managed Event Ledger | Pulls GitHub/Linear/etc. events into the selected local or managed graph. |
| Self-hosted | Configured ledger URL | Uses the same pull/replay-token contract against a user-run ledger. |

Using a managed ledger does not imply `cloud push`, `cloud pull`, or managed
graph storage. It only gives the selected graph a source-event feed.

## Local Setup Contract

`potpie setup` is idempotent. On first local run, the CLI installs/starts the
daemon service, then asks the daemon to run setup. Service-manager registration
and detached daemon start are hard only for the normal local daemon profile; they
are skipped for in-process/dev profiles and are not part of `potpie login`. The
daemon-hosted setup flow:

1. creates local config/data directories;
2. initializes local auth;
3. provisions the selected local GraphBackend and related stores;
4. runs migrations;
5. creates a local `default` pot and marks it active;
6. registers the repo source;
7. optionally installs skills for the requested agent harness.

`--pot <name>` only overrides the initial pot name. If an active pot already
exists, setup reuses it unless `--pot` names another pot to create/use.

The CLI command builds a setup plan from flags, ensures the daemon is available
when the selected host mode needs it, and renders the report. The daemon-hosted
application `SetupOrchestrator` owns the ordered lifecycle calls that make the
plan real. `potpie setup --dry-run` returns a `SetupPreview` with planned actions,
owners, hard/soft classification, and skip reasons; it does not execute setup and
does not return executed `StepResult`s.

## Command Groups

All commands support human output by default and `--json` for scripts/agents.

The host-routed CLI lives at `adapters/inbound/cli/host_cli.py` (assembles the
groups) + `adapters/inbound/cli/commands/`. Every command routes
`CLI -> HostShell -> service(s) -> ports`; `commands/_common.py` owns the
`--json`/exit-code/error contract and active-pot resolution. Code slots per
group:

| Group | Code slot | Routes to |
|---|---|---|
| `setup` `status` `doctor` `config` `login` `logout` `whoami` | `commands/bootstrap.py` | `HostShell`; setup bootstraps daemon then routes to `SetupOrchestrator`; login routes to managed auth; top-level `status` is product/host status only |
| `pot` `source` | `commands/pots.py` | `HostShell.pots` (`PotManagementService`) |
| `daemon` | `commands/daemon.py` | `HostShell.daemon` (`Daemon`) |
| `ledger` | `commands/ledger.py` | `HostShell.ledger` (`LedgerFacade`) |
| `graph` | `commands/graph.py` | `HostShell.graph` (`GraphService` / Graph Workbench Port): status, catalog, describe, search-entities, read, propose, commit, history, inbox, admin |
| `backend` | `commands/backend.py` | `HostShell.backend` (`GraphBackend`) |
| `skills` | `commands/skills.py` | `HostShell.skills` (`SkillManager`) |
| `cloud` | `commands/cloud.py` | explicit snapshot sync and managed skill sync |

`adapters/inbound/cli/host_cli.py` is the `potpie` console entrypoint (see
`[project.scripts]`). The MCP server (`adapters/inbound/mcp/server.py`) binds to
the same in-process `HostShell`. The async ingestion pipeline behind the HTTP
API keeps its own composition root (`bootstrap/ingestion_server.py`) until it is
migrated onto `HostShell`.

### Bootstrap And Profile

```bash
potpie setup [--repo .] [--pot <name>] [--agent claude] [--yes] [--dry-run]
potpie login [--backend-url <url>] [--org <id>]
potpie logout
potpie whoami
potpie status [--intent feature] [--harness claude] [--json]
potpie doctor
potpie config get <key>
potpie config set <key> <value>
```

`status` is the cheap aggregate grouped by owner: host liveness, Pot Management
control plane, Graph Service data plane, GraphBackend capabilities/projections,
Event Ledger binding and consumer backlog, Skill Manager drift, login state when
relevant, and next action.

`doctor` is local-profile diagnostics: paths, logs, auth/socket state,
migrations, and skill drift.

### Local Daemon Admin

```bash
potpie daemon status [--json]
potpie daemon logs [--follow]
potpie daemon restart
potpie daemon stop
```

Daemon commands are local recovery tools, not onboarding steps.

### Pots And Sources

```bash
potpie pot list
potpie pot list [--local] [--managed] [--all]
potpie pot info [--json]
potpie pot create <name> [--repo .] [--use]
potpie pot use <name-or-id>
potpie use <name-or-id> [--local | --managed]
potpie pot rename <name-or-id> <new-name>
potpie pot reset [--confirm]
potpie pot archive <name-or-id>

potpie source add repo <path> [--name platform]
potpie source list [--json]
potpie source status [--json]
potpie source remove <source-id>
```

Local setup creates and uses `default`. `pot create` is for additional workspace
boundaries. After login, managed pots appear in the same list/use surface. If a
local pot and managed pot share a name, `potpie use` must be disambiguated with
`--local`, `--managed`, or a qualified id. Source commands route to the active
pot's Pot Management service. `potpie source add repo ...` only records source
metadata for pot resolution and visibility; it does not inspect, scan, or ingest
the repository.

### Sync

```bash
potpie cloud push [--pot <name>]
potpie cloud pull [--pot <name>]
```

Registering a source records metadata. Cloud push/pull moves a pot snapshot
between local and managed backends; it must remain explicit.

### Event Ledger

```bash
potpie ledger status [--json]
potpie ledger use managed [--org <id>]
potpie ledger use self-hosted <url>
potpie ledger sources list [--json]
potpie ledger query [--source <id>] [--type <kind>] [--since <time>] [--until <time>] [--limit <n>] [--json]
potpie ledger pull --source <id> [--filter <expr>] [--json]
potpie ledger disconnect
```

The Event Ledger is a managed or self-hostable source-event service. It owns
source-provider credentials, webhook receivers, normalized event history, and
provider-side ingestion cursors. It also exposes query/filter over the event
history and returns ordered pages with opaque replay tokens.

`ledger query` inspects ledger history without touching graph consumer state.
`ledger pull` fetches a page using the selected graph consumer cursor and any
filters, then advances the local cursor. It does not enqueue events or write
graph claims.

### Graph V1 Legacy Surface

Graph V1 keeps the existing top-level wrappers while their internals move to the
V2-aligned ontology, view, semantic mutation, validation, and inbox model.

```bash
potpie status [--json]
potpie resolve "debug refund failures" --intent debugging [--json]
potpie search "bulk refunds" [--json]
potpie record --type fix --summary "..." --scope service:refunds-api [--json]
```

| CLI command | Internal target |
|---|---|
| `potpie status` | Future `graph status` readiness shape, with V1 legacy output. |
| `potpie resolve` | `intent/include` mapped to named read views. |
| `potpie search` | Narrow entity/claim lookup over claim and semantic indexes. |
| `potpie record` | Semantic mutation validation, low-risk commit, or inbox item. |

These commands are legacy wrappers, not the long-term product contract.
They must not bypass semantic validation, evidence requirements, provenance, or
inbox handling, and they must not accept obsolete view names or non-canonical key
prefixes.

### Graph V2 Workbench

```bash
potpie graph status [--json]
potpie graph catalog --task "debug refund failures" [--json]
potpie graph describe debugging --view prior_occurrences [--examples] [--json]
potpie graph search-entities --query "bulk refunds" --subgraph features [--json]
potpie graph read --subgraph debugging --view prior_occurrences --scope service:refunds-api [--json]
potpie timeline recent [--time-window 7d] [--service refunds-api] [--json]
potpie graph propose --file mutation.json [--json]
potpie graph commit mutation-plan:01JY8T5C [--json]
potpie graph history --entity service:payments-api [--json]
potpie graph inbox add --summary "..." --evidence github:pr:acme/payments:955 [--json]
```

| CLI command | Service path |
|---|---|
| `potpie graph status` | Pot Management + Graph Service + GraphBackend + Ledger + Skill Manager readiness |
| `potpie graph catalog` / `describe` | Ontology Catalog contracts and examples |
| `potpie graph search-entities` | Identity Resolver over entity index, alternate names, external IDs, and source refs |
| `potpie graph read` / `history` | Read View Router over claim query, semantic search, traversal projections, and audit |
| `potpie timeline recent` | Project-wide event view over the active/current pot, deduped by source ref and sorted by occurrence time |
| `potpie graph propose` | Plan Validator + Plan Store; no graph write |
| `potpie graph commit` | Commit Engine applies a stored plan by `plan_id` and writes audit/history |
| `potpie graph inbox` | Pending graph work capture and processing queue |

These commands are shared across local and managed pots. The active or selected
pot decides whether they route to the local daemon or managed backend API.
Top-level `resolve`, `search`, `record`, and `context_*` tools remain Graph V1
legacy wrappers until the workbench is ready.

### Graph Admin And Backend

```bash
potpie graph export <file>
potpie graph import <file> [--pot <name>]
potpie graph repair [--semantic-index] [--all]
potpie graph reset [--confirm]
potpie graph admin ...

potpie backend list
potpie backend status [--json]
potpie backend use embedded
potpie backend doctor
```

Graph/backend commands call services and capability ports. CLI code must not
query SQLite, Neo4j, vector indexes, hosted stores, or state tables directly.

### Skills

```bash
potpie skills list
potpie skills install [<id>] --agent claude [--scope global|project] [--path .]
potpie skills update [--all] [--agent claude] [--scope global|project] [--path .]
potpie skills remove [<id>|--all] --agent claude [--scope global|project] [--path .]
potpie skills status --agent claude [--scope global|project] [--path .] [--json]
potpie skills add <path-or-url>
potpie cloud skills sync [--agent <id>]
```

Skills are CLI-managed recipes. Agents only see an advisory `skills` block in
`potpie graph status` with missing/outdated skills and an exact install command.
Cloud skill sync is explicit.

Skill install defaults to `--scope global`, writing to the harness's user-level
skills directory:

| Harness | Global path |
| --- | --- |
| Cursor | `~/.cursor/skills/<skill>/SKILL.md` |
| Claude Code | `~/.claude/skills/<skill>/SKILL.md` |
| OpenCode | `~/.config/opencode/skills/<skill>/SKILL.md` |
| Codex | `$HOME/.agents/skills/<skill>/SKILL.md` |

Use `--scope project --path .` when a repo-local install should be committed or
shared with the repository.

## Output Contract

- Human output: action-oriented summary and next command.
- `--json`: stable fields for agents/scripts; additive changes are OK.
- `setup --dry-run`: returns a preview document with planned steps; no mutation,
  daemon dependency setup, source registration, or skill install occurs.
- Mutations should be idempotent when possible.
- Destructive commands require `--confirm` or interactive confirmation.
- Exit codes:
  - `0`: success
  - `1`: command or validation failure
  - `2`: daemon/API/dependency unavailable
  - `3`: partial/degraded result
  - `4`: auth/permission failure
- JSON errors include `code`, `message`, `detail`, and
  `recommended_next_action`.

## First Use Cases

- New repo onboarding: setup, register source, run harness-led ingestion for
  explicit docs/history, ask how services fit together.
- Feature work: read feature context, topology, owners, and decisions.
- Debugging: read prior bugs, recent timeline, dependencies, and runbooks.
- Review prep: read recent decisions and project conventions for a PR.
- Incident memory: propose root cause, fix, verification, and follow-up facts.
- Managed work: log in to a managed backend, list managed pots, `potpie use` a
  managed pot, then run the same `potpie graph ...` commands.
- Managed migration: push a local pot to cloud, or pull a hosted pot for local
  work.
- Integration-backed local graph: log in to managed Potpie, bind the managed
  ledger, pull GitHub/Linear source events, and apply queued event ingestion into
  the local graph.
- Offline work: read and commit validated graph facts against the embedded backend
  without cloud auth.

## Build Order

1. Local setup + daemon lifecycle + health/logs.
2. Local Pot Management with active `default` pot and source registry.
3. Embedded GraphBackend and conformance suite.
4. V1 wrappers over V2-aligned ontology, named views, semantic mutations,
   validation, and inbox handling.
5. Canonical `potpie graph status/catalog/describe/search-entities/read` through
   daemon services.
6. `potpie graph propose/commit/history/inbox` through daemon services.
7. Event Ledger run history.
8. `backend` and `skills` commands.
9. `potpie login` against configurable `cloud.backend_url`, unified
   local/managed pot listing and `potpie use`, and explicit cloud push/pull/skills
   sync.
10. Event Ledger binding, consumer cursor storage, status, and pull/apply
   commands that route queued source events through event ingestion.
11. Managed profile routing for shared command groups.
