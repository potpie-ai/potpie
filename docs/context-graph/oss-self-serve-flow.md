# OSS Self-Serve Flow And CLI Contract

Last reviewed: 2026-05-28.

This doc is the target product contract for local open-source Potpie. Treat it as
the user and agent journey we are implementing toward, not as a description of
the current command surface.

The Potpie CLI is the primary OSS surface. Users and agents manage local pots,
sources, ingestion, graph reads/writes, graph backends, daemon lifecycle, and
agent skills through the CLI. The daemon is the local service host behind that
surface.

```text
user / agent harness
  -> potpie CLI
  -> local daemon shell
      -> Pot Management Service
      -> Graph Service
          -> GraphBackend
      -> Skill Manager Service
  -> local state DB + embedded graph backend + skill cache
```

Cloud behavior is explicit under `potpie cloud ...`. A local command must not
silently call the managed API.

## Product Shape

The OSS default should feel like:

```bash
pip install potpie
potpie setup --repo . --agent claude --scan
potpie status
potpie resolve "How should I add a Postgres-backed endpoint?"
potpie record --type decision --summary "Use repository services for inventory writes"
```

For an agent harness:

```bash
potpie skills install --agent claude
potpie skills status --agent claude
```

Installed skills teach the harness the working loop:

```text
status -> resolve/search -> work -> record durable memory -> scan when needed
```

Skills are instructions and recipes. They never create a fifth agent tool, and
skill content is not graph data.

## Boundary Contract

| Surface | Owns | Must not |
|---|---|---|
| CLI | UX, command parsing, output formatting, daemon discovery, local/cloud profile selection, skill setup orchestration, user and agent workflow surface | query physical graph stores directly; hide cloud calls behind local commands |
| Daemon shell | process lifecycle, local auth, socket/HTTP transport, config, logs, health, migrations trigger | contain pot, graph, source, or skill business logic |
| Pot Management Service | pot CRUD, active pot, local state DB, source registry, scanner state, lifecycle, export/import metadata, graph status aggregation | run physical graph queries; hold remote source credentials by default |
| Graph Service | resolve/search/record/status data plane, read orchestration, ranking, semantic routing, graph writes | know store topology; orchestrate multi-store writes |
| GraphBackend | mutation, claim query, semantic search, inspection, analytics, snapshot capabilities | leak SQLite/Neo4j/vector details upward |
| Skill Manager Service | skill catalog, skill install/update/remove, target adapters, drift detection, advisory status nudge | execute skills; run agent reasoning; become an agent tool |

## Command Groups

All commands should support a human-readable default and `--json` for stable
automation. `--pot <id-or-name>` should be accepted anywhere pot scope matters;
omitting it uses the active pot.

### Bootstrap And Health

```bash
potpie setup [--repo .] [--pot <name>] [--agent claude] [--scan] [--yes]
potpie init
potpie status [--intent feature] [--harness claude] [--json]
potpie doctor
potpie config get <key>
potpie config set <key> <value>
```

`setup` is the normal first-run command. It is idempotent and owns the full local
bootstrap: create local config/data directories, initialize local auth, install
and start the daemon, run migrations, create the initial pot, mark it active,
register a repo source, optionally run the first scan, and optionally install
agent skills for the requested harness.

If no pot exists, `setup` creates a local pot named `default` and uses it as the
active pot. `--pot <name>` only overrides the initial pot name; it is not required
for the normal path. If an active pot already exists, setup reuses it unless
`--pot` names a different pot to create/use.

`init` is the lower-level setup step for advanced/scripted flows. It prepares
local config and state, but the happy path should not require users to run daemon
lifecycle commands by hand. If daemon installation/startup is needed, `setup` or
the next local command should handle it automatically.

`status` is the user-facing aggregate: daemon health, active pot, sources,
ingestion freshness, backend capability/readiness, semantic embedder/index
readiness, installed skills, and recommended next actions.

`doctor` is diagnostic and more verbose: local paths, daemon logs, auth/socket
state, migration state, backend readiness, scanner registry, and skill drift.

### Daemon Administration

```bash
potpie daemon install
potpie daemon start
potpie daemon stop
potpie daemon restart
potpie daemon status [--json]
potpie daemon logs [--follow]
```

These commands are for troubleshooting, service-manager integration, and
recovery. They are not part of the normal first-run path. Because most Potpie
features require the daemon, `potpie setup` installs/starts it, and every local
command should check daemon health. If the daemon is missing or stopped, the CLI
should install/start it when safe; if unhealthy, the CLI should show the log path
and the next recovery command.

### Pots And Sources

```bash
potpie pot create <name> [--repo .] [--use]
potpie pot list
potpie pot use <name-or-id>
potpie pot info [--json]
potpie pot rename <name-or-id> <new-name>
potpie pot reset [--confirm]
potpie pot archive <name-or-id>

potpie source add repo <path> [--name platform]
potpie source list [--json]
potpie source remove <source-id>
potpie source status [--json]
```

A pot is the local isolation boundary. Source registration is metadata in the
local state DB. Source registration does not mean the graph contains facts; an
ingestion command must write claims.

After initial setup, `default` is the active pot unless the user passed
`--pot <name>`. `pot create` is for additional workspaces; commands that omit
`--pot` operate on the active pot.

### Ingestion

```bash
potpie ingest scan [--source <id>] [--changed] [--watch]
potpie ingest status [--json]
potpie ingest runs
potpie ingest show <run-id> [--json]
potpie ingest replay <run-id>
```

Local ingestion defaults to deterministic paths:

1. Scanner adapters read local files and manifests.
2. Scanners emit validated graph mutations or structured records.
3. Graph Service applies mutations through `GraphMutationPort`.
4. The active backend writes canonical claims and updates/rebuilds projections.

Good first scanners: CODEOWNERS, dependency manifests, Kubernetes/Helm,
OpenAPI, CI workflow files, service manifests, ADR indexes, and runbook indexes.

Raw-event reconciliation is optional locally. It should be a separate opt-in
mode, not required for the default scan/record loop.

### Query And Memory

```bash
potpie resolve "<task>" [--intent feature] [--include infra_topology,prior_bugs] [--json]
potpie search "<lookup>" [--include prior_bugs] [--json]
potpie record --type fix --summary "..." [--details details.json] [--source-ref ...]
potpie record --type preference --summary "..." [--scope service:inventory-svc]
```

These commands are CLI adapters over the four-tool agent contract:

| CLI command | Agent tool / service path |
|---|---|
| `potpie resolve` | `context_resolve` -> Graph Service -> ReadOrchestrator -> `AgentEnvelope` |
| `potpie search` | `context_search` -> same envelope path |
| `potpie record` | `context_record` -> deterministic record emitter -> graph mutation |
| `potpie status` | `context_status` + Pot Management status + Skill Manager nudge |

The CLI may add formatting, summaries, tables, and transport metadata, but the
machine contract for resolve/search remains the `AgentEnvelope` in
[`agent-contract.md`](./agent-contract.md).

### Graph And Backend Operations

```bash
potpie graph status [--json]
potpie graph inspect <entity-key> [--depth 2] [--json]
potpie graph export <file>
potpie graph import <file> [--pot <name>]
potpie graph repair [--semantic-index] [--all]

potpie backend list
potpie backend status [--json]
potpie backend use embedded
potpie backend doctor
```

`graph` commands call Graph Service and GraphBackend capability ports. They do
not query SQLite, Neo4j, or any vector store directly from CLI code.

`backend use` changes the storage profile binding. For OSS V1, `embedded` is the
default profile and must work without Docker, Neo4j, Postgres, Potpie cloud auth,
or an external embedding API.

### Skills And Agent Setup

```bash
potpie skills list
potpie skills install [<id>] --agent claude [--path .]
potpie skills update [--all] [--agent claude]
potpie skills remove <id> --agent claude
potpie skills status --agent claude [--json]
potpie skills add <path-or-url>
```

Skills are managed by CLI/admin commands. Agents see only an advisory `skills`
block in `context_status`, including missing/outdated skills and an exact
install command. The agent should relay that command to the user; it should not
install skills as an agent tool side effect.

### Cloud Commands

```bash
potpie cloud login
potpie cloud status
potpie cloud push
potpie cloud pull
potpie cloud skills sync
```

Cloud commands are explicitly cloud-scoped. Cloud push/pull uses portable pot
snapshots and sync metadata; it must not fork the graph model.

## Local Daemon API Contract

The CLI talks to the daemon through a stable local transport: Unix socket or
named pipe preferred, loopback HTTP acceptable. The transport is not the business
contract; service DTOs are.

Minimum daemon surfaces:

| Surface | Routes to | Purpose |
|---|---|---|
| health/readiness | daemon shell + hosted services | liveness, readiness, version, local paths |
| pot management | Pot Management Service | pot CRUD, active pot, source registry, scan state |
| context tools | Graph Service | resolve/search/record/status |
| graph admin | Graph Service / GraphBackend | inspect, analytics, repair, snapshot import/export |
| ingestion | scanner use cases + Graph Service | scan runs, replay, watch |
| skills | Skill Manager Service | catalog, install/update/remove/status |

Local auth should be OS-user scoped, socket-permission based, or use a local
token. It should not require `POTPIE_API_KEY`.

## Output And Error Contract

- Human output is optimized for action: what happened, what is stale/missing, and
  the next command to run.
- `--json` output is stable enough for agents and scripts; fields may be added
  but should not be renamed casually.
- Commands that mutate state should be idempotent where possible.
- Destructive commands require `--confirm` or an interactive confirmation.
- Exit code `0` means success; `1` means command/validation failure; `2` means
  daemon or dependency unavailable; `3` means partial/degraded result; `4` means
  auth/permission failure.
- Error payloads should include `code`, `message`, `detail`, and
  `recommended_next_action` when `--json` is used.

## First Valuable Use Cases

Build toward these flows first:

- **New repo onboarding:** run setup, which creates/uses the active pot,
  registers the repo, scans, then ask how services fit together.
- **Feature work:** resolve preferences, infra topology, owners, and decisions
  before editing.
- **Debugging:** resolve prior bugs, recent timeline, infra dependencies, and
  runbooks for a symptom.
- **Review prep:** search recent decisions and project preferences for a PR
  scope.
- **Incident memory:** record root cause, fix, verification, and follow-up so
  future agents can retrieve it.
- **Team conventions:** record local preferences/policies without waiting for a
  cloud integration.
- **Offline work:** query and record against the embedded backend with no cloud
  login.

## Implementation Sequence

Use this sequence to keep the CLI contract grounded:

1. Define command groups and `--json` schemas for `setup`, `status`, `pot`,
   `daemon`, `resolve`, `search`, `record`, `ingest`, `graph`, `backend`,
   and `skills`.
2. Implement setup orchestration, daemon discovery, lifecycle commands, local
   auth, health, and logs. `setup` must install/start the daemon so the happy
   path does not expose daemon mechanics.
3. Implement local Pot Management: state DB, default active pot creation, source
   registry, pot CRUD, and source status.
4. Implement `GraphBackend` ports and the embedded backend conformance suite.
5. Route `resolve/search/status` through the local daemon and Graph Service.
6. Implement scanner ingestion and ingestion run history.
7. Implement deterministic `context_record` emitters for the first high-value
   record types: `preference`, `fix`, `bug_pattern`, `decision`, `verification`.
8. Implement graph inspect/status/repair/export/import through capability ports.
9. Implement Skill Manager ports and `potpie skills ...`.
10. Add cloud push/pull and cloud skill sync as explicit commands.

Do not let implementation shortcuts punch through the boundaries above. If a CLI
command needs graph data, add or use a service/capability port.
