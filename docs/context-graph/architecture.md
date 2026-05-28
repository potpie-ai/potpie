# Context Graph Architecture

Last reviewed: 2026-05-28.

The Context Graph has two deployment shapes:

1. **Local open-source self-serve.** `pip install potpie`, run the CLI/MCP
   against a local daemon, and keep graph state on the user's machine.
2. **Managed Potpie cloud.** Host the same graph model and core engine behind
   Potpie auth, shared pots, managed workers, and cloud integrations.

The local shape is the default product direction for open source. Managed
Potpie remains a supported deployment, not a separate architecture.

## Boundary Rule

Keep one graph model and one agent contract. Only deployment adapters differ.

| Boundary | Local OSS | Managed cloud |
|---|---|---|
| User entry | CLI and MCP default to local | Hosted API, cloud MCP/profile |
| Process | User daemon | Hosted API + workers |
| Auth | Local token or IPC permissions | Potpie auth and policy |
| State | Local config/state DB | Hosted DBs |
| Graph store | Local adapter, SQLite first unless changed | Hosted graph/DB adapter |
| Ingestion | Agent-mediated structured writes and scanners | Webhooks, workers, reconciliation |
| Sync | Explicit export/import or cloud push | Native hosted pots |

Do not put cloud users, billing, shared auth, hosted webhooks, or remote source
credentials into the local daemon. Do not fork the graph schema or agent tools
for cloud.

## Graph Model

The portable graph model is the contract both deployments implement:

- **Pot:** isolation boundary for every entity, claim, record, and query.
- **Entity:** stable project object identified by `(pot_id, entity_key)`.
- **Claim:** typed fact about an entity or relationship, with predicate,
  source refs, observed time, valid time, and invalidation fields.
- **Source ref:** compact pointer back to evidence such as a file path, PR,
  issue, doc URL, alert, deploy, or scanner output.
- **Record:** agent-facing durable write that can be lowered into claims.

The graph stores compact claims and evidence pointers, not full source payloads.
Readers, CLI, MCP, and cloud APIs must not depend on a physical database shape.

## Local Architecture

```text
agent harness
  | uses skills / MCP / CLI
  v
potpie CLI and MCP client
  | local profile by default
  v
local daemon
  |-- local auth and daemon lifecycle
  |-- pot/repo/source registry
  |-- context graph API
  |-- scanner and structured-record services
  |-- export/import and optional cloud sync client
  v
core context engine
  |-- domain model and ontology
  |-- read orchestrator and readers
  |-- graph writer and claim query ports
  |-- validation, ranking, envelope builder
  v
local state and graph store
```

The daemon is deliberately boring. It owns persistence, health, migrations,
local auth, and the stable API surface. The agent harness owns source access
and reasoning. Local graph use must not require `POTPIE_API_KEY` or a Potpie
cloud login.

## Managed Architecture

```text
hosted clients / cloud MCP
  v
Potpie API and auth
  v
managed context-graph adapter
  |-- user/project/pot ownership
  |-- hosted source registry
  |-- workers and queues
  |-- webhook/event ingestion
  |-- observability and policy
  v
same core context engine
  v
hosted graph and operational stores
```

Managed Potpie can keep queue-based raw-event reconciliation, cloud source
credentials, multi-user authorization, and hosted observability. Those are
adapter concerns around the same graph ports.

## Core Code To Reuse

| Area | Current path | Role in the new shape |
|---|---|---|
| Domain model | `app/src/context-engine/domain/` | Keep ontology, graph query models, mutation types, context records, ranking, and agent envelope as deployment-neutral core. |
| Graph ports | `domain/ports/context_graph.py`, `domain/ports/claim_query.py` | Keep as the main graph boundary. Local and managed stores implement these contracts. |
| Graph service | `adapters/outbound/graph/context_graph_service.py` | Keep as the read/write facade over graph writer, claim query, read orchestrator, and answer synthesis. |
| Readers | `application/readers/`, `application/services/read_orchestrator.py` | Keep as the only read trunk for agent-visible evidence families. |
| Structured records | `domain/context_records.py`, `domain/ontology.py` | Make this the primary local write path for durable agent memory. |
| Scanners | `domain/ports/config_scanner.py`, `application/use_cases/scan_working_tree.py`, `adapters/outbound/scanners/` | Reuse for local repo scans. Add more scanners here, not in the CLI command body. |
| CLI | `adapters/inbound/cli/` | Reuse Typer app, output helpers, pot resolution, and agent bundle installer. Split local commands from explicit `cloud` commands. |
| MCP | `adapters/inbound/mcp/server.py` | Default to the local daemon. Keep managed mode as an explicit profile. |
| HTTP/FastAPI | `adapters/inbound/http/`, `bootstrap/standalone_container.py` | Reuse route patterns and hardening for daemon API where appropriate. |
| Managed adapter | `app/modules/context_graph/` | Keep cloud-specific API, DB, user, and worker concerns here. |

The main missing local pieces are `bootstrap/local_container.py`, daemon
lifecycle commands, a local state store, local migrations, a local graph store
adapter, local auth/IPC, and export/cloud-sync clients.

## Graph Read Path

All reads should pass through the same trunk:

```text
context_resolve / context_search
  -> ContextGraphQuery
  -> ContextGraphService
  -> ReadOrchestrator
  -> reader-backed include families
  -> ClaimQueryPort
  -> AgentEnvelope
```

Current reader-backed include families are:

- `coding_preferences`
- `infra_topology`
- `timeline`
- `prior_bugs`
- `raw_graph`

Advertised-but-not-yet-backed include families should be returned as
`unsupported_include` with `reason=not_implemented`, not as empty success.

## Graph Write Path

Local OSS should prefer deterministic writes:

1. `context_record` writes structured durable records.
2. Record emitters lower records to canonical claims.
3. Repo scanners emit facts from files and manifests.
4. Optional harness skills may submit validated mutation plans.
5. Raw event reconciliation is optional locally and normal in managed cloud.

Managed Potpie can continue to use raw-event ledgers, workers, and LLM-backed
reconciliation where those features are useful. Local mode should not need a
daemon-side LLM key for the default loop.

## Local Storage

For OSS V1, optimize for installation and portability:

- The graph model is canonical; the physical store is an adapter.
- SQLite is the preferred local default unless benchmarks prove it cannot carry
  the required query patterns.
- Neo4j can remain an optional local profile for development or parity testing.
- Managed cloud can keep hosted graph infrastructure.

This is a deliberate shift from "one physical graph store" to "one graph model
and one graph API."

## Daemon Lifecycle

The CLI should manage the daemon:

```bash
potpie init
potpie daemon install
potpie daemon start
potpie daemon status
potpie doctor
```

Target service managers:

- macOS: `launchd` user agent
- Linux: `systemd --user`, with foreground fallback
- Windows: Task Scheduler or Windows service wrapper

Every local command should check daemon health. If the daemon is missing, the
CLI can start it. If unhealthy, it should show the log path and a restart path.

## Cloud And Event Ledger

Cloud integration is explicit:

```bash
potpie cloud login
potpie cloud push
potpie cloud pull
potpie cloud status
```

Webhook ingestion should become a separate event-ledger service. It receives
cloud webhooks, stores normalized events, and exposes cursor-based reads. A
local user or agent harness can pull from that ledger and decide what to record
into the local graph. Local daemons should not need to be internet-facing.

## Implementation Order

1. **Docs and package boundary.** Make local-first the default in docs and CLI
   language; keep managed as explicit cloud profile.
2. **Local daemon skeleton.** Add foreground daemon, health/status, local token,
   local pot registry, and `potpie daemon` commands.
3. **Local graph store.** Implement local `GraphWriterPort` and
   `ClaimQueryPort`, likely SQLite first.
4. **Structured writes.** Route `context_record` to deterministic local claim
   emitters.
5. **Scanners and skills.** Seed useful context from repo files and harness
   skills without source credentials in the daemon.
6. **Service installers.** Add launchd/systemd/Windows lifecycle support.
7. **Export/import/cloud push.** Move a local pot to managed Potpie with clear
   provenance and no surprise uploads.
8. **Event ledger.** Separate webhook capture from graph ingestion.

## Non-Negotiables

- Local graph use works without Potpie cloud auth.
- CLI/MCP local mode and cloud mode are visibly different.
- The agent-facing four tools stay stable.
- Graph facts store compact claims and source references, not full source
  payloads.
- Managed-only concerns stay outside the local daemon.
- Benchmarks must run against the same graph contract for local and managed
  adapters.
