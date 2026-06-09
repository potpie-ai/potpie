# Context Graph Architecture

Last reviewed: 2026-05-29.

This document is the implementation map for the Context Graph. It follows the
same shape as the product: CLI first, the same service modules hosted either by
a local daemon or managed backend API, swappable storage backends, explicit
managed-backend login, and explicit cloud sync or ledger commands.

## Anatomy

```mermaid
flowchart TB
  cg_actor["user or agent"]
  cg_cli["potpie CLI"]
  cg_host["host shell<br/>local daemon or managed backend API"]
  cg_pot_service["Pot Management Service"]
  cg_graph_service["Graph Service"]
  cg_backend["GraphBackend"]
  cg_skill_service["Skill Manager Service"]
  cg_state_db[("state DB<br/>local or hosted")]
  cg_graph_store[("graph/search store<br/>local or hosted")]
  cg_skill_cache[("skill catalog/cache<br/>local or hosted")]

  cg_actor --> cg_cli
  cg_cli --> cg_host
  cg_host --> cg_pot_service
  cg_host --> cg_graph_service
  cg_host --> cg_skill_service
  cg_pot_service --> cg_state_db
  cg_pot_service --> cg_graph_service
  cg_graph_service --> cg_backend
  cg_backend --> cg_graph_store
  cg_skill_service --> cg_skill_cache
```

| Piece | Responsibility |
|---|---|
| CLI | User/agent surface, command UX, setup/login flags and output, JSON rendering, active-pot selection, local/managed filters. |
| Host shell | Local daemon or managed backend API. Owns process/API lifecycle, auth, IPC/HTTP, health, logs, and migrations/dependency setup trigger. |
| Pot Management | Active pot, pot CRUD, source registry, lifecycle, status aggregation, export/import metadata. |
| Graph Service | `resolve`, `search`, `record`, `status`, read orchestration, ranking, record lowering. |
| GraphBackend | Store capability bundle: mutation, claim query, semantic search, inspection, analytics, snapshot. |
| Skill Manager | Skill catalog and install/update/remove into agent harnesses. |

The host shell hosts services; it does not contain their business logic. The
same Graph Service (and Skill Manager) modules run in the local daemon and in
the managed backend API. Pot Management is **not** shared: the managed backend
keeps its own pots behind the thin `PotResolutionPort` (scope decision
2026-05-29). Storage adapters and operational dependencies differ by deployment.

## Deployment Shapes

```mermaid
flowchart LR
  cg_entry["potpie CLI"]

  subgraph cg_local_profile["Local profile"]
    direction LR
    cg_local_daemon["local daemon"]
    cg_local_services["Pot Management<br/>Graph Service<br/>Skill Manager"]
    cg_local_storage[("local stores")]
    cg_local_daemon --> cg_local_services --> cg_local_storage
  end

  subgraph cg_managed_profile["Managed backend"]
    direction LR
    cg_managed_api["managed backend API"]
    cg_managed_services["Pot Management<br/>Graph Service<br/>Skill Manager"]
    cg_managed_storage[("hosted stores")]
    cg_managed_api --> cg_managed_services --> cg_managed_storage
  end

  cg_entry --> cg_local_daemon
  cg_entry -. "login + managed pot selection" .-> cg_managed_api
```

Both profiles run the same service modules. Only the host shell and storage
adapters change.

```mermaid
flowchart LR
  cg_sources["GitHub / Linear / other sources"]
  cg_ledger["Event Ledger<br/>managed or self-hosted"]
  cg_event_batches["filtered event pages<br/>and replay tokens"]
  cg_local_graph["local graph"]
  cg_managed_graph["managed graph"]

  cg_sources --> cg_ledger --> cg_event_batches
  cg_event_batches --> cg_local_graph
  cg_event_batches --> cg_managed_graph
```

| Boundary | Local OSS | Managed backend |
|---|---|---|
| Entry | `potpie setup` creates local daemon; CLI defaults to local active pot | `potpie login` authenticates to configured backend URL; selecting a managed pot routes commands there |
| Host | Local daemon shell | Managed backend API |
| Service modules | Pot Management, Graph Service, Skill Manager | Same modules hosted in managed backend API |
| Auth | Local token/socket/OS user | Potpie auth and policy |
| Pot state | Local DB with active `default` pot after setup | Hosted operational DB |
| Graph backend | Embedded/local GraphBackend profile | Hosted graph/search GraphBackend profile |
| Skills | Local catalog/cache and agent target adapters | Hosted catalog/state plus managed target adapters |
| Event Ledger | Optional managed or self-hosted ledger, pulled explicitly | Managed or self-hosted ledger, consumed by hosted workers/services |
| Source integrations | Local scanners by default; no source-provider credentials in the daemon unless configured | Hosted connectors, webhook receivers, queues, and workers |
| Sync | Explicit `potpie cloud ...`; ledger pull does not move graph state to cloud | Native hosted pots; snapshot push/pull remains explicit |

## Component Lifecycle & Setup

`potpie setup` is the single local first-run flow. It is idempotent, makes
Potpie usable in one command, and users never run `daemon` or install commands
on the happy path. The CLI owns the setup UX: flags, validation, local bootstrap
output, and next-command hints. Its local bootstrap job is to install/start the
daemon service. Once the daemon is running, the daemon-hosted
`SetupOrchestrator` owns lifecycle ordering: the ordered sequence of component
setup calls. Each step it calls is a **bespoke per-component method** owned
independently, so the config, storage, auth, pot, skills, and ingestion pieces
can be built and handed over separately.

Three rules make independent ownership work:

- **Daemon first, then dependencies.** The CLI ensures the local daemon service
  exists and is running. The daemon hosts `SetupOrchestrator`, which calls
  `config.ensure_home()`, `backend.provision(plan)`, `pot_service.init(...)`,
  `auth.init_local()`, and so on. Each is a different owner's slot; the
  orchestrator only sequences them.
- **Backends self-provision.** A `GraphBackend` stands up its own store —
  `embedded` writes a local file, `postgres` creates the DB, enables pgvector,
  and runs DDL, `neo4j` pulls its container. The `build_backend(profile)`
  registry already selects by profile; provisioning lives behind the same
  profile, not in setup.
- **Hard deps are profile-scoped.** Every method returns a `StepResult`
  (`done | skipped | not_implemented | failed`); an unbuilt body raises
  `CapabilityNotImplemented("<dotted.slot>")` (e.g. `host.auth.init_local`).
  `SetupOrchestrator` splits **hard deps** (must succeed) from **soft steps**
  (best-effort), but hardness depends on the selected setup host mode. Detached
  daemon service install/start is hard for the normal local daemon profile,
  skipped for in-process/dev profiles, and not part of managed login.

### Setup sequence

```mermaid
sequenceDiagram
  participant User
  participant CLI
  participant Daemon
  participant Setup as daemon-hosted setup
  participant Config
  participant Backend as GraphBackend
  participant Pot as Pot Management
  participant Auth
  participant Skills as Skill Manager
  participant Ingest

  User->>CLI: potpie setup --repo . --backend embedded --agent claude --scan
  CLI->>Daemon: install/start service
  CLI->>Daemon: setup request with SetupPlan
  Daemon->>Setup: run(plan)
  Setup->>Config: ensure_home; write_defaults(plan)
  Setup->>Backend: provision(plan)
  Setup->>Pot: init(mode, backend)
  Setup->>Auth: init_local
  Setup->>Pot: add_source(repo)
  Setup->>Skills: install(agent)
  Setup->>Ingest: scan when --scan
  Setup-->>Daemon: SetupReport
  Daemon-->>CLI: SetupReport
  CLI-->>User: per-step state + next command
```

Ordered steps and their dependency class:

1. CLI bootstrap: install/start local daemon service — hard for local daemon
   profile, skipped if present, skipped for in-process/dev profiles
2. `config.ensure_home()` + `config.write_defaults(plan)` inside the daemon — hard
3. `backend.provision(plan)` — hard, graph/vector self-provision
4. `pot_service.init(mode, backend)` — hard, state store + migrate + default pot
5. `auth.init_local()` — soft, skipped for local no-auth
6. `pot_service.add_source(repo)` — soft
7. `skills.install(agent)` — soft
8. `ingest.scan()` when `--scan` — soft

`--pot <name>` only overrides the initial pot name; without it setup creates and
uses `default`. `--dry-run` calls `orchestrator.preview(plan)` and returns a
`SetupPreview` without executing or returning run `StepResult`s. Re-running setup
is safe: each method is `ensure`-shaped.

### Seam → owner map

Dependency-ordered. Each row is independently ownable behind its method
signature; the orchestrator depends only on those signatures.

| # | Component | Bespoke method(s) | Dep | Code slot |
|---:|---|---|---|---|
| 1 | Daemon process | `ensure`, `install`, `start`, `stop`, `restart` | hard for local daemon profile; skipped for in-process/dev | `host/daemon.py` |
| 2 | Config / workspace | `ensure_home`, `write_defaults`, `get`/`set` | hard | `application/services/config_service.py` |
| 3 | Graph/vector backend | `provision`, `health`, `capabilities` | hard | `domain/ports/graph/backend.py` + backend adapters |
| 4 | Relational state store | `pot_service.init`, `state_store.provision`, `migrator.migrate` | hard | `application/services/pot_management.py` + `adapters/outbound/pots/` |
| 5 | Local auth | `init_local`, `whoami`, `logout` | soft | `application/services/auth_service.py` |
| 6 | Skills | `install(agent)` | soft | `application/services/skill_manager.py` |
| 7 | Ingestion / scan | `scan` | soft | scanner use case |

### Setup skeletons

Lifecycle orchestration is an application use case (`SetupOrchestrator`) hosted
by the daemon, so dependency setup stays outside inbound adapters and outside the
daemon shell. `commands/bootstrap.py:setup` owns the CLI contract: build a
`SetupPlan` from flags → install/start the daemon service → send the plan to the
daemon-hosted orchestrator → render the `SetupReport`.

```python
# domain/lifecycle.py — shared value objects
@dataclass(frozen=True)
class SetupPlan:
    mode: str = "local"        # local setup; managed auth uses LoginPlan
    host_mode: str = "daemon"  # daemon | in_process
    backend: str = "embedded"  # embedded | postgres | neo4j | in_memory
    repo: str | None = "."
    pot: str = "default"
    agent: str = "claude"
    scan: bool = False
    assume_yes: bool = False

@dataclass(frozen=True)
class StepResult:
    step: str
    state: str                 # done | skipped | not_implemented | failed
    detail: str | None = None

@dataclass(frozen=True)
class SetupReport:
    plan: SetupPlan
    steps: tuple[StepResult, ...]
    ok: bool

@dataclass(frozen=True)
class PlannedSetupStep:
    step: str
    hard: bool
    owner: str
    action: str
    skip_reason: str | None = None

@dataclass(frozen=True)
class SetupPreview:
    plan: SetupPlan
    steps: tuple[PlannedSetupStep, ...]
    ok_to_run: bool

class SetupOrchestrator(Protocol):
    def preview(self, plan: SetupPlan) -> SetupPreview: ...   # dry-run
    def run(self, plan: SetupPlan) -> SetupReport: ...        # ensure each, in order

@dataclass(frozen=True)
class LoginPlan:
    backend_url: str | None = None  # defaults to config cloud.backend_url
    org: str | None = None
```

Each named method is a stub raising `CapabilityNotImplemented("<dotted.slot>")`
until its owner ships the body — the same convention the rest of the skeleton
uses today (`host.daemon.install`). The per-component shapes setup calls:

```python
config.ensure_home() -> Path
config.write_defaults(plan: SetupPlan) -> Path

daemon.ensure() -> StepResult                  # install/start service when needed

backend.provision(plan: SetupPlan) -> StepResult           # create DB/index, DDL, docker
pot_service.init(*, mode: str, backend: str) -> StepResult # provision state store + migrate
pot_service.create_pot(name, use=True) -> PotInfo          # create + activate the default pot
auth.init_local() -> StepResult
```

### Managed backend login

Managed backend access is a separate lifecycle from local setup. `potpie login`
authenticates against `cloud.backend_url`, stores the managed session, and makes
managed pots visible to the same pot commands. The default URL points at Potpie
managed; users can point it at a compatible self-hosted backend:

```bash
potpie config set cloud.backend_url https://potpie.example.com
potpie login
potpie pot list --managed
potpie use <managed-pot-name> --managed
```

The CLI treats local and managed pots as the same product object. Internally, an
active pot selection includes an origin (`local` or `managed`) plus a pot id/name.
Commands route by the selected pot: local pots route to the local daemon, while
managed pots route to the authenticated managed backend. If a name exists in both
places, the user must disambiguate with `--local`, `--managed`, or an equivalent
qualified pot id.

## Runtime Flows

### Resolve/search

```mermaid
sequenceDiagram
  participant Agent
  participant CLI
  participant Host as Daemon/API Host
  participant Pot as Pot Management
  participant Graph as Graph Service
  participant Backend as GraphBackend

  Agent->>CLI: potpie resolve "<task>"
  CLI->>Host: context_resolve DTO
  Host->>Pot: active pot + source freshness
  Host->>Graph: resolve(pot, intent, scope, include)
  Graph->>Backend: claim query + semantic search
  Backend-->>Graph: claims + scored keys
  Graph-->>Host: AgentEnvelope
  Host-->>CLI: response DTO
  CLI-->>Agent: human output or --json
```

Code slots: `commands/query.py:resolve` → `HostShell.agent_context`
(`application/services/agent_context.py`) → `GraphService.resolve`
(`application/services/graph_service.py`) → `ReadOrchestrator`
(`application/services/read_orchestrator.py`) over `GraphBackend.claim_query` →
`AgentEnvelope` (`domain/agent_envelope.py`).

### Record

```mermaid
flowchart LR
  cg_record_cmd["potpie record"]
  cg_validate["validate record"]
  cg_emit["deterministic claim emitter"]
  cg_write["GraphMutationPort"]
  cg_project["rebuild/update projections"]

  cg_record_cmd --> cg_validate --> cg_emit --> cg_write --> cg_project
```

Code slots: `commands/query.py:record` → `HostShell.agent_context` →
`GraphService.record` (`graph_service.py:_lower_record` lowers a `RecordRequest`
to a `ReconciliationPlan`) → `GraphBackend.mutation.apply`
(`domain/ports/graph/mutation.py`). POC lowering emits a generic `RELATES_TO`
claim; mapping `record_type` → ontology predicates is the next step.

### Ingestion

```mermaid
flowchart LR
  cg_register["potpie source add repo ."]
  cg_scan["potpie ingest scan"]
  cg_scanners["scanner adapters"]
  cg_ledger_pull["potpie ledger pull"]
  cg_ingest_events["ingestion event run"]
  cg_harness["processing harness"]
  cg_graph_service["Graph Service"]
  cg_backend["GraphBackend"]

  cg_register --> cg_scan --> cg_scanners --> cg_graph_service --> cg_backend
  cg_ledger_pull --> cg_ingest_events --> cg_harness --> cg_graph_service --> cg_backend
```

Registering a source records metadata. Scanning and ledger event processing both
enter through ingestion. Pulling from an Event Ledger reads normalized source
events from a managed or self-hosted ledger; with `--apply`, the CLI/host starts
an ingestion processing run for those events. The processing harness lowers events
into structured records or validated mutations, and Graph Service owns the final
graph write path to the active GraphBackend.

### Event Ledger Pull

```mermaid
sequenceDiagram
  participant CLI
  participant Host as Daemon/API Host
  participant Pot as Pot Management
  participant Ledger as Event Ledger
  participant Queue as Consumer Ingestion Ledger
  participant Ingest as Ingestion Harness
  participant Graph as Graph Service
  participant Backend as GraphBackend

  CLI->>Host: potpie ledger pull --apply
  Host->>Pot: active pot + consumer cursor
  Host->>Ledger: query/fetch event page with filters + cursor
  Ledger-->>Host: events + next cursor
  Host->>Queue: persist events + next cursor as an ingestion run
  Host->>Pot: advance consumer cursor after durable enqueue
  Queue->>Ingest: process pending events
  Ingest->>Graph: write structured records / mutations
  Graph->>Backend: mutations + projection updates
  Ingest->>Queue: mark applied / retryable failed / terminal failed / timed out
  Queue->>Pot: update source freshness and lag
```

Code slots: `commands/ledger.py:pull` → `HostShell.ledger`
(`LedgerFacade` in `host/shell.py`) orchestrates query/fetch → durable enqueue →
advance consumer cursor: `EventLedgerClientPort.query/fetch`
(`adapters/outbound/ledger/`) → consumer ingestion ledger
(`LedgerEventRunStorePort`, local or hosted graph-side state) →
`LedgerCursorStorePort.set` after durable enqueue → ingestion event
processor/harness (`application/use_cases/` plus scanner/reconciliation adapters
as needed) → `GraphService.record` or a Graph Service ingestion method →
`GraphBackend` mutation ports. The ledger client never writes to the graph
backend directly.

The graph consumer owns processing state for pulled events:

| State | Meaning |
|---|---|
| `pending` | Event was durably enqueued from the ledger and awaits processing. |
| `processing` | A worker/harness has leased the event. |
| `applied` | Graph Service accepted the resulting record or mutation. |
| `failed_retryable` | Processing failed but can be retried with backoff. |
| `failed_terminal` | Processing cannot proceed without data/code/operator action. |
| `timed_out` | A processing lease expired; the event can be retried or inspected. |

Because events are first written into the consumer ingestion ledger, the consumer
cursor can advance after durable enqueue, not after every event is fully applied.
Retries, timeouts, dead-letter decisions, and idempotency checks happen from the
consumer ingestion ledger. If that local/hosted consumer state is lost, the Event
Ledger replay tokens and query/filter API let the graph rehydrate from an earlier
position.

The local daemon does not need to be internet-facing. A local profile can log in
to Potpie managed, use the managed Event Ledger for GitHub/Linear/webhook
events, and still keep the graph store local. A self-hosted Event Ledger follows
the same pull contract.

## Agent Contract

Agents see one data-plane contract in every deployment. In OSS self-serve, the
agent uses the `potpie` CLI and the CLI calls the local daemon. In managed
Potpie, the hosted API implements the same contract.

| Tool | Role |
|---|---|
| `context_resolve` | Primary task-context read. Use before non-trivial work. |
| `context_search` | Targeted lookup when the agent already knows what to find. |
| `context_record` | Durable write for reusable project memory. |
| `context_status` | Cheap health, readiness, capability, freshness, and skill check. |

New use cases become parameters, include families, readers, record types, or
skills. They do not become new public tools.

Common request fields:

| Field | Meaning |
|---|---|
| `pot_id` | Pot scope. Callers may use `current`, `local/<name-or-id>`, or `managed/<name-or-id>`; the selected pot determines local-daemon vs managed-backend routing. |
| `intent` | Task shape: `feature`, `debugging`, `review`, `operations`, `planning`, `docs`, `onboarding`, `refactor`, `test`, `security`, or `unknown`. |
| `include` | Evidence families to retrieve. |
| `scope` | Repo, service, file, PR, ticket, user, environment, or time window. |
| `mode` | Retrieval depth: `fast`, `balanced`, `verify`, or `deep`. |
| `source_policy` | Evidence policy: `references_only`, `summary`, `verify`, or `snippets`. |

Reader-backed includes today:

- `coding_preferences`
- `infra_topology`
- `timeline`
- `prior_bugs`
- `raw_graph`

Planned includes must appear in `unsupported_includes` with
`reason=not_implemented` until they are actually backed:

- `decisions`
- `docs`
- `owners`

The resolve/search response is an `AgentEnvelope`:

| Field | Meaning |
|---|---|
| `items[]` | Ranked evidence items with include, score, coverage status, payload, and source refs. |
| `coverage[]` | Per-include availability and completeness. |
| `unsupported_includes[]` | Requested includes that cannot be served yet. |
| `overall_confidence` | High-level confidence for the returned context. |
| `metadata` | Additive transport or implementation metadata. |

Public `context_record` types:

- `preference`
- `policy`
- `bug_pattern`
- `fix`
- `verification`
- `decision`
- `investigation`
- `diagnostic_signal`
- `workflow`
- `feature_note`
- `service_note`
- `runbook_note`
- `integration_note`
- `incident_summary`
- `doc_reference`

`context_status` is sectioned by owner so readiness remains debuggable:

| Section | Owner | Reports |
|---|---|---|
| `host` | Host shell / daemon / managed backend API | liveness, version, IPC/HTTP reachability, auth transport, logs path, managed backend URL host when safe |
| `pot` | Pot Management | active pot id/name/origin, migrations, source registry, source freshness, graph consumer cursor lag |
| `graph_service` | Graph Service | data-plane readiness, supported includes, unsupported includes, record types, reader availability |
| `backend` | GraphBackend | profile/name, capabilities, canonical store health, semantic index readiness, projection repair status |
| `ledger` | Event Ledger consumer binding | binding kind, auth, source list reachability, consumer cursor, retry backlog, timed-out leases, dead-letter backlog |
| `skills` | Skill Manager | catalog readiness, installed-vs-recommended drift, optional install/update nudge |

A skill nudge may include an exact `potpie skills install ...` command; the
install still happens through the CLI.

## GraphBackend

A backend is a set of capability ports, not a database handle.

```mermaid
flowchart TB
  cg_backend["GraphBackend"]
  cg_mutation["GraphMutationPort"]
  cg_query["ClaimQueryPort"]
  cg_semantic["SemanticSearchPort"]
  cg_inspection["GraphInspectionPort"]
  cg_analytics["GraphAnalyticsPort"]
  cg_snapshot["GraphSnapshotPort"]

  cg_backend --> cg_mutation
  cg_backend --> cg_query
  cg_backend --> cg_semantic
  cg_backend --> cg_inspection
  cg_backend --> cg_analytics
  cg_backend --> cg_snapshot
```

| Capability | Required | Notes |
|---|---:|---|
| Mutation | yes | Apply validated mutations, invalidations, resets, readiness. |
| Claim query | yes | Read canonical claims for readers and label lookup. |
| Semantic search | yes | Vector search over claim facts; local embedder by default. |
| Inspection | derivable | Neighborhoods, paths, labels, graph slices. |
| Analytics | derivable | Counts, freshness, quality checks, repair. |
| Snapshot | derivable | Portable pot export/import. |

The canonical claim store is the only source of truth. Vector indexes,
inspection views, and analytics rollups are projections that can be rebuilt.

Default profiles:

| Profile | Purpose |
|---|---|
| `embedded` | OSS default: local, no Docker, vector search included. |
| `in_memory` | Tests and conformance. |
| `neo4j`, `postgres/pgvector`, `chroma` | Optional profiles behind the same ports. |
| hosted profile | Managed API server storage/search adapter. |

## Skill Manager

Skills teach agent harnesses how to use the CLI and four-tool workflow. They are
not graph facts and not new agent tools.

```mermaid
flowchart LR
  cg_skills_cli["potpie skills ..."]
  cg_skill_manager["Skill Manager"]
  cg_skill_catalog["catalog"]
  cg_target_adapter["agent target adapter"]
  cg_status_nudge["context_status skills nudge"]

  cg_skills_cli --> cg_skill_manager
  cg_skill_manager --> cg_skill_catalog
  cg_skill_manager --> cg_target_adapter
  cg_skill_manager --> cg_status_nudge
```

`context_status` may report missing/outdated skills and provide an install
command. The install still happens through the CLI.

## Managed Backend API

Managed Potpie hosts the same service modules behind a managed backend API:

- Potpie auth, teams, roles, billing, and collaboration policy.
- Hosted operational, graph/search, and skill/catalog stores.
- Workers and queues for async ingestion that call the same services.
- Hosted graph/search profile.
- Hosted observability and cost telemetry.
- Cloud skill sync.

It does not add a cloud-only graph model or a separate agent contract. The
managed backend API is another host for Pot Management, Graph Service, and Skill
Manager, backed by hosted databases. `potpie login` authenticates to the
configured backend URL; this can be Potpie managed or a compatible self-hosted
backend.

## Event Ledger

The Event Ledger is a separate managed or self-hostable service for source
events. It is not the Context Graph source of truth.

```mermaid
flowchart LR
  cg_sources["GitHub / Linear / other sources"]
  cg_event_ledger["Event Ledger<br/>webhooks + normalized events<br/>query/filter + replay tokens"]
  cg_local_consumer["local graph<br/>consumer cursor + ingestion ledger"]
  cg_managed_consumer["managed graph<br/>consumer cursor + ingestion ledger"]

  cg_sources --> cg_event_ledger
  cg_event_ledger --> cg_local_consumer
  cg_event_ledger --> cg_managed_consumer
```

Responsibilities:

- receive webhooks and poll source APIs;
- normalize source events and keep replayable event history;
- expose ordered event pages, replay tokens, source listings, and query/filter
  APIs for pull consumers;
- keep third-party credentials and webhook receivers out of the local daemon by
  default;
- let local graphs use managed integrations without pushing graph state to
  managed storage.

The Graph Service remains responsible for turning records/events into claims and
applying graph mutations. The Event Ledger only supplies ordered source events.
Consumer cursor state belongs to the graph deployment that pulled the events,
along with per-event processing state, retry counters, lease timeouts, and
dead-letter records. The ledger may separately keep provider ingestion cursors
for webhook/polling work, but it does not own a graph's last-applied or
last-enqueued position.

## Code Map

All paths are under `app/src/context-engine/`. The engine runs one mode-based
read contract (`resolve`/`search` → `AgentEnvelope`, no server-side answer
synthesis) across two composition roots: the local agent spine
(`build_host_shell`, behind the CLI + MCP) and the managed HTTP ingestion server
(`build_ingestion_server`, consumed by the parent app). Rows marked _(local POC)_
have a working body sufficient for the local profile with deeper production work
deferred; _(deferred)_ rows are the managed pipeline not yet migrated onto
`HostShell`.

| Area | Path |
|---|---|
| Agent contract (4 tools) | `domain/ports/agent_context.py` (`AgentContextPort` + request/response DTOs) |
| Services (interfaces) | `domain/ports/services/{graph_service,pot_management,skill_manager}.py` |
| Graph capability ports | `domain/ports/graph/{backend,mutation,claim_query,semantic,inspection,analytics,snapshot}.py` |
| Event Ledger consumer ports | `domain/ports/ledger/{client,cursor,reconciler}.py` (`LedgerEventRunStorePort` / `run_store.py` _(planned — HU4)_) |
| Event Ledger reconciler port | `domain/ports/ledger/reconciler.py` (`EventReconcilerPort`); local impl `adapters/outbound/ledger/reconciler.py` _(parked: LLM-vs-deterministic strategy)_ |
| Host shell + daemon | `host/{shell,daemon}.py` |
| Composition root | `bootstrap/host_wiring.py` (`build_host_shell`) |
| Agent contract impl | `application/services/agent_context.py` (composes the three services) |
| Graph Service _(local POC)_ | `application/services/graph_service.py` over a `GraphBackend` + `read_orchestrator.py`; `record` lowers `record_type` → ontology predicate via `RECORD_TYPES` |
| Pot Management _(local POC)_ | `application/services/pot_management.py` + `adapters/outbound/pots/local_pot_store.py` |
| Skill Manager _(local POC)_ | `application/services/skill_manager.py` + `adapters/outbound/skills/{bundle_catalog,claude_target}.py` |
| GraphBackend adapters | `adapters/outbound/graph/backends/{in_memory,embedded,neo4j}_backend.py` + `build_backend` registry; `claim_query_analytics.py` gives any claim-backed profile real analytics |
| Event Ledger adapters _(local POC)_ | `adapters/outbound/ledger/{managed_client,self_hosted_client,cursor_store,reconciler}.py` (`run_store.py` _(planned — HU4)_) |
| Ingestion (working-tree scan) | `application/services/ingest_service.py` + `adapters/outbound/scanners/default_registry.py` → `HostShell.ingest` (`potpie ingest scan`) |
| CLI (host-routed) | `adapters/inbound/cli/host_cli.py` + `adapters/inbound/cli/commands/` |
| Readers | `application/readers/`, `application/services/read_orchestrator.py` |
| Scanners | `domain/ports/config_scanner.py`, `application/use_cases/scan_working_tree.py`, `adapters/outbound/scanners/` |
| Structured records / ontology | `domain/context_records.py`, `domain/ontology.py`, `domain/agent_context_port.py` (vocab) |
| Managed read/write facade | `domain/ports/context_graph.py`, `adapters/outbound/graph/context_graph_service.py` — envelope-only (`query`/`query_async` → one read trunk; no goal/synthesis branches) |
| MCP (host-routed) | `adapters/inbound/mcp/server.py` — the four tools served in-process via `build_host_shell` |
| HTTP ingestion server _(deferred)_ | `adapters/inbound/http/`, `bootstrap/{ingestion_server,standalone_container}.py` — its own composition root; the async pipeline migrates onto `HostShell` over time |
| Reconciliation agent _(parked)_ | `adapters/outbound/reconciliation/` (candidate body for the ingestion event-processing harness) |
| Managed API adapter | `app/modules/context_graph/` |

### Interfaces

The stable contracts. Methods may be added; the interfaces and their package
boundaries are not expected to change.

| Interface | File | Role |
|---|---|---|
| `AgentContextPort` | `domain/ports/agent_context.py` | The four tools (`resolve`/`search`/`record`/`status`); the only public agent surface. |
| `GraphService` | `domain/ports/services/graph_service.py` | Data plane: readers, ranking, record lowering, envelopes. |
| `PotManagementService` | `domain/ports/services/pot_management.py` | Control plane: pots, active pot, sources, readiness rollup. |
| `SkillManager` + `AgentTargetPort` | `domain/ports/services/skill_manager.py` | Catalog + per-harness install drift; advisory `SkillNudge`. |
| `GraphBackend` | `domain/ports/graph/backend.py` | Six-capability storage bundle + `profile` + `capabilities()`. |
| `GraphMutationPort` / `ClaimQueryPort` | `domain/ports/graph/{mutation,claim_query}.py` | Canonical source of truth (write / read). |
| `SemanticSearchPort` / `GraphInspectionPort` / `GraphAnalyticsPort` / `GraphSnapshotPort` | `domain/ports/graph/{semantic,inspection,analytics,snapshot}.py` | Rebuildable projections. |
| `EventLedgerClientPort` | `domain/ports/ledger/client.py` | Query/filter/pull normalized source events from the Event Ledger. |
| `LedgerCursorStorePort` | `domain/ports/ledger/cursor.py` | Graph-consumer cursor: last enqueued/applied position per (pot, source). |
| `EventReconcilerPort` | `domain/ports/ledger/reconciler.py` | Lowers normalized events → graph claims (parked: deterministic vs LLM strategy). |
| `LedgerEventRunStorePort` _(planned — HU4)_ | `domain/ports/ledger/run_store.py` | Per-event processing state: status, retries, leases, dead letters. |
| `HostShell` / `Daemon` | `host/{shell,daemon}.py` | In-process facade over the services + local lifecycle. |

An unbuilt capability raises `domain.errors.CapabilityNotImplemented` (a dotted
`graph.<profile>.<cap>.<method>` slot), which inbound adapters render as the
structured not-implemented contract — never a bare `NotImplementedError`.

## Extension Points

The rule of thumb: add behavior at the narrowest service boundary that owns it.
CLI adapts user intent, the daemon hosts services, Pot Management owns control
plane behavior, Graph Service owns data-plane behavior, and GraphBackend owns
physical storage.

```mermaid
flowchart LR
  cg_cli_command["CLI command"]
  cg_use_case["application use case"]
  cg_service_boundary["service boundary"]
  cg_domain_port["domain port"]
  cg_adapter["adapter"]

  cg_cli_command --> cg_use_case --> cg_service_boundary --> cg_domain_port --> cg_adapter
```

| Change | Put it here | Rule |
|---|---|---|
| Reader/include | `application/readers/` + read orchestrator | Read through `ClaimQueryPort`; do not query stores directly. |
| Scanner | `adapters/outbound/scanners/` + scanner use case | Emit validated graph mutations or structured records through Graph Service. |
| Ledger event processor | ingestion use case + processing harness | Pull events are durably enqueued in consumer state, processed through ingestion, then written through Graph Service. Do not let ledger clients write to GraphBackend directly. |
| Record type | `domain/ontology.py`, `domain/context_records.py` | Add deterministic claim emission when possible. |
| Entity/predicate | `domain/ontology.py` | Add identity, endpoint rules, freshness, and source-of-truth metadata. |
| Graph backend | `domain/ports/graph/` + backend adapter | Implement mandatory ports, preserve pot isolation, pass conformance. |
| Skill | Skill catalog + `AgentTargetPort` adapter | Keep skill content harness-neutral; do not put it in the graph. |
| Pot behavior | Pot Management Service | Preserve first-setup active `default` pot. |
| Event Ledger connector | Event Ledger service adapter | Normalize provider events, own webhook/polling concerns, expose query/filter and replay-token-based pull. |
| Managed host behavior | Managed backend API + shared services | Reuse Pot Management, Graph Service, and Skill Manager; swap only host/storage adapters. |
| Setup/lifecycle step | Component's bespoke method + `SetupOrchestrator` sequence | Return a `StepResult`; raise `CapabilityNotImplemented` until built; declare hard vs soft for the selected host mode. |
| CLI command | CLI -> selected pot service/use case | Commands route by active/selected pot; selecting a managed pot after login is the explicit remote boundary. |

Do not extend by bypassing the read orchestrator, querying physical stores from
CLI/readers, making projections a second source of truth, putting service
business logic in the daemon shell, exposing skill management as an agent tool,
or duplicating ontology enums in docs/CLI/cloud-only code.

## Implementation Order

The architectural skeleton landed steps 1–2, 5–7, and 9 as **stable interfaces +
host-routed CLI + POC adapters** (see the Code Map). What remains is replacing
POC bodies with real implementations behind the unchanged ports: a persistent
embedded store with real vectors (step 3), a detached daemon (step 4), snapshots
+ cloud sync (step 8), and managed hosting (step 10) — plus folding the parked
Neo4j and reconciliation modules in behind the GraphBackend and ingestion
event-processing seams.

1. Create graph capability ports and `GraphBackend`.
2. Add conformance suite and in-memory backend.
3. Implement embedded backend with vector search.
4. Add setup-aware daemon with local auth, health, logs, and service-manager
   install/start.
5. Add local Pot Management with `default` active pot creation and source
   registry.
6. Route CLI `resolve/search/record/status` through the daemon.
7. Finish `pot`, `source`, `ingest`, `graph`, `backend`, and `skills` commands.
8. Add login to configurable managed backend URL, unified local/managed pot
   selection, snapshots, cloud push/pull, and managed skill sync.
9. Add Event Ledger client, consumer cursor/run storage, `ledger` CLI,
   query/filter commands, and ingestion event-processing path with retry and
   timeout handling.
10. Add managed backend API hosting the same services on hosted stores.

## Rules

- OSS graph use works without cloud auth.
- CLI is the primary user/agent surface.
- Setup creates the daemon; the daemon-hosted setup flow creates the active local
  `default` pot and provisions local dependencies.
- Setup hard dependencies are scoped to host mode: service-manager registration
  is hard for detached local daemon setup, skipped for in-process/dev setup, and
  outside `potpie login`.
- `potpie setup --dry-run` returns `SetupPreview`, not executed `StepResult`s.
- `potpie login` authenticates to a configured managed backend URL; it does not
  run local setup.
- Local and managed pots use the same CLI pot surface; `--local` and `--managed`
  filters disambiguate.
- `context_status` is sectioned by owner: host, pot, graph service, backend,
  ledger, and skills.
- The same service modules run in the local daemon and managed backend API.
- Pot Management owns control plane; Graph Service owns data plane.
- CLI/readers never query physical stores directly.
- Skills are CLI-managed recipes, not graph data and not a fifth tool.
- The Event Ledger is a source-event stream, not graph storage.
- Event Ledger consumers store their own cursor, per-event status, retries,
  leases, and dead letters.
- Local graphs may pull from managed or self-hosted ledgers only after explicit
  ledger configuration.
- Managed-only concerns stay outside the local daemon unless explicitly
  configured by the user.
