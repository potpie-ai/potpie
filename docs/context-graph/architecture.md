# Context Graph Architecture

Last reviewed: 2026-05-28.

The Context Graph has two deployment shapes:

1. **Local open-source self-serve.** `pip install potpie`, run the CLI against
   a local daemon shell, and keep graph state on the user's machine.
2. **Managed Potpie cloud.** Host the same services behind Potpie auth,
   shared pots, managed workers, hosted storage, and cloud integrations.

The local shape is the default product direction for open source. Managed
Potpie remains a hosted deployment of the same service modules, not a separate
graph product.

## Boundary Rule

Keep one graph model, one agent contract, and one set of service interfaces.
Only process wiring, auth, storage adapters, and hosted integrations differ.

| Boundary | Local OSS | Managed cloud |
|---|---|---|
| User entry | CLI defaults to local | Hosted API/profile |
| Process | User daemon shell | Hosted API + workers |
| Auth | Local token, socket, or OS user permissions | Potpie auth and policy |
| Pot state | Local state DB with an active `default` pot after setup | Hosted operational DB |
| Graph backend | Embedded profile (single store) | Hosted graph/search profile |
| Ingestion | Agent-mediated records and scanners | Webhooks, workers, reconciliation |
| Sync | Explicit export/import or cloud push | Native hosted pots |

Do not put cloud users, billing, shared auth, hosted webhooks, or remote source
credentials into the local daemon. Do not fork the graph schema or agent tools
for cloud.

The CLI-first OSS operating contract lives in
[`oss-self-serve-flow.md`](./oss-self-serve-flow.md). That doc defines the target
user journey and command semantics; this architecture doc defines the service
boundaries those commands call.

---

## Service Boundaries

The system is a set of service boundaries hosted by the daemon shell. Each one
is movable: the same module can run inside the local daemon, a managed API
process, or a managed worker. The contract between boundaries is an interface,
never a physical store.

```text
potpie CLI / local agents (Claude Code, Codex, …)
  v
[1] daemon shell ......... lifecycle, local auth, IPC/HTTP, config, logs, health
  |
  |-- [2] Pot Management Service ... control plane: pot CRUD, status, source
  |                                  registry, analytics, lifecycle, export
  |
  |-- [3] Graph Service ............ data plane: resolve/search/record/status
  |        |                         trunk, read orchestrator, ranking,
  |        |                         answer synthesis
  |        v
  |     [4] Graph Backend .......... swappable graph layer: a bundle of
  |            |                     capability ports (mutation, claim query,
  |            |                     inspection, analytics, semantic, snapshot)
  |            v
  |        storage profile .......... binds capabilities to physical stores
  |                                   (embedded SQLite by default)
  |
  |-- [5] Skill Manager Service .... agent-skill lifecycle: catalog, install,
  |        |                         update, drift/nudge, cloud sync
  |        v
  |     agent target adapter ....... renders/installs a skill into a harness
  |                                   (Claude Code, Codex, generic AGENTS.md)
  v
local state DB (Pot Management) + graph backend (Graph Service)
       + skill cache (Skill Manager)
```

Each boundary has a single responsibility, a stable interface, and an explicit
list of things it must **not** do. This table is the contract; the rest of the
doc expands each row.

| # | Boundary | Owns | Interface in | Must not |
|---|---|---|---|---|
| 1 | **Daemon shell** | process lifecycle, OS service install, local auth/IPC, config, logs, health, migrations trigger | transport (HTTP/socket) | hold graph, pot, source, or skill business logic; import domain types beyond DTOs |
| 2 | **Pot Management Service** | pot CRUD, status, source registry, local permissions, analytics summaries, export/import metadata, lifecycle (reset/archive) | `PotResolutionPort`, `PotSourceListingPort`, local state DB | run physical graph queries; hold source-provider credentials in local mode; require an LLM in local mode |
| 3 | **Graph Service** | the agent-facing resolve/search/record/status trunk, read orchestration, ranking, answer synthesis, capability routing | `ContextGraphPort` (facade) over a `GraphBackend` | know which physical store is underneath; orchestrate multi-store writes |
| 4 | **Graph Backend** | a bundle of graph capability ports for one pot's graph | `GraphBackend` (see [Graph Service Interfaces](#graph-service-interfaces)) | leak store topology upward; expose more than the capability ports |
| 5 | **Skill Manager Service** | agent-skill catalog, install/update/remove into harnesses, drift detection, the `context_status` nudge, OSS-catalog download, cloud sync | `SkillCatalogPort`, `AgentTargetPort`, `SkillSyncPort` (see [Skill Manager Service](#skill-manager-service)) | become a fifth agent tool; run agent reasoning or execute skills; put skill content in the graph |

The naming discipline matters: **Pot Management Service** is the control plane
(business/admin), **Graph Service** is the data plane (graph read/write/search),
**Skill Manager Service** is the agent-integration plane (what skills the harness
should have). Do not call them all "graph service." Pot Management *orchestrates
and exposes*; the Graph Service *computes*; the Skill Manager *provisions
harnesses*. Analytics and status are computed by the Graph Backend and surfaced
by Pot Management — Pot Management never reimplements them.

---

## Graph Model

The portable graph model is the contract every backend implements:

- **Pot:** isolation boundary for every entity, claim, record, and query.
- **Entity:** stable project object identified by `(pot_id, entity_key)`.
- **Claim:** typed canonical fact about an entity or relationship, with
  predicate, source refs, observed time, valid time, and invalidation fields.
  Claims are stored as canonical `:RELATES_TO` edges (Position B).
- **Source ref:** compact pointer back to evidence (file path, PR, issue, doc
  URL, alert, deploy, scanner output).
- **Record:** agent-facing durable write that lowers into claims.

Two invariants govern everything below:

1. **The canonical claim store is the single source of truth.** Every other
   structure a backend keeps — semantic/vector index, full-text index, a
   graph-native traversal copy, analytics rollups — is a *derived projection*
   that can be rebuilt from claims. Nothing is ever a co-equal second source of
   truth.
2. **The graph stores compact claims and evidence pointers, not full source
   payloads.** Readers, CLI, cloud APIs, and Pot Management must not depend
   on a physical database shape.

Invariant 1 is what makes multiple stores safe (see
[Consistency](#consistency-the-projection-rule)): consistency degrades to
"a projection is stale," which is repairable by reindex, never to "split brain,"
which is not.

---

## Graph Service Interfaces

This is the part that makes the graph layer swappable. Read it as the contract
for adding or replacing a backend.

### Capabilities are interfaces; stores are bindings

The graph layer is defined by **what it can do** (capabilities), decoupled from
**where the data physically lives** (stores). These are two independent axes:

- A **capability** is a narrow port — one slice of graph behavior.
- A **storage profile** binds each capability to a physical store.

The default profile binds *all* capabilities to *one* embedded store. A heavy
profile may bind semantic search to pgvector and traversal to Neo4j. Same
capability set, different topology. This is why "one embedded store by default"
(install simplicity) and "six capabilities" (clean abstraction) are not in
tension — they live on different axes.

### The capability ports

Six capability ports live under `domain/ports/graph/` — store-neutral, no
adapter may import them from a Neo4j (or any store) module. They form a
**3 + 3 ladder**: three are mandatory, three are derivable from the mandatory
ones.

| Capability | Port | Tier | Responsibility |
|---|---|---|---|
| Mutation | `GraphMutationPort` | **mandatory** | Apply validated entity/claim/source-ref mutations and invalidations; reset a pot; ensure store readiness. |
| Claim query | `ClaimQueryPort` | **mandatory** | Read canonical claims for readers; bulk entity-label lookup. |
| Semantic search | `SemanticSearchPort` | **mandatory** | Vector index/search over claim facts. Every backend ships it; embeddings come from a local model by default (no cloud key). |
| Inspection | `GraphInspectionPort` | derivable | Neighborhoods, paths, degree, labels, predicates, raw graph slices for visualization/debug. |
| Analytics | `GraphAnalyticsPort` | derivable | Counts, freshness, orphaned nodes, source coverage, quality checks; projection `repair()`. |
| Snapshot | `GraphSnapshotPort` | derivable | Export/import a portable pot snapshot for backup, migration, and cloud push. |

Illustrative signatures (canonical types live in code; treat these as shape,
not gospel):

```python
# domain/ports/graph/  — store-neutral capability ports

class GraphMutationPort(Protocol):                       # MANDATORY
    @property
    def enabled(self) -> bool: ...
    async def ensure_ready(self) -> bool: ...            # indexes / migrations
    async def apply(self, plan, *, pot_id, provenance=None): ...
    async def invalidate(self, ..., *, pot_id) -> None: ...
    async def reset_pot(self, pot_id: str) -> dict: ...

class ClaimQueryPort(Protocol):                          # MANDATORY
    def find_claims(self, f: ClaimQueryFilter) -> list[ClaimRow]: ...
    def entity_labels(self, *, pot_id, entity_keys) -> Mapping[str, tuple[str, ...]]: ...

class SemanticSearchPort(Protocol):                      # MANDATORY (vector)
    def index(self, claims) -> None: ...                 # embed + upsert into vector index
    def search(self, *, pot_id, query, filter, limit) -> list[ScoredKey]: ...

class GraphInspectionPort(Protocol):                     # derivable
    def neighborhood(self, *, pot_id, entity_key, depth=1) -> GraphSlice: ...
    def path(self, *, pot_id, src, dst, max_hops) -> list[GraphSlice]: ...

class GraphAnalyticsPort(Protocol):                      # derivable
    def status(self, *, pot_id) -> GraphStatus: ...      # counts/freshness/coverage
    def repair(self, *, pot_id) -> RepairReport: ...     # rebuild projections from claims

class GraphSnapshotPort(Protocol):                       # derivable
    def export(self, *, pot_id) -> PotSnapshot: ...
    def import_(self, snapshot, *, pot_id) -> ImportReport: ...
```

**Why a 3 + 3 ladder.** Mutation, claim query, and vector semantic search are
the three a backend must implement itself — they need real storage behavior the
claim edges cannot fake. Vector search is mandatory because preference and bug
recall *are* semantic recall; lexical-only retrieval is not an acceptable
shippable mode. The other three have a default implementation derived from
claims (a `ClaimDerivedBackend` base): inspection by walking claim edges,
analytics by aggregating claims, snapshot by streaming claims. An adapter author
implements the three mandatory ports and **overrides a derivable port only when
the store can beat the naive version** (Neo4j overrides inspection with Cypher).
The cost of a new adapter is three ports, not six.

This is also the guard against building a Neo4j clone as the abstraction: the
contract is defined by what readers derive from *claims*, not by what Cypher can
express. Neo4j is one binding, never the mental model.

### The `GraphBackend` bundle

A backend is assembled into one composition object that the Graph Service
depends on. This replaces the loose wiring where a writer and a reader could
silently point at different stores.

```python
class GraphCapability(Enum):
    MUTATION; CLAIM_QUERY; INSPECTION; ANALYTICS; SEMANTIC; SNAPSHOT

class GraphBackend(Protocol):
    name: str                                # "sqlite-embedded", "neo4j+pgvector"
    capabilities: frozenset[GraphCapability]
    mutation: GraphMutationPort              # mandatory, always present
    claims: ClaimQueryPort                   # mandatory, always present
    semantic: SemanticSearchPort             # mandatory, vector-backed
    def inspection(self) -> GraphInspectionPort: ...
    def analytics(self) -> GraphAnalyticsPort: ...
    def snapshot(self) -> GraphSnapshotPort: ...
```

`capabilities` is introspectable on purpose: `context_status` reports the active
backend `name`, its capability set, and semantic readiness (embedder + vector
index). That makes "what can this deployment actually do?" a first-class answer,
not a guess.

### Consistency (the projection rule)

> **Store topology is an adapter secret.** The Graph Service calls one
> `GraphMutationPort.apply(...)` and gets one transactional promise. If a profile
> spans multiple physical stores, the *adapter* reconciles them — never the
> application — by writing the canonical claim store first and treating every
> other store as a rebuildable projection.

Consequences:

- **Embedded default:** one store holds claims + the vector index (+ full-text),
  so an apply is one local transaction. No saga, no two-phase commit, no
  eventual-consistency machinery.
- **Multi-store profile:** the adapter writes claims (authoritative), then
  best-effort updates projections. A crash leaves a *stale projection*, not a
  corrupt graph. Recovery is `analytics().repair()` (reindex), never a
  distributed rollback.
- **The application never learns there are two stores.** The day a reader has to
  ask "is the vector index caught up?", the abstraction has leaked.

### Semantic search

Vector semantic search is a required capability of every backend, behind the
Graph Service. It is not coupled to one vector database — the store is a binding.

- The read orchestrator routes semantic candidate retrieval through
  `SemanticSearchPort` for all backends. There is no lexical-only shippable mode.
- Embeddings are generated by a **local embedding model by default** — a small,
  bundled, CPU-friendly embedder that runs offline. Vector search therefore works
  out of the box with no Potpie API key and no external/cloud model key.
- The vector index is a projection of the canonical claims: it is rebuilt by
  re-embedding claims (`analytics().repair()`), so it never becomes a second
  source of truth.
- Managed cloud may swap the local embedder for a hosted embedder and the
  embedded index for a hosted vector store; the capability and contract are the
  same.

The store binding is free to choose: the embedded profile uses a SQLite vector
index (e.g. `sqlite-vec`); heavier profiles use pgvector, Chroma, or Neo4j
vector indexes. The embedder is configurable, but a working local default always
ships.

---

## Storage Profiles

A profile is a factory that returns a `GraphBackend`. Profiles are the only
place physical-store choices live.

| Profile | Stores | Capabilities | Use |
|---|---|---|---|
| `embedded` (default) | one SQLite file (claims + vector index via `sqlite-vec` + FTS) | all six, single transaction; vector search via the bundled local embedder | `pip install potpie`; no Docker, no second service |
| `neo4j` / `neo4j+pgvector` | Neo4j (+ pgvector for vectors) | binds semantic to pgvector, overrides inspection (Cypher) | heavy local or managed; accepts consistency cost behind the adapter |
| `in_memory` | process memory | all six; brute-force in-memory vector search | tests, benchmark seeding, the conformance baseline |

Rules:

- For OSS V1, default to `embedded`. Postgres/pgvector, Chroma, and Neo4j are
  opt-in profiles, never default dependencies.
- A local daemon must not require both a relational DB service and a graph DB
  service to be useful — that would feel like running Potpie cloud on a laptop.
  Start with the smallest store shape that satisfies the interface, then add
  heavier profiles where they earn their cost.
- Open question to settle before committing `embedded` to the roadmap: validate
  that SQLite serves **infra multi-hop traversal** and **timeline ordering**
  acceptably. Preference/bug recall on SQLite is easy; recursive-CTE traversal
  is the unproven part. Spike that query first.

---

## Adapter Conformance

What makes "swap the backend" real (not just a diagram) is one shared,
parametrized conformance suite. It takes any `GraphBackend` factory and runs the
same assertions: seed → mutate → claim query → inspection → analytics → semantic
→ snapshot round-trip → `repair()`. A backend is "done" when it passes.

- `in_memory` is the reference baseline and must pass first.
- Every new profile (`embedded`, `neo4j`, …) passes the same suite.
- The benchmark seed/read scenarios run through `GraphBackend` too, so
  "in-memory, SQLite, and Neo4j return equivalent agent envelopes for the same
  corpus" is a test, not a hope. That test is also the local/managed parity
  check (see [`bench-plan.md`](./bench-plan.md)).

---

## Graph Read Path

All agent-visible reads pass through the same trunk:

```text
context_resolve / context_search
  -> ContextGraphQuery
  -> Graph Service
  -> ReadOrchestrator
  -> reader-backed include families
  -> ClaimQueryPort / SemanticSearchPort   (via the active GraphBackend)
  -> AgentEnvelope
```

Current reader-backed include families: `coding_preferences`, `infra_topology`,
`timeline`, `prior_bugs`, `raw_graph`. Advertised-but-not-yet-backed families
(`decisions`, `docs`, `owners`) must be returned as `unsupported_include` with
`reason=not_implemented`, never as empty success.

## Graph Write Path

Local OSS prefers deterministic writes; every write resolves to a validated
mutation through `GraphMutationPort`:

1. `context_record` writes structured durable records.
2. Record emitters lower records to canonical claims.
3. Repo scanners emit facts from files and manifests.
4. Optional harness skills may submit validated mutation plans.
5. Raw-event reconciliation is optional locally and normal in managed cloud.

Managed Potpie keeps raw-event ledgers, workers, and LLM-backed reconciliation
where useful. Local mode does not need a daemon-side LLM key for the default
loop. (The quality gap this creates between the two write paths is real and is
tracked as a product decision, not an adapter detail.)

---

## Pot Management Service

The control plane for a pot. It answers "what graph exists?", "what sources are
attached?", "is the graph ready?", "which backend is active?", "can this pot be
exported?"

Responsibilities: create the local `default` pot on first setup and mark it
active; create/list/update/archive/reset pots; manage local pot metadata,
repositories, source references, and scanner state; expose graph status,
freshness, coverage, and backend health (by calling the Graph Service); expose
analytics summaries; coordinate migrations, export/import, and cloud push/pull
metadata; enforce local pot isolation and permissions.

Non-responsibilities: no physical graph queries; no source-provider credentials
in local mode; no LLM reconciliation requirement in local mode; no cloud
user/team/billing policy in local mode.

Code anchors: managed pot APIs/models in `app/modules/context_graph/`; pot
resolution ports in `domain/ports/pot_resolution.py` and
`domain/ports/pot_source_listing.py`; local mode needs a daemon-local
implementation wired through `bootstrap/local_container.py` (not yet present).

## Skill Manager Service

The Skill Manager owns the lifecycle of the **skills** that third-party agent
harnesses (Claude Code, Codex, …) need in order to use the four tools well. A
skill is a portable recipe — instructions plus a four-tool workflow — that lives
in this repo and/or a public OSS skill catalog. The four tools are the engine's
contract; skills are how a harness is taught to drive them. They are *not* a
fifth agent tool.

It answers: "which skills should this harness have for this pot/intent?", "are
they installed and current?", "fetch this skill from the catalog", "install it
into Claude Code", "sync my skill set with my cloud agent."

### Skill model

- **Skill** — portable unit: id, version, title, description, the intents it
  applies to, its four-tool recipe, and harness-compatibility metadata.
- **Catalog** — the resolvable set of skills: first-party skills shipped in this
  repo plus skills downloadable from the public OSS skill catalog.
- **Bundle** — a skill (or set) rendered into one harness's on-disk format.
- **Agent target** — a swappable per-harness adapter that knows a harness's skill
  format and install location (Claude Code `.claude/commands/`, generic
  `.agents/skills/` + `AGENTS.md`, Codex). This is the same binding pattern as a
  graph storage profile: the skill is neutral; the target renders it.

### Interfaces

```python
# domain/ports/skills/ — harness-neutral skill ports

class SkillCatalogPort(Protocol):
    def list(self) -> list[SkillSpec]: ...                  # repo + remote OSS catalog
    def get(self, skill_id, version=None) -> SkillSpec: ...
    def fetch(self, skill_id, version=None) -> SkillPayload: ...   # download + verify + cache

class AgentTargetPort(Protocol):                            # one per harness
    name: str                                               # "claude", "codex", "default"
    def installed(self, *, root) -> list[InstalledSkill]: ...      # detect + versions
    def install(self, skill: SkillPayload, *, root) -> InstallResult: ...  # render + write, idempotent
    def remove(self, skill_id: str, *, root) -> InstallResult: ...

class SkillSyncPort(Protocol):                              # managed/cloud only
    def push(self, skills, *, agent) -> SyncResult: ...
    def pull(self, *, agent) -> list[SkillSpec]: ...
```

`AgentTargetPort` adapters are how "integrate with different agents" works: add a
harness by adding a target, not by changing skills or the engine. Installs are
idempotent and marker-fenced so re-running them is safe.

### The nudge (advisory, through `context_status`)

The Skill Manager never installs by itself and is never an agent tool. Detection
of "missing/outdated skills for this harness" is surfaced **read-only** through
`context_status`: the status assembler calls the Skill Manager for the active
harness and pot/intent, and the response carries a `skills` block —
`recommended` / `installed` / `missing` / `outdated` plus the exact
`potpie skills install …` command. The agent relays that to the user; the user
(or a CLI step) runs the install. This keeps the nudge inside the four-tool
contract and keeps `context_status` strictly advisory — it recommends, it never
causes a side effect.

### CLI

```bash
potpie skills list                         # catalog: repo + OSS, with versions
potpie skills install <id> [--agent claude] [--path .]
potpie skills update [--all]               # bump installed skills to catalog latest
potpie skills remove <id>
potpie skills status                       # installed vs recommended vs outdated, per harness
potpie skills add <path|url>               # author/register a new skill into the catalog
potpie cloud skills sync                   # explicit; requires cloud login
```

### Local vs managed

- **Local:** download from the public OSS catalog with no Potpie API key; install
  into the local harness. The skill cache is a local directory.
- **Managed:** `SkillSyncPort` syncs a user's skill set with their cloud agents.
  Sync is explicit and requires cloud login; it never happens silently.

Non-responsibilities: not a fifth agent tool; does not run agent reasoning or
execute skills (the harness does); does not put skill content in the graph
(skills are agent config, not project facts); no silent cloud sync; no source or
cloud credentials in the local skill path.

Code anchors: `adapters/inbound/cli/agent_installer.py` (today's installer,
generalizes into the `default`/`codex`/`claude` agent targets) and
`adapters/inbound/cli/templates/{agent_bundle,claude_bundle}/` (today's bundles,
become first-party catalog entries). The ports under `domain/ports/skills/`, the
remote OSS catalog client, and `SkillSyncPort` are new.

## Local Daemon

```text
agent harness  -> potpie CLI (local profile)  -> daemon shell
  daemon shell: local auth + lifecycle | Pot Management | Graph Service
                | Skill Manager | Scanner/Record | Export/Sync
  v
local state DB + GraphBackend (embedded profile) + skill cache
```

The daemon shell is deliberately boring: process lifecycle, OS service install,
health checks, migrations, local auth, logs, config, and a stable IPC/HTTP
surface. It hosts services; it does not contain graph business logic, and it
must not import domain types beyond DTOs. Local graph use must not require
`POTPIE_API_KEY`, cloud login, a daemon-side LLM key, or remote source
credentials.

The CLI is the product-facing local control surface. It owns command UX, output
formatting, daemon discovery, local/cloud profile selection, and skill setup
orchestration. It must call daemon services and capability ports through stable
DTOs; it must not query SQLite, Neo4j, vector indexes, or local state tables
directly.

## Managed Architecture

```text
hosted clients -> Potpie API and auth
  |-- Managed Pot Management Service
  |-- Managed Graph Service           (same trunk, hosted GraphBackend profile)
  |-- Managed Skill Manager Service   (catalog + SkillSyncPort to cloud agents)
  |-- Managed Event Ledger / workers
  v
hosted state DB + hosted graph/search profile
```

Managed Potpie keeps cloud-specific concerns: user/team/role/policy/billing;
hosted pot CRUD and collaboration; hosted source credentials and registry;
webhook receivers and event-ledger services; queue-based reconciliation workers;
hosted graph/search storage and observability; and skill sync to a user's cloud
agents via `SkillSyncPort`. These wrap the same Graph Service, Skill Manager, and
graph model; they do not fork the local daemon's agent contract.

## Cloud And Event Ledger

```bash
potpie cloud login
potpie cloud push
potpie cloud pull
potpie cloud status
```

Webhook ingestion is a separate event-ledger service: it receives cloud
webhooks, stores normalized events, and exposes cursor-based reads. A local user
or agent harness can pull from the ledger and decide what to record into the
local graph. Local daemons do not need to be internet-facing. The event ledger
is operational input; the context graph remains the fact store.

---

## Core Code To Reuse

| Area | Current path | Role in the new shape |
|---|---|---|
| Domain model | `domain/` | Keep ontology, graph query models, mutation types, context records, ranking, agent envelope as deployment-neutral core. |
| Graph capability ports | new `domain/ports/graph/` | Home for `GraphMutationPort` (moved out of the Neo4j module), `ClaimQueryPort` (moved in), and the inspection/analytics/semantic/snapshot ports. |
| Graph facade port | `domain/ports/context_graph.py` | Keep `ContextGraphPort` as the Graph Service facade over a `GraphBackend`. |
| Graph Service | `adapters/outbound/graph/context_graph_service.py` | Take a `GraphBackend` (not a bare writer); stay the read/write/search facade over capabilities, readers, ranking, synthesis. |
| Readers | `application/readers/`, `application/services/read_orchestrator.py` | Keep as the only read trunk for agent-visible evidence families. |
| Mutation adapter | `adapters/outbound/graph/neo4j_writer.py` | Becomes the Neo4j `GraphMutationPort` impl; the port itself moves to `domain/ports/graph/`. |
| Backend base | new `adapters/outbound/graph/claim_derived.py` | `ClaimDerivedBackend`: default inspection/analytics/snapshot over `ClaimQueryPort` (semantic is mandatory and vector-backed, not derived). |
| Conformance suite | new `tests/graph/conformance/` | One parametrized suite every `GraphBackend` must pass. |
| Pot model/resolution | `domain/ports/pot_resolution.py`, `domain/ports/pot_source_listing.py`, `app/modules/context_graph/` | Split local Pot Management from managed tenancy while preserving the pot boundary. |
| Scanners | `domain/ports/config_scanner.py`, `application/use_cases/scan_working_tree.py`, `adapters/outbound/scanners/` | Reuse for local repo scans; add scanners here, not in the CLI body. |
| Structured records | `domain/context_records.py`, `domain/ontology.py` | Primary local write path for durable agent memory. |
| Skill install | `adapters/inbound/cli/agent_installer.py`, `adapters/inbound/cli/templates/{agent_bundle,claude_bundle}/` | Generalize into the Skill Manager: `agent_installer` becomes the `default`/`codex`/`claude` agent targets; templates become first-party catalog entries. |
| Skill ports | new `domain/ports/skills/` | `SkillCatalogPort`, `AgentTargetPort`, `SkillSyncPort`. |
| CLI / HTTP | `adapters/inbound/{cli,http}/`, `bootstrap/standalone_container.py` | Reuse; default CLI commands to the local daemon; split explicit `cloud` commands; add `potpie skills …`. |
| Managed adapter | `app/modules/context_graph/` | Keep cloud-specific API, DB, user, source, worker, and skill-sync concerns here. |

Main missing local pieces: `bootstrap/local_container.py`, daemon lifecycle
commands, local Pot Management storage and migrations, the `embedded`
`GraphBackend`, the `ClaimDerivedBackend` base, the conformance suite, graph
inspection/analytics surfaces, the Skill Manager (catalog client, agent-target
generalization, `domain/ports/skills/`), and export/cloud-sync clients.

## Setup And Daemon Lifecycle

```bash
potpie setup --repo . --agent claude --scan
potpie daemon status
potpie doctor
```

`potpie setup` owns the normal daemon installation/startup flow. A user should
not need to run `potpie daemon install` or `potpie daemon start` before Potpie is
usable. Daemon subcommands remain available for service-manager integration,
debugging, and recovery.

Setup also owns initial pot readiness. On first run it creates a local pot named
`default`, marks it active, and registers the requested repo source into that
pot. `--pot <name>` is an override for naming/using a different initial pot, not
a required onboarding step.

Target service managers: macOS `launchd` user agent; Linux `systemd --user`
with foreground fallback; Windows Task Scheduler or a service wrapper. Every
local command checks daemon health; if missing or stopped the CLI can
install/start it when safe; if unhealthy it shows the log path and a restart
path.

## Implementation Order

1. **Graph port package.** Create `domain/ports/graph/`. Move `GraphMutationPort`
   out of `neo4j_writer.py`; move `ClaimQueryPort` in. Add `SemanticSearchPort`
   and the three derivable port protocols. (Do this before a second adapter
   imports the old path.)
2. **Backend bundle + base.** Add `GraphBackend`, `GraphCapability`, and the
   `ClaimDerivedBackend` base. Add the local embedder behind the semantic port.
   Change `ContextGraphService` to take a `GraphBackend`.
3. **Conformance suite.** Promote the in-memory reader to a full `in_memory`
   `GraphBackend` (incl. brute-force vector search) and make it pass the suite.
4. **Embedded backend.** Implement the SQLite `embedded` profile *with vector
   search* (`sqlite-vec` + bundled embedder); pass the suite. Spike
   infra-traversal / timeline ordering before committing.
5. **Setup-aware local daemon skeleton.** Foreground daemon, health/status, local
   auth, default active pot registry, service-manager install/start,
   `potpie setup`, and `potpie daemon` recovery/admin commands.
6. **Local Pot Management.** Local state DB, migrations, pot CRUD, source
   registry, status, analytics summaries.
7. **CLI local default.** Route resolve/search/record/status through the
   daemon; keep cloud commands explicit. Wire `context_status` to report backend
   name, capabilities, and semantic/embedder readiness.
8. **CLI contract completion.** Implement the command groups and `--json`
   schemas in [`oss-self-serve-flow.md`](./oss-self-serve-flow.md): `pot`,
   `source`, `ingest`, `graph`, `backend`, and `skills`. Keep every command
   routed through daemon services rather than physical stores.
9. **Skill Manager.** Add `domain/ports/skills/`; generalize `agent_installer`
   into `default`/`codex`/`claude` agent targets; make the bundled templates
   first-party catalog entries; add `potpie skills …`; feed the `skills` nudge
   block into `context_status`. Then add the OSS catalog client and (managed)
   `SkillSyncPort`.
10. **Snapshots and export/import.** Portable pot snapshot for backup, migration,
   and cloud push — designed now so local and managed schemas cannot drift.
11. **Optional store profiles & managed event ledger.** Neo4j/pgvector/Chroma and
    hosted embedders behind the same ports; separate webhook capture from graph
    ingestion.

## Non-Negotiables

- Local graph use works without Potpie cloud auth.
- Local CLI mode and cloud mode are visibly different.
- The daemon shell hosts services; it does not absorb their business logic.
- Pot Management owns control-plane concerns; Graph Service owns the data plane;
  the Skill Manager owns agent-skill provisioning.
- The agent-facing four tools stay stable. Skills are installed by the Skill
  Manager (CLI), never as a fifth tool; the skill nudge is advisory only, through
  `context_status`.
- Skill content is agent config, not project facts; it never goes in the graph.
- The graph stores compact claims and source refs, not full source payloads.
- The canonical claim store is the only source of truth; every other store is a
  rebuildable projection.
- A write is one transaction against one mutation port; store topology is an
  adapter secret.
- Vector semantic search is a required backend capability, not an option;
  embeddings run on a bundled local model so it works with no cloud key. There is
  no lexical-only shippable mode.
- Physical stores are bindings behind capability ports; the default profile is
  embedded and single-store.
- A backend is valid only when it passes the conformance suite.
- Managed-only concerns stay outside the local daemon.
