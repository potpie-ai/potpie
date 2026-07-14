---
id: SPEC-PACKAGE-BOUNDARY
title: Potpie / Context Engine Package Boundary
kind: module-spec
status: accepted
version: 2.0.0
owners:
  - Potpie Engineering
depends_on:
  - SPEC-PROCESS
  - SPEC-GLOSSARY
  - SPEC-PRODUCT
  - SPEC-SYSTEM
related_specs: []
related_decisions:
  - ADR-0002
affects:
  - SPEC-PRODUCT
  - SPEC-SYSTEM
open_questions: []
verification:
  code_status: implementation_claimed
  verified_commit: null
  verified_at: null
  verified_by: null
  behavior_scope: []
  evidence: []
  cross_spec_status: passed
  cross_spec_checked_against:
    - SPEC-GLOSSARY
    - SPEC-PRODUCT
    - SPEC-SYSTEM
    - ADR-0002
  drift_status: stale
---

# Potpie / Context Engine Package Boundary

## Purpose

This spec defines the accepted ownership and interaction boundary between the
root `potpie` product distribution and the standalone `potpie-context-engine`
library distribution.

It is the normative source for the migration tracked in
[package-boundary-migration-plan.md](../../docs/context-graph/package-boundary-migration-plan.md).
Version 1.0.0 of the boundary was implemented and verified at `f435fb4`; its
detailed evidence remains in
[the 1.0.0 verification record](../verification/package-boundary-1.0.0.md).
Version 2.0.0 changes the public graph-store command vocabulary and awaits
commit-scoped verification.

Decision: [ADR-0002: Separate Product Runtime from Context Engine](../decisions/ADR-0002-potpie-context-engine-boundary.md).

## Ownership And Boundaries

### Implementation evidence

- Root [pyproject.toml](../../pyproject.toml) owns `potpie`, `potpie-daemon`, and
  `potpie-mcp` and selects `potpie-context-engine[embedded]==0.2.0`.
- Engine [pyproject.toml](../../potpie/context-engine/pyproject.toml) exports no
  process entrypoint, has a Pydantic-only core, and declares nine explicit
  capability extras.
- [potpie/runtime/composition.py](../../potpie/runtime/composition.py) constructs
  `PotpieRuntime`, `LocalEngineClient`, or `DaemonEngineClient`.
- [potpie/daemon/rpc.py](../../potpie/daemon/rpc.py) declares a protocol-v1,
  allowlisted `engine.*` registry with typed request/result adapters.
- [potpie/mcp/server.py](../../potpie/mcp/server.py) owns exactly the four public
  tools and routes context work through `runtime.engine.context`.
- The standalone engine is namespaced under `potpie_context_engine`, imports no
  root `potpie`, and accepts caller-owned configuration and dependencies.

### Accepted ownership

| Concern | Root `potpie` | `potpie-context-engine` |
|---|---|---|
| Product executables | `potpie`, `potpie-daemon`, `potpie-mcp` | None |
| Presentation | Typer, Rich, prompts, setup wizard, human/JSON rendering | None |
| Product composition | `PotpieRuntime`, settings, paths, runtime mode | None |
| Product account auth | Login, logout, whoami, credential persistence | Request actor contract only |
| Provider integrations | OAuth, keyring, account/resource selection | Credential-agnostic connector adapters |
| Skills and installation | Catalog, bundled resources, targets, drift, install/update/remove | None |
| Setup | Product workflow and reporting | Pure engine provision inspect/apply |
| Status | Runtime/daemon/skills/setup enrichment and recommendations | Pure engine readiness facts |
| Daemon | Process, lifecycle, transport, RPC, local UI | No process or lifecycle ownership |
| MCP | Process, four tool declarations, product status | Context operations called by root |
| Telemetry | Product Sentry/PostHog and build defaults | Generic observability ports and optional OTel |
| Context and graph | Product command exposure | Domain, use cases, ports, stores, backends |
| Persistence | Product path resolution | State under an explicitly supplied data directory |
| HTTP/webhooks | Product hosting when applicable | Optional injected engine app factories |

### Landed root structure

```text
potpie/
├── cli/
│   ├── main.py
│   ├── commands/
│   ├── output/
│   └── ui/
├── runtime/
│   ├── composition.py
│   ├── settings.py
│   ├── paths.py
│   ├── modes.py
│   └── telemetry/
├── auth/
│   ├── account/
│   ├── integrations/
│   └── credentials/
├── setup/
├── skills/
│   ├── resources/
│   └── service.py
├── install/
├── daemon/
│   ├── client.py
│   ├── rpc/
│   ├── process/
│   └── http/ui/
└── mcp/
```

### Landed engine structure

```text
potpie/context-engine/
├── pyproject.toml
├── src/potpie_context_engine/
│   ├── __init__.py
│   ├── engine.py
│   ├── config.py
│   ├── dependencies.py
│   ├── contracts/
│   ├── domain/
│   ├── application/
│   ├── ports/
│   ├── adapters/
│   │   ├── inbound/http/
│   │   └── outbound/
│   ├── composition/
│   └── resources/
├── tests/
└── benchmarks/
```

## Scope And Non-Goals

### In scope

- Python distribution and import ownership.
- Root product composition and engine-client boundary.
- Local and daemon execution parity.
- Versioned daemon RPC.
- Product auth, config, skills, setup, status, telemetry, daemon, CLI, and MCP
  ownership.
- Engine configuration, context capabilities, ports, adapters, and optional
  HTTP factories.
- CLI hierarchy, JSON envelope, exit codes, and clean-break migration.
- Package metadata, extras, entrypoints, wheel contents, tests, and docs.

### Non-goals

- Reorganizing the `potpie/integrations`, `potpie/parsing`, or `potpie/sandbox`
  workspace packages.
- Changing graph, pot, ledger, credential, or daemon-state storage formats.
- Redesigning context resolve/search/record semantics beyond typed boundary DTOs.
- Expanding MCP beyond four tools.
- Providing import, command, facade, or RPC compatibility aliases.
- Defining a hosted deployment or hosted-account architecture.

## Terminology

Use [SPEC-GLOSSARY](../glossary.md), especially Product distribution, Context
engine, Product service, Engine operation, PotpieRuntime, EngineClient,
Product settings, and Engine configuration.

## Actors And Permissions

### Product user

Installs root `potpie`, runs CLI commands, authenticates accounts and providers,
selects product settings, and manages the daemon and skills.

### Coding agent

Uses CLI or the four MCP tools within the product user's configured scope. It
receives only the context capabilities and identity the product runtime grants.

### Product operator

Starts, stops, diagnoses, and inspects the root-owned daemon and supporting local
services.

### Engine embedder

Imports `potpie_context_engine`, supplies explicit configuration and
dependencies, and invokes typed engine operations without root Potpie.

### Daemon transport peer

A local root-Potpie client authenticated through the daemon's existing local
transport protection. It may call only registered `engine.*` RPC methods.

## Normative Requirements

PKG-OWN-001: Root `potpie` MUST own every user-facing product process and product
service, including CLI, MCP, daemon lifecycle, product UI, auth, config, setup,
skills, installation, product status, and product telemetry.

PKG-OWN-002: `potpie-context-engine` MUST NOT import root `potpie`, infer product
paths, access product credentials, or own product processes.

PKG-API-001: The engine's only supported Python import surface MUST be
`potpie_context_engine` and explicitly exported modules under
`potpie_context_engine.contracts`.

PKG-RUNTIME-001: Root consumers MUST obtain a `PotpieRuntime`; every engine
operation MUST be visible under `runtime.engine.*`, while product services remain
sibling runtime capabilities.

PKG-MODE-001: Root Potpie MUST default to daemon mode, MUST allow explicit
`daemon` or `in-process` selection, and MUST NOT silently fall back to in-process
execution when the configured daemon is unavailable.

PKG-RPC-001: Daemon RPC MUST be versioned, typed, allowlisted, limited to
`engine.*` methods, and independent of Python module/class-path serialization.

PKG-CONFIG-001: Root product settings and engine configuration MUST be distinct
types; a persistent standalone engine MUST require an explicit data directory,
and an in-memory engine MUST NOT read or write the user's home directory.

PKG-AUTH-001: Product account and provider authentication MUST be root-owned;
the engine MAY receive a typed request actor but MUST NOT read product keyrings,
sessions, or provider selections.

PKG-SKILL-001: Skill catalog, bundled resources, install targets, drift state,
and command validation MUST be root-owned; skill installation MUST NOT inspect
Typer dynamically at runtime.

PKG-SETUP-001: Root Potpie MUST own the product setup workflow; the engine MUST
expose only typed provision inspection and application operations without
rendering UI, starting product processes, installing skills, or emitting product
analytics.

PKG-STATUS-001: The engine MUST return a pure `EngineStatusReport`; root Potpie
MUST compose the flat public status result used by CLI and MCP.

PKG-MCP-001: Root Potpie MUST own `potpie-mcp` and MUST expose exactly
`context_resolve`, `context_search`, `context_record`, and `context_status`.

PKG-CLI-001: Root Potpie MUST expose the workflow-first CLI hierarchy defined in
this spec and MUST reject removed legacy command paths as unknown.

PKG-CLI-002: Every `--json` command MUST write exactly one versioned success or
failure envelope to stdout, MUST keep logs and diagnostics on stderr, and MUST
not prompt interactively.

PKG-QUEUE-001: Context-graph job submission MUST use an injected engine queue
port; engine construction MUST NOT mutate `sys.path` or import a legacy product
application queue.

PKG-OBS-001: Product Sentry/PostHog telemetry MUST remain in root Potpie; the
engine MAY expose generic observability ports and optional OpenTelemetry
adapters but MUST NOT emit product funnel events.

PKG-DIST-001: Root `potpie` MUST own all three console scripts; the engine wheel
MUST contain only the `potpie_context_engine` runtime namespace and engine-owned
resources.

PKG-VERIFY-001: The migration MUST pass isolated-wheel, import-boundary,
local/daemon parity, CLI contract, MCP contract, persistent-data compatibility,
and cross-spec verification before this spec becomes `verified`.

## Data And State Model

### ProductSettings

Root-owned state covering:

- product home and existing persistent-data locations;
- runtime mode;
- account and provider configuration;
- active backend and other persisted product choices;
- UI, installation, skills, daemon, and product telemetry configuration.

Runtime-mode precedence is:

1. global `--runtime` CLI option;
2. `POTPIE_RUNTIME_MODE`;
3. persisted product setting;
4. product default `daemon`.

### EngineConfig

Library configuration has two explicit construction modes:

```python
EngineConfig.persistent(data_dir=path, ...)
EngineConfig.in_memory(...)
```

The persistent form has an explicit path. The in-memory form has no persistent
path and performs no user-home discovery.

### EngineStatusReport

```text
schema_version
pot_id
pot_name
backend
backend_ready
storage_ready
ingestion_ready
source_count
last_ingestion_at
degraded_reasons
```

### Flat product status

```text
schema_version
ready
runtime_mode
daemon_state
pot_id
pot_name
backend
backend_ready
storage_ready
ingestion_ready
source_count
last_ingestion_at
skills_state
setup_state
issues
recommended_next_action
```

`daemon_state` is `running`, `stopped`, `unreachable`, or `not_applicable`.
Issues carry a stable code, component, message, and severity. A recommended
action carries a command and reason or is null.

### CLI JSON envelope

Success:

```json
{
  "ok": true,
  "data": {},
  "meta": {
    "schema_version": "1",
    "command": "graph.read",
    "runtime_mode": "daemon",
    "request_id": "uuid-or-null"
  }
}
```

Failure:

```json
{
  "ok": false,
  "error": {
    "code": "RUNTIME_DAEMON_UNAVAILABLE",
    "message": "The Potpie daemon is not reachable.",
    "details": {},
    "retryable": true,
    "recommended_next_action": {
      "command": "potpie daemon start",
      "reason": "The configured runtime mode is daemon."
    }
  },
  "meta": {
    "schema_version": "1",
    "command": "status",
    "runtime_mode": "daemon",
    "request_id": null
  }
}
```

List data uses `{ "items": [], "count": 0, "next_cursor": null }`, not a bare
array.

## State Machines

### Runtime selection

```text
resolve ProductSettings
        |
        v
 select daemon ---------------------- select in-process
        |                                      |
        v                                      v
 health/discovery succeeds              construct local engine
        |                                      |
        v                                      v
 DaemonEngineClient                       LocalEngineClient

daemon unavailable -> typed failure + `potpie daemon start`
```

There is no transition from daemon failure to implicit local construction.

### Product setup

```text
resolve settings and paths
  -> product/account/integration checks
  -> reconcile or start configured daemon
  -> obtain runtime.engine
  -> inspect/apply engine provisioning
  -> install/update skills
  -> compose product setup report
```

### Spec implementation

```text
accepted 1.0.0
  -> implemented by commits 2-13
  -> verified at f435fb4
```

## Interfaces

### Engine public API

```python
from potpie_context_engine import (
    ContextEngine,
    EngineClient,
    EngineConfig,
    EngineDependencies,
    create_engine,
)
from potpie_context_engine.contracts import ...
```

`EngineClient` is asynchronous and exposes:

```text
engine.context.resolve/search/record/status
engine.pots.*
engine.sources.*
engine.graph.*
engine.ledger.*
engine.timeline.*
engine.provision.inspect/apply
engine.aclose
```

Every operation accepts one typed request DTO and returns one typed result DTO.
Engine DTOs do not include product UI, daemon lifecycle, credentials, skills,
installation, or product-telemetry state.

### Root runtime

```python
PotpieRuntime(
    settings=ProductSettings,
    engine=EngineClient,
    auth=AccountAuthService,
    integrations=IntegrationAuthService,
    config=ProductConfigService,
    skills=SkillService,
    installer=Installer,
    setup=ProductSetupService,
    status=ProductStatusService,
    daemon=DaemonService,
    telemetry=ProductTelemetry,
)
```

CLI and MCP obtain the runtime from `get_runtime()`. Engine-backed command
handlers call `runtime.engine.*`; product handlers call the relevant sibling
service.

### Daemon RPC

Request:

```json
{
  "protocol_version": "1",
  "request_id": "uuid",
  "method": "engine.graph.read",
  "params": {}
}
```

Success:

```json
{
  "protocol_version": "1",
  "request_id": "uuid",
  "ok": true,
  "result": {}
}
```

Failure:

```json
{
  "protocol_version": "1",
  "request_id": "uuid",
  "ok": false,
  "error": {
    "code": "STABLE_ERROR_CODE",
    "message": "Human-readable message",
    "details": {},
    "retryable": false
  }
}
```

An explicit registry maps each method to its request model, result model, and
handler. `/healthz` is a root-owned transport endpoint. `/rpc` contains no auth,
setup, skills, config, installation, or daemon-lifecycle method.

### CLI command hierarchy

```text
potpie resolve
potpie search
potpie record
potpie status

potpie setup
potpie doctor
potpie login
potpie logout
potpie whoami
potpie ui

potpie pot list|info|create|use|linked|rename|reset|archive|default
potpie source add|list|status|remove
potpie timeline recent

potpie graph catalog|read|search-entities|mutation-template|nudge|status
potpie graph propose|commit|history|inspect|export|import|repair
potpie graph inbox ...
potpie graph quality ...
potpie graph bulk ...
potpie graph store list|status|use|doctor

potpie ledger status|query|use|disconnect|pull|sources

potpie integration list
potpie integration status [PROVIDER]
potpie integration github login|logout|list
potpie integration linear login|logout|list|select
potpie integration jira login|logout|list|select
potpie integration confluence login|logout|list|select

potpie skills list|install|update|remove|status|add

potpie daemon start|status|logs|restart|stop

potpie config list|get|set
potpie telemetry status|enable|disable
```

Command migration:

| Removed command | Landed replacement |
|---|---|
| `potpie use` | `potpie pot use` |
| `potpie github ...` | `potpie integration github ...` |
| `potpie linear ...` | `potpie integration linear ...` |
| `potpie jira ...` | `potpie integration jira ...` |
| `potpie confluence ...` | `potpie integration confluence ...` |
| `potpie auth ...` | account `login/logout/whoami` or `integration status` |
| `potpie backend ...` | `potpie graph store ...` |
| `potpie service ...` | removed; V1 has no daemon-managed supporting services |
| `potpie cloud ...` | removed; no replacement for unfinished sync operations |
| `potpie graph mutate ...` | `graph propose` then `graph commit` |
| `potpie graph describe ...` | `graph catalog --subgraph ...` |
| `potpie graph neighborhood ...` | a named `graph read` neighborhood view |

Provider resource enumeration is named `list`; deprecated provider aliases do
not remain registered.

### MCP

Root `potpie-mcp` exposes exactly:

- `context_resolve`
- `context_search`
- `context_record`
- `context_status`

The first three preserve their existing argument and result semantics while
routing through `runtime.engine.context`. Status routes through
`runtime.status` and returns the flat product status data without the CLI JSON
envelope.

## Invariants

PKG-API-001 and PKG-OWN-002 establish a one-way dependency: root Potpie may
depend on the public engine API, while the engine never depends on root Potpie.

PKG-RUNTIME-001 establishes that product and engine capabilities are siblings,
not flattened into a shell facade.

PKG-MODE-001 establishes that an unavailable daemon cannot change execution
mode or persistence behavior implicitly.

PKG-DIST-001 establishes that installing the engine alone cannot expose or
start a product process.

Persistent graph, pot, ledger, credential, and daemon-state formats remain
unchanged by this boundary migration.

## Consistency, Ordering, And Idempotency

- Engine DTO serialization is deterministic for a given DTO value and protocol
  version.
- An RPC response carries the same request ID as its request.
- Protocol validation and method allowlisting occur before handler dispatch.
- Product setup checks existing state before applying provisioning or skill
  installation so rerunning setup is safe.
- Proposal/commit graph semantics are unchanged; removing `graph mutate` does
  not introduce a second write path.
- Status reads do not mutate product or engine state.

## Failure Modes And Recovery

| Failure | Required result | Recovery |
|---|---|---|
| Configured daemon is stopped or unreachable | `RUNTIME_DAEMON_UNAVAILABLE`, exit `3`; no local fallback | `potpie daemon start` |
| CLI and daemon protocol versions differ | `RPC_PROTOCOL_MISMATCH`, exit `3`; no dual protocol | `potpie daemon restart` after upgrade alignment |
| RPC method is unknown | Reject before dispatch with stable protocol error | Upgrade caller or use a registered method |
| RPC DTO is malformed | Validation failure, exit `2` at CLI boundary | Correct request fields |
| Product auth is missing or denied | Root-owned auth error, exit `4`; no engine credential lookup | Run the appropriate login command |
| Product is configured but not ready | Flat status with issues and next action; exit `5` for health commands | Follow returned action |
| Standalone persistent engine lacks `data_dir` | Engine configuration validation error | Supply an explicit path |
| Optional engine adapter is unavailable | Typed capability/dependency error | Install the named engine extra or inject an adapter |
| Skill template references a removed command | Build/test validation failure; no invalid wheel release | Update template or command manifest |
| Unexpected internal failure | Stable JSON error and exit `70`; diagnostic on stderr | Run with verbose diagnostics and report defect |
| User interruption | Exit `130`; no unexpected-error telemetry | Retry intentionally |

CLI exit codes are:

| Code | Meaning |
|---:|---|
| `0` | Success or healthy status |
| `1` | Expected operation failure |
| `2` | Usage or validation failure |
| `3` | Runtime or daemon unavailable |
| `4` | Authentication or permission failure |
| `5` | Degraded, conflict, or not-ready state |
| `70` | Unexpected internal error |
| `130` | User interruption |

`ok` in a JSON envelope means the command executed successfully. A health
command can therefore return `ok: true`, report a degraded data state, and exit
`5`.

## Security And Privacy

- Root Potpie retains product credentials and provider sessions. They do not
  become engine configuration or RPC payloads unless an individual engine
  request requires a scoped, non-secret actor or connector capability.
- RPC remains local and protected by root daemon discovery/authentication.
- RPC allowlisting prevents arbitrary object traversal and method invocation.
- Removing module/class-path deserialization eliminates remote class loading as
  part of the wire contract.
- JSON errors and product status exclude secrets, tokens, credential paths,
  request bodies, graph payloads, and telemetry identifiers not required by the
  public schema.
- Engine in-memory construction does not inspect the user's filesystem.
- Product telemetry continues its existing privacy scrubbing and does not move
  into the engine.

## Observability And Auditability

- Root command and daemon boundaries attach command, runtime mode, request ID,
  result class, and stable error code to product diagnostics.
- Engine operations emit generic spans/metrics through injected observability
  ports without importing Sentry or PostHog product code.
- RPC carries request IDs end to end for correlation.
- Product setup and status expose actionable step/issue data without requiring
  telemetry access.
- Graph proposal/commit history remains the audit record for graph writes.

## Performance And Limits

- The boundary introduces no additional daemon round trip beyond one typed RPC
  call per engine operation.
- List DTOs remain bounded by each operation's existing limits and cursor rules.
- JSON mode emits one document and does not stream interleaved diagnostics.
- `EngineConfig.in_memory()` is intended for tests, embedding, and explicit
  local use; it does not promise cross-process durability.
- Backend-specific limits remain owned by engine adapter contracts and were not
  changed by this boundary.

## Compatibility, Migration, And Rollout

This is a clean break for imports, commands, facades, executables, and RPC.

- Engine imports move from generic top-level packages to
  `potpie_context_engine` public exports.
- Root Potpie becomes the only user-facing install target.
- A running old daemon is restarted; there is no protocol bridge.
- Removed CLI aliases fail as unknown and are documented in the migration table.
- Existing on-disk product data and paths remain compatible.
- `potpie-context-engine` moves from `0.1.0` to `0.2.0`.
- Root remains at the branch's planned `2.0.0` release.
- Root depends on explicit engine capabilities rather than `[all]`.
- The engine exposes explicit extras: `embedded`, `http`, `postgres`, `neo4j`,
  `embeddings`, `github`, `reconciliation`, `hatchet`, and `observability`.

Implementation was delivered as the fourteen separately verified commits in the
migration plan.

## Examples

### Normal product command in daemon mode

1. `potpie graph read ...` parses CLI options.
2. The handler obtains `PotpieRuntime`.
3. It calls `runtime.engine.graph.read(request)`.
4. `DaemonEngineClient` validates and sends `engine.graph.read` protocol v1.
5. The daemon invokes its in-process `ContextEngine`.
6. Root renders the typed result as Rich output or one JSON envelope.

### Explicit in-process embedding

```python
from potpie_context_engine import EngineConfig, create_engine

engine = create_engine(EngineConfig.in_memory())
```

This succeeds without root Potpie installed and without creating files under the
user's home directory.

### Unavailable daemon

`potpie status --json` in daemon mode receives a root-owned status/error result
that identifies the daemon as unavailable and recommends
`potpie daemon start`. It does not construct an in-process engine.

### Product setup

`potpie setup` can diagnose and start the root daemon before requesting engine
provisioning. Engine provisioning returns structured steps; root handles
prompts, rendering, skill installation, and product analytics.

### MCP status

`context_status` calls root `ProductStatusService`, which obtains pure engine
status and adds runtime, daemon, skills, setup, issues, and a recommended action.
The returned status data has the same fields as `potpie status --json`'s `data`.

### Removed graph write alias

`potpie graph mutate` is unknown. A caller creates a typed proposal with
`potpie graph propose` and explicitly applies it with `potpie graph commit`.

## Acceptance Criteria

| Behavior | Testable acceptance criterion |
|---|---|
| `PKG-OWN-001` | Root modules own every product service and process listed in the ownership table. |
| `PKG-OWN-002` | AST/import scan finds zero `potpie` imports under the engine source namespace. |
| `PKG-API-001` | Isolated engine wheel imports through `potpie_context_engine`; private top-level packages are absent. |
| `PKG-RUNTIME-001` | CLI/MCP architecture tests show no direct engine import and all engine calls pass through `runtime.engine`. |
| `PKG-MODE-001` | Daemon-unavailable tests fail explicitly; no in-process engine constructor is called. |
| `PKG-RPC-001` | Protocol, malformed DTO, unknown method, allowlist, and version-mismatch tests pass without class-path encoding. |
| `PKG-CONFIG-001` | Persistent construction rejects a missing path; in-memory construction writes nothing under a temporary home. |
| `PKG-AUTH-001` | Engine import and dependency scans find no keyring, product auth, or provider-session ownership. |
| `PKG-SKILL-001` | Root wheel installs skills from packaged resources and build tests reject invalid command snippets. |
| `PKG-SETUP-001` | Setup tests cover fresh, existing, daemon-unavailable, provision-failed, and missing-skill states. |
| `PKG-STATUS-001` | CLI data and MCP status have the same flat status fields; engine status contains no product fields. |
| `PKG-MCP-001` | MCP discovery reports exactly four approved tool names from the root distribution. |
| `PKG-CLI-001` | Exact command snapshot passes and every removed command is unknown. |
| `PKG-CLI-002` | JSON tests parse exactly one stdout value, see no prompts, and verify stderr separation and exit codes. |
| `PKG-QUEUE-001` | Search and tests find no `sys.path` mutation or legacy queue import; injected/default queue tests pass. |
| `PKG-OBS-001` | Engine dependency/import scan finds no Sentry/PostHog product modules; root telemetry tests pass. |
| `PKG-DIST-001` | Root wheel owns three scripts; engine wheel owns none and contains only its namespace/resources. |
| `PKG-VERIFY-001` | The final verification record names the commit, all behavior IDs, evidence, related specs, and no unknown gap. |

Additional release gates:

- Local and daemon results are equivalent after transport-only metadata is
  removed.
- Existing pot, graph, ledger, credential, and daemon-state fixtures load from
  their current paths.
- Both wheels and source distributions build and pass metadata validation in
  clean environments.
- Root and engine test suites run separately to avoid conflicting test package
  roots.
- No pre-existing telemetry worktree changes are included in migration commits.

## Cross-Spec Consistency

- `SPEC-GLOSSARY` defines all boundary terms used here.
- `SPEC-PRODUCT` identifies product users, engine embedders, goals, and non-goals
  without adding conflicting package behavior.
- `SPEC-SYSTEM` delegates initial package guarantees to this module spec.
- `ADR-0002` records the ownership, clean-break, MCP, runtime-mode, and interface
  decisions and links back to this spec.
- `spec/questions/open.md` records no unresolved package-boundary question.
- The version 2.0.0 `graph store` vocabulary was reviewed against the glossary,
  product spec, system spec, and ADR-0002. It changes only the public CLI label;
  the internal `GraphBackend` abstraction and persisted `backend` setting remain
  unchanged.

No cross-spec contradiction was found. Commit-scoped implementation verification
for version 2.0.0 remains pending.

## Open Questions

None. All package-boundary choices needed for implementation are resolved in
`ADR-0002` and represented by stable behavior IDs above.

## Rationale

The reusable engine should model context and graph behavior, while the product
should decide how users authenticate, configure, install, operate, observe, and
invoke that engine. A single `HostShell` obscures that distinction and makes the
engine wheel carry product concerns. A visible `runtime.engine.*` boundary makes
ownership reviewable in code and makes local/daemon parity a protocol property
rather than dynamic object proxy behavior.

## Implementation Notes

This section is non-normative; requirements remain in the behavior IDs above.

### Primarily mechanical moves

| Original location | Landed location |
|---|---|
| Engine domain, application, ports, reusable adapters | `src/potpie_context_engine/...` |
| `potpie/cli/auth` credential/OAuth/provider clients | `potpie/auth/...` |
| `potpie/cli/templates` | `potpie/skills/resources` |
| Engine skill resource loading | Root `potpie/skills` |
| Engine local installer helpers | Root `potpie/install` |
| Engine-owned playbooks and schemas | `potpie_context_engine/resources` |
| Engine benchmarks | Project-level `benchmarks/`, excluded from runtime wheel |

### Required rewrites

| Component | Replacement |
|---|---|
| `HostShell` | `ContextEngine` plus root `PotpieRuntime` |
| `RemoteHostShell` / `RemoteSurface` | Typed `DaemonEngineClient` |
| Engine bootstrap | Engine-only composition from config and injected dependencies |
| Root runtime | `create_runtime()` / `get_runtime()` and explicit sibling capabilities |
| Configuration | `ProductSettings` plus `EngineConfig` |
| Setup | Root workflow plus engine provision inspect/apply |
| Status | Pure engine report plus flat product enrichment |
| Authentication | Root account and integration services plus request-actor mapping |
| Skills | Root catalog/resources/targets plus static command-manifest validation |
| Queue wiring | Injected engine queue port and safe defaults |
| RPC | Versioned DTO registry and typed client |
| CLI | Parse, call runtime, render |
| MCP | Root server using the same runtime as CLI |
| Observability | Root product telemetry plus generic engine observability |
| Packaging | Root scripts plus namespaced engine wheel and explicit extras |

### Final removals

- `HostShell`, `RemoteHostShell`, `RemoteSurface`, and dynamic RPC dispatch.
- Generic top-level `domain`, `application`, `adapters`, `bootstrap`, and `host`
  import packages.
- Engine product auth/config/skills/install/daemon-lifecycle ownership.
- `bootstrap/potpie_path.py` and hard-coded legacy queue imports.
- Engine `potpie-mcp` entrypoint and runtime `mcp` dependency.
- Root `potpie-context-engine[all]` dependency and the engine `[all]` extra.
- Stale build exclusions, hook paths, command aliases, and documentation after
  their replacements are verified.
