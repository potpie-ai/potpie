---
id: ADR-0007
title: Replace Both Daemon Stacks Through A Same-PR Migration
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-20T04:04:32+05:30
---

# ADR-0007: Replace Both Daemon Stacks Through A Same-PR Migration

## Context

The accepted runtime contracts require one typed daemon architecture, one
runtime entry path, one canonical discovery writer, removal of Context Core as
an independent boundary, removal of HostShell and reflection RPC, and explicit
retirement criteria for temporary compatibility adapters. They deliberately do
not select which current daemon implementation to evolve or the order in which
callers and obsolete paths migrate.

The shipped daemon launches `potpie.daemon.main`, builds HostShell, mirrors its
object graph through reflective `surface` and `method` requests, transports
Python class identity, serializes work through a process-wide lock, and writes
overlapping discovery records. A second candidate runtime contains an explicit
operation-registry experiment and some lifecycle mechanics, but it is not the
launched product runtime, registers no product operations by default, passes an
opaque HostShell dependency bag, and carries speculative plugin, managed-service,
transport, and discovery architecture.

The repository also publishes Context Core separately and contains callers that
reach HostShell through CLI host acquisition, bootstrap wiring, daemon lifecycle
commands, tests, and package metadata. A migration needs intermediate seams so
reviewable commits remain coherent, but those seams must not become a third
architecture or survive merge.

## Decision

Potpie will build one new canonical runtime by extracting only individually
reviewed minimal primitives and behavioral evidence from the existing daemon
stacks. Neither the active reflective stack nor the incomplete candidate stack
is the target architecture, and extraction does not preserve their module or API
identity or establish implementation conformance.

The migration will use branch-local, monotonically shrinking compatibility
shims within the same pull request. Every shim must have an owner, an
introduction commit, an explicit deletion condition, and a deletion commit in
the same pull request. No shim may be released or remain when the pull request
merges.

This decision resolves `OQ-DAEMON-IMPLEMENTATION-001` and `OQ-COMPAT-001`.

## Reuse Classification

### Minimal primitives eligible for extraction

The following are candidates for extraction only after focused review and
independent tests place them behind the accepted ownership boundary:

- External child-process creation and environment projection mechanics.
- Secure token comparison and private-file mechanics.
- The finite typed-operation catalog concept and duplicate-name rejection.
- Partial-start rollback, readiness waiting, and ordered teardown behavior.
- Characterization and integration tests that preserve supported behavior and
  failure evidence.

Eligibility does not approve the current types, dependencies, module layout,
transport, lifecycle model, or discovery representation. In particular, opaque
`Any` dependency bags, stringly error contracts, and generic registries are not
approved primitives.

### Behavioral evidence only

The active reflective stack supplies evidence about shipped CLI callers,
daemon lifecycle commands, setup behavior, domain results, failure translation,
and process entry points. The candidate stack supplies evidence about explicit
operation registration, partial-start cleanup, transport readiness, and managed
service failure cases.

Tests and observed behavior can inform replacement coverage without conferring
authority on either implementation architecture.

### Rejected reuse

The replacement will not reuse or preserve as target architecture:

- HostShell or a renamed HostShell-shaped façade.
- RemoteHostShell, RemoteSurface, or arbitrary member forwarding.
- Reflective RPC endpoints or `surface`, `method`, `args`, and `kwargs` dispatch.
- Cross-wire Python module or class identity codecs.
- A process-wide operation mutex.
- Either existing discovery schema or competing discovery writers.
- DaemonRuntime, generic component or transport plugin registries, opaque
  dependency containers, or the candidate managed-service architecture as a
  whole.
- A router, translator, or selector that can keep two daemon runtimes active.

## Migration Sequence

Implementation commits that depend on another deferred decision must wait until
that decision is separately accepted. Within that constraint, this pull request
will proceed in this order:

1. Characterize accepted CLI and daemon entry points, human journeys, direct
   Context Engine domain behavior, package installation, daemon lifecycle, and
   hosted operation behavior. Characterize the supported behavior of
   `CONTEXT_ENGINE_HOST_MODE`, `CONTEXT_ENGINE_BACKEND`, `GRAPH_DB_BACKEND`, and
   persistent backend and data visibility before deleting HostShell or host
   wiring. Classify callers as accepted product surfaces, repository-internal
   imports, or potentially external but uncontracted Python imports.
2. Add the replacement Context Engine composition, external controller,
   canonical foreground runtime, and typed-client boundaries without selecting
   them as the shipped path.
3. Migrate repository Context Core imports into the Context Engine distribution.
   The old namespace may remain only as a branch-local import shim while the
   repository caller allowlist shrinks.
4. Migrate daemon lifecycle commands from `get_host().daemon` to the external
   controller before migrating hosted domain operations.
5. Migrate hosted CLI operation families from `get_host()` and RemoteHostShell
   to finite typed-client operations. HostShell-shaped seams remain available
   only to an explicit, monotonically shrinking allowlist of unmigrated callers.
6. Migrate direct compatible hosts and tests from `build_host_shell` to the
   separately accepted public Context Engine construction boundary.
7. After replacement runtime, controller, client, and hosted end-to-end evidence
   pass, switch the `potpie-daemon` console entry and launcher exactly once to
   the canonical foreground runtime. Do not route between old and new runtimes
   or translate reflective RPC to the typed protocol.
8. After extracting any approved primitives, remove the dormant candidate stack
   first because it has no shipped launcher or repository production call path.
   Retain the active reflective stack only long enough to complete caller
   migration.
9. Before merge, remove the active reflective server and client, Python-class
   codecs, HostShell and host wiring, the separate Context Core package,
   duplicate discovery writers and readers, the dormant candidate runtime, every
   migration allowlist, and every temporary shim.
10. Replace characterization checks that freeze reflective routes with positive
    checks for the final entry points and negative checks for every forbidden
    architecture surface.

Intermediate commits may contain both old and replacement code only to keep the
migration reviewable and testable. The final pull-request state, not an earlier
commit, determines whether the deletion requirement is satisfied.

## Compatibility Scope

The migration preserves accepted Potpie CLI and daemon entry points and the
accepted context-domain behavior while repository callers move. Existing
command names and human presentation may remain stable while their internal
dependency direction changes.

Same-pull-request shims do not preserve source compatibility for arbitrary
external imports of `potpie_context_core`, HostShell, or other uncontracted
Python implementation paths because the shims are deleted before merge. If an
external Python import is later declared a supported public API, its deprecation
or breaking-release policy requires a separate accepted compatibility decision.

The migration must not change exact machine JSON fields, numeric exit mappings,
or other deferred observable contracts under the guise of internal caller
compatibility.

## Temporary Shim Rules

Only these narrowly bounded old-path seams are eligible during migration:

- `get_host` and `set_host` for an explicit shrinking allowlist of unmigrated CLI
  modules and tests.
- HostShell and `build_host_shell` only for unmigrated old-path callers.
- RemoteHostShell, RemoteSurface, the reflective client, and `/rpc` and `/attr`
  only as one paired old path until every hosted CLI family uses the typed client.
- The existing legacy discovery reader and duplicate writers only as frozen
  parts of the active old stack while branch commits may still start it; no new
  or third discovery format, record, reader, or writer may be introduced.
- The old Context Core namespace only as a branch-local import-forwarding shim
  while repository imports and dependency metadata migrate.

No new production caller may adopt one of these seams. A generic compatibility
gateway, HostShell-shaped typed client, dual-runtime selector, or old-to-new
protocol translator is prohibited.

## Merge Gates

The pull request must not merge until all of the following are true:

- No production or test caller imports or references HostShell, RemoteHostShell,
  RemoteSurface, the reflective daemon client, `build_host_shell`, or
  `potpie_context_core`, except an explicitly permitted historical observation.
- `get_host`, `set_host`, their shared host state, and every production or test
  caller of those seams are deleted, and no generic CLI host-acquisition façade
  replaces them.
- No reflective endpoint, arbitrary member dispatch, generic attribute
  forwarding, or cross-wire Python class codec remains.
- Exactly one daemon console entry, launcher target, runtime entry, external
  controller, typed client path, and canonical discovery writer remain.
- No dormant alternate runtime, selectable registry, shim allowlist, or duplicate
  discovery format remains.
- Root and Context Engine wheels and source distributions build and install in
  isolation without a Context Core package dependency.
- Accepted CLI command and daemon entry-point behavior remains covered.
- Supported behavior for `CONTEXT_ENGINE_HOST_MODE`, `CONTEXT_ENGINE_BACKEND`,
  `GRAPH_DB_BACKEND`, and persistent backend and data visibility is preserved or
  changed only through a separately accepted compatibility decision.
- Direct Context Engine and replacement hosted CLI journeys cover equivalent
  context-domain semantics, including a real replacement-daemon journey rather
  than only the existing in-process smoke path.
- Static negative checks reject every forbidden architecture surface.

## Deferred Decisions Preserved

This decision does not select or define:

- Exact public Context Engine operation groups or synchronous/asynchronous API.
- Resource Manager reuse, eviction, or idle lifetime.
- Exact daemon-controller API or supervisor integration.
- Daemon transport, wire protocol, handshake fields, or discovery fields.
- Cancellation, timeout, disconnect, idempotency, retry, or concurrency policy.
- Destructive-intent representation.
- Exact machine JSON fields or complete numeric nonzero exit-code mapping.
- The exact replacement operation catalog.

## Authority And Sources

> authority [active]: user:dsantra
> observation [active]: code:potpie/daemon/main.py@047cbe067c9c726e7e14f066675453372d8a8406
> observation [active]: code:potpie/daemon/client.py@047cbe067c9c726e7e14f066675453372d8a8406
> observation [active]: code:potpie/daemon/rpc.py@047cbe067c9c726e7e14f066675453372d8a8406
> observation [active]: code:potpie/daemon/runtime/shell.py@047cbe067c9c726e7e14f066675453372d8a8406
> observation [active]: code:potpie/daemon/runtime/__main__.py@047cbe067c9c726e7e14f066675453372d8a8406
> observation [active]: code:potpie/cli/commands/_common.py@047cbe067c9c726e7e14f066675453372d8a8406
> observation [active]: code:pyproject.toml@047cbe067c9c726e7e14f066675453372d8a8406

## Consequences

- The migration can remain bisectable without shipping a permanent compatibility
  layer or parallel daemon architecture.
- The candidate runtime is removed before the active reflective path, but neither
  is treated as the replacement design.
- The active implementation remains available only long enough to characterize
  and migrate supported callers.
- Potentially external Python import compatibility is not overstated; it requires
  a separate decision if it becomes a supported product obligation.
- The implementation pull request is larger than a clean break, but every
  temporary seam has a mechanically checkable deletion end state.

## Alternatives Considered

### Evolve the active reflective daemon

This minimizes immediate caller movement but preserves HostShell mirroring,
reflection, class-coupled serialization, global serialization, and competing
discovery through the center of the redesign.

### Complete the candidate runtime

This starts from an explicit operation experiment but blesses speculative
plugin, managed-service, transport, service-locator, lifecycle, and discovery
architecture that has never hosted the product operation surface.

### Ship one deprecation release

This offers stronger source compatibility to external Python importers but
ships the duplicate boundaries and daemon debt that the accepted migration end
state requires this pull request to remove.

### Perform a coordinated breaking change without shims

This produces the smallest final concept set but makes intermediate commits
harder to keep green and increases caller-migration risk.

### Preserve both daemon stacks behind a wrapper

This creates a third architecture and defeats the accepted deletion end state.

## Contract And Impact Review

No accepted behavior changes. The decision selects an implementation sequence
already permitted by the accepted contracts and therefore requires no contract
revision or spec-change record.

| Artifact or area | Required change after acceptance | No-change reason |
|---|---|---|
| SPEC-PRODUCT | — | Observable Context Engine behavior remains governed by PROD-005. |
| SPEC-SYSTEM | — | Temporary migration mechanisms and implementation gaps are already permitted and constrained. |
| SPEC-CONTEXT-ENGINE | — | Context Core, HostShell, and temporary import-shim end states are already binding. |
| SPEC-POTPIE-RESOURCE-MANAGER | — | The lease and resource ownership boundary is unchanged. |
| SPEC-DAEMON | — | Single-stack deletion, reflection removal, discovery convergence, and shim retirement are already binding. |
| SPEC-CLI | — | Command dependency migration is already permitted; JSON and exit details remain deferred. |
| SPEC-PROCESS | — | This is an additive decision and question resolution, not a contract mutation. |
| ADR-0001 through ADR-0006 | — | Existing accepted decisions remain immutable. |
| SPEC-CHANGE-0001 through SPEC-CHANGE-0008 | — | No behavior operation or contract revision occurs. |
| Existing conformance | — | No conformance records exist and this decision makes no implementation claim. |

The derived index gains this decision. After acceptance only,
`OQ-DAEMON-IMPLEMENTATION-001` and `OQ-COMPAT-001` resolve to this ADR while all
other questions remain deferred.

## Acceptance

Accepted by `user:dsantra` at `2026-08-20T04:04:32+05:30`. This acceptance
resolves `OQ-DAEMON-IMPLEMENTATION-001` and `OQ-COMPAT-001` while preserving
all other deferred questions. It does not claim implementation or verification
conformance.
