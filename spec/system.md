---
id: SPEC-SYSTEM
title: Context Runtime System Contract
kind: system-spec
revision: 1
maturity: accepted
owners:
  - team:potpie
depends_on:
  - SPEC-GLOSSARY
  - SPEC-PRODUCT
change_ref: SPEC-CHANGE-0004
---

# Context Runtime System Contract

## Purpose

This contract defines the target boundary between the importable Context
Engine and the Potpie product host. It separates daemon lifecycle control from
hosted domain execution without freezing deferred transport or public API
details.

## Ownership And Boundaries

| Boundary | Owns | Excludes |
|---|---|---|
| Context Engine | Context-domain semantics, a thin use-case façade, typed engine outcomes, engine-owned state | Product selection, caller security policy, host-resource provisioning, daemon behavior, presentation |
| Potpie Resource Manager | Context resolution, authorization, host resources, engine composition, authorized leases | Product-operation dispatch, domain semantics, service lookup, daemon transport, presentation |
| Daemon controller | Foreground runtime creation and process observation | Readiness claims, product-domain operations, daemon protocol |
| Daemon runtime | Instance identity, lock, discovery, live readiness, typed handlers, protocol behavior | Process creation, domain semantics, terminal presentation |
| Typed daemon client | Finite typed operations and protocol translation | Dynamic service mirroring, human presentation, domain semantics |
| Potpie CLI | Command input, user intent, human and machine presentation, streams and exits | Context resolution, resource composition, daemon internals, domain semantics |

## Scope And Non-Goals

This revision covers the current Context Engine, Potpie resource-management,
daemon-hosting, and CLI boundaries. It leaves parsing unchanged and publishes
no extension, manifest, or external-host protocol.

The exact Python façade methods, controller API, transport, wire envelope,
cancellation protocol, retry keys, JSON fields, and numeric nonzero exit-code
mapping remain deferred.

## Actors And Permissions

| Actor | Boundary interaction |
|---|---|
| Library host | Supplies context identity, explicit dependencies, and dependency ownership to Context Engine |
| Human CLI user | Supplies command intent, selectors, credentials, and destructive confirmation where applicable |
| Automation caller | Supplies complete non-interactive input and explicit destructive intent where applicable |
| Daemon controller | Creates and observes the foreground runtime before a live endpoint exists |
| Typed daemon client | Authenticates live protocol requests and translates typed outcomes |
| Typed daemon handler | Acquires an authorized context lease and invokes one explicit engine operation |
| Resource Manager | Resolves context, applies authorization, and supplies the scoped lease |
| Context Engine | Executes context-domain behavior for its bound identity |

## Normative Requirements

SYS-001 [active]: The canonical Potpie-hosted domain call path MUST be CLI to typed daemon client to typed daemon operation handler to authorized context lease to an explicit operation on the lease's context-bound Context Engine.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  > decision [active]: decision:ADR-0005

SYS-002 [active]: Context Engine MUST NOT import or control Potpie daemon, CLI, context-selection, caller-security, process-lifecycle, or presentation concerns.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  > observation [active]: code:potpie/context-engine/src/potpie_context_engine/host/shell.py@a341978880b9d4c1b403831931279ccedf6184ae
  @ SYS-001

SYS-003 [active]: A host MUST supply Context Engine dependencies and context identity explicitly.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ SYS-002

SYS-004 [active]: Potpie MUST own context selection, authorization policy, host-managed resources, dependency composition, context-bound engine construction, and authorized context leases.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ SYS-003

SYS-005 [active]: The daemon runtime MUST own the authenticated live process boundary, instance identity, ownership lock, discovery, readiness, runtime lifecycle, and typed request handling.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

SYS-006 [active]: The CLI MUST own command parsing, user interaction, human and machine rendering, standard-stream policy, and process exit mapping.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

SYS-007 [active]: SelectionError, AuthenticationError, AuthorizationError, ResourceLifecycleError, DomainError, DependencyError, EngineLifecycleError, ProtocolTransportError, DaemonInternalError, and PresentationError MUST remain structurally distinguishable across boundaries.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ GLOSS-007

SYS-008 [active]: Caller authentication and operation authorization MUST occur outside Context Engine before protected domain execution.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ SYS-002
  @ SYS-004

SYS-009 [active]: Human confirmation, destructive-intent assertion, caller authentication, operation authorization, and destructive domain execution MUST remain distinct steps.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-008

SYS-010 [active]: Potpie hosting MUST NOT retarget an existing Context Engine instance to a different context identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ SYS-003

SYS-011 [active]: Work performed under this contract revision MUST NOT redesign parsing behavior.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006

SYS-012 [active]: A target contract that precedes implementation conformance MUST identify the current implementation gap.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001

SYS-013 [active]: The daemon controller MUST own the Potpie control request for foreground runtime creation and process observation outside the daemon runtime.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-005

SYS-014 [active]: Daemon process creation before a live endpoint exists MUST use the controller.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-001
  @ SYS-013

SYS-015 [active]: This contract revision MUST NOT publish a public extension, plugin-registration, or manifest contract.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006

SYS-016 [active]: This contract revision MUST NOT publish an external-host transport or deployment protocol.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006

SYS-017 [active]: A proposed or accepted target contract MUST NOT be represented as completed implementation or verified conformance without a matching conformance record.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ SYS-012

SYS-018 [active]: A host MUST retain ownership of every dependency supplied to Context Engine unless a typed construction contract explicitly transfers that dependency's ownership.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ SYS-003

SYS-019 [active]: A typed daemon context-domain operation handler MUST invoke the explicit Context Engine operation after acquiring an authorized context lease.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  > decision [active]: decision:ADR-0005
  @ SYS-001
  @ SYS-004

SYS-020 [active]: A CLI-produced destructive-intent assertion MUST remain untrusted until Potpie validates it against the authenticated actor, authorized operation, and resolved context identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-009

SYS-021 [active]: Readiness and live daemon operations after an endpoint exists MUST use the typed daemon client.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-001
  @ SYS-014

SYS-022 [active]: A typed daemon context-domain operation handler MUST NOT delegate domain-operation dispatch to the Resource Manager.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  > decision [active]: decision:ADR-0005
  @ SYS-019

SYS-023 [active]: The daemon controller MAY delegate physical process creation to an operating-system supervisor selected by the controller.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-013

## Cross-Module Interfaces

The two target paths summarize `SYS-001`, `SYS-013`, `SYS-014`, and
`SYS-019`:

```text
daemon lifecycle control

CLI -> daemon controller -> foreground daemon runtime
                              -> live authenticated readiness via typed client

hosted domain execution

CLI -> typed daemon client -> typed daemon operation handler
                               -> handler requests lease from Resource Manager
                               -> Resource Manager issues authorized context lease
                               -> handler invokes explicit ContextEngine operation
                               -> handler releases lease
```

Direct library composition begins at Context Engine: a compatible host supplies
the identity, dependencies, ownership declarations, and permitted calls.

## Data And State Model

| Identity or state | Owner | Lifetime |
|---|---|---|
| Context identity | Host supplies; Context Engine binds | One engine lifetime |
| Context selection | Potpie Resource Manager | One lease acquisition |
| Authorized context lease | Potpie Resource Manager | One declared access scope |
| Host-resource binding | Host or Resource Manager | Declared resource lifetime |
| Daemon instance identity | Daemon runtime | One process lifetime |
| Presentation mode | CLI | One invocation |
| Protocol compatibility | Typed client and daemon runtime | One connection or request session |

## Failure Taxonomy

This table summarizes `SYS-007` and does not define additional behavior.

| Category | Canonical origin |
|---|---|
| SelectionError | Potpie context resolution |
| AuthenticationError | Daemon authentication policy |
| AuthorizationError | Potpie authorization policy |
| ResourceLifecycleError | Potpie host-resource management |
| DomainError | Context Engine domain semantics |
| DependencyError | Context Engine port invocation |
| EngineLifecycleError | Context Engine lifetime enforcement |
| ProtocolTransportError | Typed client or daemon protocol boundary |
| DaemonInternalError | Daemon runtime defect boundary |
| PresentationError | CLI-local parsing or rendering |

## Destructive Operation Flow

This sequence summarizes `SYS-009` and `SYS-020`:

```text
human confirmation or explicit automation flag
  -> untrusted typed destructive-intent assertion
  -> daemon authenticates caller
  -> Potpie resolves context and authorizes actor + operation + context
  -> Potpie validates the assertion against the resolved authorization
  -> handler receives validated intent and authorized context lease
  -> ContextEngine receives an explicit destructive domain command
```

## Compatibility, Migration, And Rollout

Later commits migrate the current implementation toward these behaviors.
Temporary compatibility does not become a second permanent architecture. An
intentional observable behavior change follows `PROD-005` before implementation.

## Acceptance Criteria

- The two call paths are reconstructable without implementation code.
- The Resource Manager cannot satisfy the contract by dispatching product
  operations.
- The error and trust-boundary models agree with every module contract.
- Destructive confirmation cannot be mistaken for authentication or
  authorization.
- Current implementation gaps remain observations rather than conformance.

## Current Implementation Gap Snapshot

Observed at base `a341978880b9d4c1b403831931279ccedf6184ae`:

- `potpie/daemon/main.py` is the active launcher target and serves reflective
  `/rpc` and `/attr` endpoints.
- `potpie/daemon/client.py` dynamically mirrors HostShell surfaces.
- `potpie/daemon/rpc.py` places Python module and class identity on the wire.
- `potpie/daemon/runtime/` contains a separate incomplete candidate runtime.
- `potpie_context_engine/host/shell.py` combines domain services with product
  lifecycle, auth, installation, skills, configuration, and pot management.
- `bootstrap/host_wiring.py` composes Potpie and daemon concerns from inside
  the engine package.
- Context Engine still depends on a separately published Context Core package
  and exposes an extension type.
- CLI human and machine rendering are partly centralized, while destructive
  confirmation is not consistently enforced end to end.

These observations explain the migration and do not alter the accepted target.
