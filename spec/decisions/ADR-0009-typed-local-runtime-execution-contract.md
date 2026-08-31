---
id: ADR-0009
title: Define The Typed Local Runtime Execution Contract
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-20T14:09:34+05:30
---

# ADR-0009: Define The Typed Local Runtime Execution Contract

## Context

The accepted runtime boundary requires an external controller, one foreground
daemon, an authenticated live handshake, one canonical discovery writer,
finite typed operations, authorized context leases, scoped coordination, and
structured failures. ADR-0007 selects a replacement runtime and same-PR
migration, but exact lifetime, controller, transport, envelope, concurrency,
and destructive-intent choices remain deferred.

The replacement foundations need those choices before they can be implemented
without recreating a service locator, global mutex, discovery-as-readiness, or
automatic mutation replay.

## Decision

### Resource Manager lifetime and lease

`ContextResourceManager` performs selection, authentication, authorization,
resource composition, and engine acquisition in that order. Acquisition returns
`Success[AuthorizedContextLease]` or a typed selection, authorization, or
resource-lifecycle failure.

`AuthorizedContextLease` is an asynchronous lease containing the resolved
`ContextIdentity`, authenticated actor-operation-context scope, context-bound
`ContextEngine`, and explicit ownership metadata. It exposes idempotent release
and no domain-operation dispatch or arbitrary service lookup.

Compatible engines are cached by `(ContextIdentity, CompositionFingerprint)`
until daemon shutdown. Releasing a request lease reduces active use but does not
close the cached engine. Shutdown stops new acquisition, drains active leases,
closes cached engines, and releases owned dependencies in deterministic reverse
acquisition order.

On acceptance, this part resolves `OQ-RM-LIFETIME-001`.

### Controller and foreground process

`DaemonController` directly starts and observes the canonical foreground child
process. It does not delegate to an operating-system supervisor in this
migration. The controller owns pre-endpoint start, stop, restart, status, logs,
failure reporting, and stale-process cleanup; it does not serve domain
operations or claim readiness from a PID or file.

Normal stop uses the authenticated typed daemon-control operation. If no live
endpoint can be authenticated, the controller uses bounded operating-system
termination and then reconciles stale runtime records.

On acceptance, this part resolves `OQ-DAEMON-CONTROLLER-001`.

### Local transport, discovery, and authentication

The canonical transport is HTTP over an owner-only Unix-domain socket on
supported POSIX systems. Platforms without supported UDS behavior use an
authenticated loopback TCP endpoint. Remote network binding is not supported.

Each daemon boot generates a cryptographically random 256-bit bearer secret.
The same authentication scheme applies to UDS and TCP. The secret is stored in
a separate owner-only credential file and is compared in constant time. The
owner-only canonical `discovery.json` contains only:

- Discovery schema version.
- Daemon instance identity and PID.
- Transport kind and endpoint.
- Supported protocol range.
- Authentication scheme and credential-file reference, never the bearer
  secret itself.

The discovery record is written atomically after the endpoint can accept an
authenticated handshake and is removed during owned shutdown. Possessing the
discovery metadata alone cannot authorize a request.

On acceptance, this part resolves `OQ-DAEMON-TRANSPORT-001`.

### Versioned operation envelope and handshake

Protocol version 1 uses one finite discriminated request union at
`POST /v1/operations`. Every request contains:

- `protocol_version`.
- Unique `request_id`.
- Finite `operation` discriminator.
- Exact context-selection input when the operation requires context.
- Operation-specific typed `payload`.
- Optional typed `destructive_intent`.

Every response to a structurally valid envelope repeats `protocol_version` and
`request_id` and contains exactly one `Success[result]` or `Failure[error]`. If
the envelope cannot supply a valid request identity, the runtime assigns the
response correlation ID and returns a typed malformed-envelope failure. Error
payloads contain a canonical category, stable code, safe message, optional
structured details, recommended next action, and retry posture. Human
presentation and tracebacks are not protocol fields.

The authenticated `daemon.handshake` operation returns the supported protocol
range, current instance identity, `ready` lifecycle state, capability set, and
operation-catalog fingerprint. A client must complete a compatible handshake
before sending domain or administrative operations. `daemon.shutdown` is a
separate typed control discriminator rather than process reflection.

Authentication failures, incompatible versions, malformed envelopes, unknown
operations, and runtime defects use the same response envelope and canonical
error families. HTTP status communicates transport classification but never
replaces the typed response.

On acceptance, this part resolves `OQ-DAEMON-PROTOCOL-001`.

### Clients and handler boundary

`EngineClient` mirrors the accepted public Context Engine method catalog and
typed outcomes. A client instance is bound to one exact context-selection
input:

- `LocalEngineClient` invokes the same typed operation handlers and Resource
  Manager policy without transport.
- `DaemonEngineClient` adds the bound selector to each authenticated protocol
  request and translates only typed envelopes.

Potpie-owned administrative behavior uses a separate finite typed admin
operation family where daemon execution is required. It is not added to
`EngineClient` or `ContextEngine`.

No final client performs protocol auto-detection, reflective fallback, dynamic
member forwarding, or old-to-new protocol translation.

### Operation safety and conflict keys

Every registered operation declares one safety class and the data needed to
derive its conflict keys:

| Safety class | Coordination rule |
|---|---|
| shared context read | Reads for the same context may run together and conflict with an exclusive mutation for that context. |
| exclusive context mutation | Conflicts with reads and mutations for the same context; unrelated contexts may proceed. |
| exclusive resource mutation | Conflicts only on the same typed resource identity and any explicitly affected context. |
| daemon lifecycle control | Process-exclusive and ordered with draining and shutdown. |

Conflict keys derive from the typed operation, resolved context identity, and,
where applicable, resource type and resource identity. The coordinator does not
use one process-wide domain-operation mutex.

On acceptance, this part resolves `OQ-DAEMON-CONCURRENCY-001`.

### Destructive intent

A destructive request carries this untrusted assertion inside its authenticated
request envelope:

```text
confirmed = true
operation = exact enclosing operation discriminator
selector = exact enclosing context-selection input
request_id = exact enclosing request identity
```

After authenticating the caller and resolving and authorizing the context, the
Resource Manager compares every assertion field with the enclosing request and
resolved authorized scope. Missing or mismatched intent returns
`AuthorizationError` and no lease. The assertion is not signed, is not a
general authorization token, and is never replayed automatically.

On acceptance, this part resolves `OQ-DESTRUCTIVE-INTENT-001`.

### Retry, disconnect, and deferred behavior

The typed client performs no automatic mutation replay. A transport failure
after a mutating request may have been dispatched returns
`ProtocolTransportError` with unknown-outcome retry posture rather than
reissuing the request.

This decision does not add protocol cancellation, operation handles, automatic
read retries, or mutation idempotency keys. `OQ-DAEMON-CANCEL-001` and
`OQ-DAEMON-IDEMPOTENCY-001` remain deferred until such behavior is proposed.
`OQ-CLI-JSON-001` also remains deferred; existing machine output and exit
categories are preserved by adapters outside the protocol.

## Consequences

- Engine reuse is deterministic for one daemon boot without defining idle
  eviction.
- One controller implementation is sufficient for this migration; supervisor
  integration can be added only through a later decision.
- UDS and TCP exercise one authenticated protocol and outcome model.
- Discovery remains connection metadata rather than a bearer credential or a
  readiness claim.
- Local and daemon execution share handler, selection, authorization, and lease
  policies.
- Independent contexts and resources can progress without a global request
  lock.
- Ambiguous mutations are surfaced rather than replayed.

## Alternatives Considered

### Close an engine when its last request lease ends

This minimizes retained resources but adds construction churn and makes
request release determine engine lifetime during the migration.

### Add bounded idle eviction now

This introduces timers, races, and shutdown policy before baseline lifetime
behavior has migrated.

### Add supervisor-backed process creation now

This broadens platform integration without changing the accepted controller
boundary needed for the local CLI.

### Use loopback TCP on every platform

This is portable but gives up the filesystem permissions and local endpoint
shape available on supported POSIX systems.

### Store the bearer secret inside discovery.json

This would make possession of discovery metadata sufficient to authorize a
request and would contradict `DAEMON-017`.

### Use per-operation routes or reflection

Per-operation routes duplicate protocol framing; reflection reintroduces the
internal object graph and arbitrary member dispatch prohibited by the accepted
contract.

### Sign destructive intent separately

The request is already authenticated locally. A second signed token adds key
and replay lifecycle that is unnecessary when the assertion is bound to the
same request and never retried automatically.

## Contract And Impact Review

No accepted behavior changes. This decision selects details explicitly
deferred by the accepted contracts and requires no contract revision or
spec-change record.

| Artifact or area | Required change after acceptance | No-change reason |
|---|---|---|
| SPEC-POTPIE-RESOURCE-MANAGER | — | `RM-001` through `RM-016`, `RM-020` through `RM-035` already permit and constrain the selected lease and lifecycle policy. |
| SPEC-DAEMON | — | The selected controller, transport, readiness, coordination, failure, and shutdown details satisfy the existing daemon behaviors. |
| SPEC-CLI | — | Client/controller selection and destructive confirmation are already binding; exact machine presentation remains deferred. |
| SPEC-CONTEXT-ENGINE | — | The Resource Manager supplies one authorized context-bound engine without changing domain semantics. |
| SPEC-SYSTEM | — | Ownership, trust, error origins, and direct/hosted call paths remain unchanged. |
| ADR-0001 through ADR-0008 | — | Existing accepted decisions remain immutable; ADR-0008 is independently accepted or rejected. |
| SPEC-CHANGE-0001 through SPEC-CHANGE-0008 | — | No behavior operation or contract revision occurs. |
| Existing conformance | — | No conformance records exist and this decision makes no implementation claim. |

## Authority And Sources

> decision [active]: decision:ADR-0004
> decision [active]: decision:ADR-0005
> decision [active]: decision:ADR-0006
> decision [active]: decision:ADR-0007
> observation [active]: code:potpie/daemon/main.py@5a54cbb2060b67f43718bf47e6e453bebc598325
> observation [active]: code:potpie/daemon/client.py@5a54cbb2060b67f43718bf47e6e453bebc598325
> observation [active]: code:potpie/daemon/process/launcher.py@5a54cbb2060b67f43718bf47e6e453bebc598325
> observation [active]: code:potpie/daemon/runtime/shell.py@5a54cbb2060b67f43718bf47e6e453bebc598325

## Acceptance

Accepted by `user:dsantra` at `2026-08-20T14:09:34+05:30`. This acceptance
resolves `OQ-RM-LIFETIME-001`, `OQ-DAEMON-CONTROLLER-001`,
`OQ-DAEMON-TRANSPORT-001`, `OQ-DAEMON-PROTOCOL-001`,
`OQ-DAEMON-CONCURRENCY-001`, and `OQ-DESTRUCTIVE-INTENT-001`. It does not
claim implementation or verification conformance.
