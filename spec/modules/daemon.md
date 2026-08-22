---
id: SPEC-DAEMON
title: Potpie Daemon Contract
kind: module-spec
revision: 1
maturity: accepted
owners:
  - team:potpie
depends_on:
  - SPEC-GLOSSARY
  - SPEC-SYSTEM
  - SPEC-POTPIE-RESOURCE-MANAGER
  - SPEC-CONTEXT-ENGINE
change_ref: SPEC-CHANGE-0007
---

# Potpie Daemon Contract

## Purpose

The Potpie daemon subsystem provides a narrow external controller and one
foreground daemon runtime. The controller creates and observes the process.
The runtime owns its authenticated live boundary, typed handlers, discovery,
readiness, lifecycle, and transport behavior.

## Ownership And Boundaries

The daemon controller owns process creation and observation before or outside a
live endpoint. It does not claim readiness or execute domain operations.

The daemon runtime owns:

- One active runtime and entry path.
- Per-boot instance identity and lifetime ownership lock.
- Atomic connection discovery and authenticated live readiness.
- A finite typed operation registry and typed client contract.
- Explicit operation handlers, protocol failures, and observability.
- Runtime state transitions, draining, and shutdown cleanup.

The runtime hosts one narrow Resource Manager. Each typed context-domain
handler acquires an authorized context lease and invokes Context Engine
directly. Control and host-management handlers remain outside that path.
Neither the controller nor runtime owns context-domain semantics or terminal
presentation.

## Scope And Non-Goals

This revision binds the controller/runtime split and process invariants while
leaving transport and wire details open. It does not choose UDS or TCP, freeze
discovery fields, define credential storage, specify cancellation, or choose
which current runtime implementation supplies reusable code.

## Actors And Permissions

| Actor | Interaction |
|---|---|
| CLI | Uses the controller before an endpoint and the typed client for live operations |
| Daemon controller or OS supervisor | Starts and observes one foreground runtime |
| Typed daemon client | Discovers, authenticates, handshakes, and invokes typed operations |
| Typed operation handler | Acquires a lease and invokes one explicit Context Engine operation |
| Resource Manager | Resolves, authorizes, composes, and returns the lease |
| Context Engine | Executes domain semantics for one bound identity |

## Normative Requirements

DAEMON-001 [active]: Potpie MUST have exactly one supported daemon runtime architecture and runtime entry path.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  > observation [active]: code:potpie/daemon/main.py@a341978880b9d4c1b403831931279ccedf6184ae
  > observation [active]: code:potpie/daemon/runtime/__main__.py@a341978880b9d4c1b403831931279ccedf6184ae

DAEMON-002 [active]: The daemon runtime MUST run as one foreground process created directly by the controller or by an operating-system supervisor selected by the controller.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-001

DAEMON-003 [active]: One owning daemon runtime MUST compose and host exactly one logical Potpie Resource Manager.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-002

DAEMON-004 [active]: The daemon runtime MUST expose a finite explicit typed operation surface through a typed daemon client.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-003

DAEMON-005 [active]: The daemon protocol MUST NOT use arbitrary member-name dispatch, reflection RPC, generic attribute forwarding, or cross-wire Python class identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  > observation [active]: code:potpie/daemon/client.py@a341978880b9d4c1b403831931279ccedf6184ae
  > observation [active]: code:potpie/daemon/rpc.py@a341978880b9d4c1b403831931279ccedf6184ae
  @ DAEMON-004

DAEMON-006 [active]: A typed context-operation handler MUST acquire an authorized context lease from the Resource Manager before invoking Context Engine.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-003
  @ RM-008

DAEMON-007 [active]: Every daemon boot MUST generate a unique daemon instance identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

DAEMON-008 [active]: The daemon runtime MUST hold its ownership lock from successful ownership acquisition until entry into stopped releases the runtime scope.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-007

DAEMON-009 [active]: The daemon runtime MUST atomically publish at most one canonical discovery record for its active instance in a Potpie runtime scope.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-007
  @ DAEMON-008

DAEMON-010 [active]: Discovery metadata alone MUST NOT establish daemon readiness.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-009

DAEMON-011 [active]: Daemon readiness MUST be established by a live authenticated handshake.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-010

DAEMON-012 [active]: A process identifier MUST be treated only as diagnostic metadata.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-007

DAEMON-013 [active]: A stale discovery record MUST fail safely because a client cannot establish readiness without a matching live authenticated handshake.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-009
  @ DAEMON-011

DAEMON-014 [active]: A daemon boot MUST distinguish starting, ready, draining, failed, and stopped lifecycle states.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-002

DAEMON-015 [active]: During clean shutdown, the daemon runtime MUST remove the canonical discovery record when it still owns that record.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-009
  @ DAEMON-014

DAEMON-016 [active]: The daemon runtime MUST authenticate the caller before dispatching a protected typed handler.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-006

DAEMON-017 [active]: Possession of daemon discovery metadata MUST NOT authorize a request.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-009
  @ DAEMON-016

DAEMON-018 [active]: The daemon runtime MUST NOT serialize all requests through a process-wide RPC mutex.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  > observation [active]: code:potpie/daemon/main.py@a341978880b9d4c1b403831931279ccedf6184ae

DAEMON-019 [active]: Request coordination MUST derive from typed operation safety, target context, and resource-safety requirements.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-018
  @ RM-013

DAEMON-020 [active]: Independent reads and operations on unrelated contexts MUST be able to progress concurrently when their dependencies permit it.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-019
  @ RM-014

DAEMON-021 [active]: The daemon runtime MUST NOT automatically replay an ambiguous mutating operation after transport failure unless an accepted idempotency contract permits that replay.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-004

DAEMON-022 [active]: AuthenticationError, AuthorizationError, ProtocolTransportError, DaemonInternalError, and lower-boundary SelectionError, ResourceLifecycleError, DomainError, DependencyError, and EngineLifecycleError MUST remain structurally distinguishable at the daemon boundary.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-004
  @ SYS-007

DAEMON-023 [active]: The daemon subsystem MUST NOT emit prompts, terminal colors, progress bars, terminal tables, or process exit codes.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-022

DAEMON-024 [active]: Daemon operational logs MUST exclude credentials and sensitive context payloads by default.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

DAEMON-025 [active]: The daemon subsystem MUST NOT implement or duplicate Context Engine domain semantics.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ RM-017

DAEMON-026 [active]: A destructive typed context-domain handler MUST receive an authorized context lease and destructive intent validated against the authenticated actor, operation, and resolved context before invoking the explicit destructive Context Engine command.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-016
  @ RM-005
  @ CE-017

DAEMON-027 [active]: Every typed product operation MUST be registered to exactly one explicit typed operation handler.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-004

DAEMON-028 [active]: A typed context-domain operation handler MUST invoke one explicitly named Context Engine façade operation itself.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-006
  @ RM-022
  @ CE-020

DAEMON-029 [active]: The daemon transport layer MUST NOT construct Context Engine internals or dependency graphs ad hoc.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-006
  @ RM-007

DAEMON-030 [active]: The daemon controller MUST cause creation of the external foreground runtime, directly or through its selected operating-system supervisor, and observe the resulting process from outside the runtime.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-013
  @ DAEMON-002

DAEMON-031 [active]: Controller observation of a process, PID, or discovery record MUST NOT be represented as daemon readiness.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-010
  @ DAEMON-012
  @ DAEMON-030

DAEMON-032 [active]: A daemon boot MUST permit only these lifecycle transitions: starting to ready, draining, or failed; ready to draining or failed; draining to stopped or failed; and failed to stopped.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-014

DAEMON-033 [active]: A daemon boot MUST NOT return to ready after it leaves ready or enters failed or stopped.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-032

DAEMON-034 [active]: Starting daemon service after a prior boot reaches stopped MUST create a new boot with a new daemon instance identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-007
  @ DAEMON-032

DAEMON-035 [active]: As part of a controlled failed-to-stopped transition, the daemon runtime MUST request release from each resource owner, remove any discovery record it still owns, and release its ownership lock atomically with entry into stopped.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-008
  @ DAEMON-009
  @ DAEMON-032

DAEMON-036 [active]: Completion of the daemon migration MUST remove the non-selected runtime implementation as a supported or dormant alternative architecture.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-001

DAEMON-037 [active]: Completion of the daemon migration MUST remove reflection RPC endpoints, arbitrary-member client dispatch, generic attribute forwarding, and cross-wire Python class codecs.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-005

DAEMON-038 [active]: Completion of the daemon migration MUST remove noncanonical daemon discovery formats and competing discovery writers while retaining exactly one atomic canonical discovery writer.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-009

DAEMON-039 [active]: A temporary daemon compatibility adapter MUST have explicit retirement criteria.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-001
  @ DAEMON-036

DAEMON-040 [active]: A readiness handshake MUST return the expected per-boot daemon instance identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-007
  @ DAEMON-011

DAEMON-041 [active]: A readiness handshake MUST establish compatible protocol semantics.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-011

DAEMON-042 [active]: A readiness handshake MUST report the daemon boot's ready lifecycle state.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-011
  @ DAEMON-014

DAEMON-043 [active]: A typed context-domain handler MUST release its authorized context lease after producing a typed outcome or encountering handler failure.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-006
  @ RM-011

DAEMON-044 [active]: The daemon runtime MUST NOT fork, daemonize, or relaunch itself into a hidden child process.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-002
  @ DAEMON-030

DAEMON-045 [active]: Readiness, runtime-control, and daemon-host-management handlers MUST NOT acquire an authorized context lease or invoke Context Engine.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-004
  @ DAEMON-006
  @ DAEMON-028

DAEMON-046 [active]: A process identifier MUST NOT be treated as daemon instance identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-007
  @ DAEMON-012

DAEMON-047 [active]: A typed context-domain operation handler MUST NOT delegate domain-operation execution to the Resource Manager or authorized context lease.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-028
  @ RM-022

DAEMON-048 [active]: A typed context-domain operation handler MUST NOT construct Context Engine internals or dependency graphs ad hoc.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-006
  @ DAEMON-029

DAEMON-049 [active]: A temporary daemon compatibility adapter MUST NOT define a second permanent daemon architecture.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-036
  @ DAEMON-039

DAEMON-050 [active]: A daemon boot MUST remain failed rather than report stopped while its discovery claim or ownership lock remains held.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-008
  @ DAEMON-015
  @ DAEMON-035

DAEMON-051 [active]: Failure of the daemon controller to create or observe the configured runtime process MUST return ResourceLifecycleError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-030
  @ SYS-007

## Lifecycle Model

The following state and transition tables summarize `DAEMON-014`,
`DAEMON-032`, `DAEMON-033`, and `DAEMON-035`.

| State | Meaning |
|---|---|
| starting | The boot owns or is acquiring its runtime scope; initialization is incomplete |
| ready | A live authenticated handshake confirms that permitted work can be accepted |
| draining | New ordinary work is restricted while owned in-flight work settles or terminates |
| failed | The boot cannot serve and failure cleanup is in progress |
| stopped | The boot is terminal and owns no runtime scope |

| From | Allowed next states |
|---|---|
| starting | ready, draining, failed |
| ready | draining, failed |
| draining | stopped, failed |
| failed | stopped |
| stopped | none for that boot identity |

A restart after `stopped` creates a new boot and instance identity.

## Handler And Lease Flow

The typed catalog separates daemon-control handlers, Potpie host or resource
handlers, and context-domain handlers. Only context-domain handlers follow the
lease-to-engine portion of this flow, as defined by `DAEMON-006`,
`DAEMON-028`, and `DAEMON-045`.

```text
typed request
  -> authenticate caller
  -> select explicit typed handler
  -> Resource Manager resolves and authorizes
  -> acquire authorized context lease
  -> handler invokes explicit ContextEngine operation
  -> preserve typed outcome
  -> release lease
```

The Resource Manager never receives the domain invocation or its result.
Readiness and runtime-control handlers remain outside the context lease path.
Abrupt process termination relies on operating-system lock release and
authenticated-handshake stale-record safety rather than a controlled
failed-to-stopped transition.

## Failure Summary

| Condition | Typed interpretation |
|---|---|
| Missing or stale discovery | ProtocolTransportError with safe retry posture |
| Rejected caller identity | AuthenticationError |
| Denied actor-operation-context scope | AuthorizationError |
| Non-ready live endpoint | ProtocolTransportError with lifecycle detail |
| Resource Manager failure | Preserve SelectionError, AuthorizationError, or ResourceLifecycleError |
| Context Engine failure | Preserve DomainError, DependencyError, or EngineLifecycleError |
| Unexpected runtime defect | DaemonInternalError with redacted correlation detail |
| Disconnect during mutation | ProtocolTransportError with unknown outcome unless separately proven |

## Compatibility, Migration, And Rollout

The current reflection daemon and incomplete candidate runtime are migration
inputs. Which source supplies reusable implementation and the removal order
remain deferred; the accepted single-stack, non-reflective deletion end state
is binding.

## Acceptance Criteria

- The controller and runtime responsibilities cannot be confused.
- A reader cannot satisfy the contract with reflection or generic service
  mirroring.
- Typed handlers, not the Resource Manager, invoke Context Engine.
- Discovery, PID, instance identity, and readiness remain distinct.
- The lifecycle transition graph is complete and restart creates a new identity.
- The target cannot retain a dormant competing runtime or duplicate discovery.
- Concurrency is scoped and ambiguous mutations are not silently replayed.
- No domain or terminal behavior belongs to the daemon subsystem.

## Implementation Notes

At base `a341978880b9d4c1b403831931279ccedf6184ae`,
`potpie.daemon.main` is the launched reflection server and
`potpie.daemon.runtime` is an incomplete separate candidate. Neither is
declared conforming.
